import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torchvision import transforms
from tqdm import tqdm

# 添加可视化相关导入
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import matplotlib.patches as patches

# print(torch.cuda.is_available())
from dataloaders.dataset import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment_Ours,
)
sys.path.append(r'E:\1_master_degree_project\LiShuai\LGAELLH0622\networks')
from net_factory import net_factory
from unet import UNet_LGAE1
from unet_de import UNet_LDMV2
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume_refinev2 as test_single_volume
from val_2D import test_single_volume_refinev3
from PIL import Image


import torch
import numpy as np

def add_noise(image, mode='gaussian', gaussian_std=0.0, sp_prob=0.0):
    """
    image: torch.Tensor or np.ndarray, 任意形状
    返回: torch.Tensor, 值域 [0,1]
    """
    # 1. 统一转成 float32 Tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    else:
        image = image.clone().float()

    # 2. 归一化到 [0,1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    if mode == 'gaussian':
        noise = torch.randn_like(image) * gaussian_std
        image = image + noise
    elif mode == 'saltpepper':
        mask = torch.rand_like(image)
        image[mask < sp_prob / 2] = 0
        image[(mask >= sp_prob / 2) & (mask < sp_prob)] = 1

    # 3. 再次 clamp，防止浮点误差
    image = torch.clamp(image, 0, 1)
    return image

# 在文件开头添加GradCAM相关导入
import cv2
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F

# ===============================================================================================

class NodeLevelGradCAM:
    """为LGAE的每个节点生成GradCAM"""
    
    def __init__(self, model, target_layer_name='lgae1'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.node_gradients = {}
        self.node_activations = {}
        self.handles = []
        
    def register_hooks_for_node(self, node_idx):
        """为特定节点注册钩子"""
        self.clear_hooks()
        
        def forward_hook(module, input, output):
            # LGAE返回 (enhanced_features, graph_info)
            if isinstance(output, tuple):
                enhanced_features, graph_info = output
                # 保存节点特征
                if 'node_features' in graph_info:
                    # node_features shape: (batch, num_nodes, channels)
                    self.node_activations[node_idx] = graph_info['node_features'][:, node_idx, :].clone()
                # 保存attention weights
                if 'attention_weights' in graph_info:
                    self.node_activations['attention'] = graph_info['attention_weights'].clone()
                # 保存增强后的特征
                self.node_activations['enhanced'] = enhanced_features.clone()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad = grad_output[0]
            else:
                grad = grad_output
            self.node_gradients[node_idx] = grad.clone()
        
        # 找到目标层
        target_layer = getattr(self.model, self.target_layer_name)
        self.handles.append(target_layer.register_forward_hook(forward_hook))
        self.handles.append(target_layer.register_full_backward_hook(backward_hook))
    
    def clear_hooks(self):
        """清除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.node_gradients.clear()
        self.node_activations.clear()
    
    def generate_node_cam(self, input_image, node_idx, target_class=1):
        """为特定节点生成CAM"""
        # 注册节点钩子
        self.register_hooks_for_node(node_idx)
        
        # 需要梯度
        input_image = input_image.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_image, save_graph_evolution=True)
        
        # 选择目标
        if len(output.shape) == 4:
            class_output = output[:, target_class, :, :]
            score = class_output.mean()
        else:
            score = output.mean()
        
        # 反向传播
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # 生成CAM
        cam = None
        if node_idx in self.node_gradients and 'enhanced' in self.node_activations:
            gradient = self.node_gradients[node_idx]
            activation = self.node_activations['enhanced']
            
            # 计算权重
            if gradient.dim() == 4:
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activation).sum(dim=1, keepdim=True)
            
            # ReLU和归一化
            cam = F.relu(cam)
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # 上采样
            cam = F.interpolate(cam, size=input_image.shape[-2:], 
                              mode='bilinear', align_corners=False)
        cam = cam.detach()
        self.clear_hooks()
        return cam
    
    def cleanup(self):
        """清理"""
        self.clear_hooks()


def visualize_node_gradcam_evolution(model, dataloader, save_dir, num_samples=3,
                                    lgae_configs={'lgae1': 12, 'lgae2': 16, 'lgae3': 20, 'lgae4': 16}):
    """
    可视化每个LGAE层中每个节点的GradCAM热力图
    
    Args:
        lgae_configs: 字典，包含每个LGAE层的节点数
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    samples_processed = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        # 处理数据
        if isinstance(batch_data, dict):
            images = batch_data['image']
            labels = batch_data.get('label')
            
            # 处理3D volume
            if len(images.shape) == 4 and images.shape[1] > 1:
                mid_slice = images.shape[1] // 2
                images = images[:, mid_slice:mid_slice+1, :, :]
                if labels is not None:
                    labels = labels[:, mid_slice, :, :]
            
            images = images.float().cuda()
            if labels is not None:
                labels = labels.cuda()
        else:
            continue
        
        # 确保单通道
        if images.shape[1] != 1:
            images = images[:, 0:1, :, :]
        
        try:
            # 获取预测
            with torch.no_grad():
                pred_output = model(images)
                pred_mask = torch.argmax(pred_output, dim=1)
                num_classes = pred_output.shape[1]
            
            # 原始图像
            img_np = images[0, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # 为每个LGAE层创建可视化
            for lgae_name, num_nodes in lgae_configs.items():
                print(f"\nProcessing {lgae_name} with {num_nodes} nodes...")
                
                # 创建节点级GradCAM
                node_gradcam = NodeLevelGradCAM(model, target_layer_name=lgae_name)
                
                # 计算所有节点的CAM
                node_cams = []
                for node_idx in range(num_nodes):
                    # 为每个类别生成CAM
                    class_cams = []
                    for target_class in range(1, min(num_classes, 4)):
                        cam = node_gradcam.generate_node_cam(images, node_idx, target_class)
                        if cam is not None:
                            # class_cams.append(cam[0, 0].cpu().numpy())
                            class_cams.append(cam[0, 0].detach().cpu().numpy())
                    
                    if class_cams:
                        # 平均所有类别的CAM
                        avg_cam = np.mean(class_cams, axis=0)
                        node_cams.append(avg_cam)
                    else:
                        node_cams.append(np.zeros_like(img_np))
                
                # 创建可视化
                visualize_node_attention_patterns(
                    img_np, node_cams, lgae_name, num_nodes,
                    save_path=os.path.join(save_dir, f'node_gradcam_{lgae_name}_sample_{samples_processed+1}.png')
                )
                
                # 创建节点激活强度矩阵
                create_node_activation_matrix(
                    node_cams, lgae_name,
                    save_path=os.path.join(save_dir, f'node_matrix_{lgae_name}_sample_{samples_processed+1}.png')
                )
                
                node_gradcam.cleanup()
            
            print(f"✓ Completed sample {samples_processed + 1}")
            
        except Exception as e:
            print(f"✗ Error processing sample {samples_processed + 1}: {e}")
            import traceback
            traceback.print_exc()
        
        samples_processed += 1
    
    print(f"\nAll samples processed!")


def visualize_all_layers_top_nodes(model, dataloader, save_dir, num_samples=3,
                                   lgae_configs={'lgae1': 12, 'lgae2': 16, 'lgae3': 20, 'lgae4': 16}):
    """
    为所有LGAE层创建综合的top节点可视化
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    samples_processed = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        # 处理数据
        if isinstance(batch_data, dict):
            images = batch_data['image']
            labels = batch_data.get('label')
            
            if len(images.shape) == 4 and images.shape[1] > 1:
                mid_slice = images.shape[1] // 2
                images = images[:, mid_slice:mid_slice+1, :, :]
            
            images = images.float().cuda()
        else:
            continue
        
        if images.shape[1] != 1:
            images = images[:, 0:1, :, :]
        
        try:
            with torch.no_grad():
                pred_output = model(images)
                num_classes = pred_output.shape[1]
            
            # 原始图像
            img_np = images[0, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # 创建4层综合图
            fig, axes = plt.subplots(4, 7, figsize=(21, 12))
            
            # 第一列显示原始图像
            for row in range(4):
                axes[row, 0].imshow(img_np, cmap='gray')
                axes[row, 0].set_title(f'lgae{row+1}', fontsize=14, fontweight='bold')
                axes[row, 0].axis('off')
            
            # 为每个LGAE层处理
            for layer_idx, (lgae_name, num_nodes) in enumerate(lgae_configs.items()):
                print(f"Processing {lgae_name}...")
                
                node_gradcam = NodeLevelGradCAM(model, target_layer_name=lgae_name)
                
                # 计算所有节点的CAM
                node_cams_with_strength = []
                for node_idx in range(num_nodes):
                    cam_list = []
                    for target_class in range(1, min(num_classes, 4)):
                        cam = node_gradcam.generate_node_cam(images, node_idx, target_class)
                        if cam is not None:
                            cam_list.append(cam[0, 0].detach().cpu().numpy())
                    
                    if cam_list:
                        avg_cam = np.mean(cam_list, axis=0)
                        mean_activation = np.mean(avg_cam)
                        node_cams_with_strength.append((node_idx, mean_activation, avg_cam))
                
                # 选取top 6
                node_cams_with_strength.sort(key=lambda x: x[1], reverse=True)
                top_6 = node_cams_with_strength[:6]
                
                # 在对应行显示top 6节点
                for col_idx, (node_idx, strength, cam) in enumerate(top_6):
                    ax = axes[layer_idx, col_idx + 1]
                    
                    # 增强可视化
                    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    threshold = 0.25
                    cam_thresholded = cam_normalized.copy()
                    cam_thresholded[cam_thresholded < threshold] = 0
                    
                    cam_colored = plt.cm.hot(cam_thresholded)[:, :, :3]
                    overlay = 0.6 * np.stack([img_np]*3, axis=-1) + 0.4 * cam_colored
                    overlay = np.clip(overlay, 0, 1)
                    
                    ax.imshow(overlay)
                    
                    # 添加大号节点编号
                    ax.text(5, 20, f'{node_idx+1}', 
                           fontsize=20, fontweight='bold', color='yellow',
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor='red', alpha=0.8))
                    
                    ax.set_title(f'N{node_idx+1}', fontsize=11)
                    ax.axis('off')
                
                node_gradcam.cleanup()
            
            plt.suptitle(f'Sample {samples_processed + 1}: Top 6 Active Nodes per Layer', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'all_layers_top_nodes_sample_{samples_processed+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved comprehensive visualization to {save_path}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            plt.close('all')
        
        samples_processed += 1
        torch.cuda.empty_cache()



def visualize_node_attention_patterns(img_np, node_cams, layer_name, num_nodes, save_path):
    """
    可视化激活最强的6个节点的注意力模式
    """
    # 计算每个节点的平均激活强度
    node_activations = []
    for idx, cam in enumerate(node_cams):
        mean_activation = np.mean(cam)
        node_activations.append((idx, mean_activation, cam))
    
    # 按激活强度排序，选取最强的6个
    node_activations.sort(key=lambda x: x[1], reverse=True)
    top_6_nodes = node_activations[:6]
    
    # 创建2x3的网格
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (node_idx, activation_strength, cam) in enumerate(top_6_nodes):
        ax = axes[i]
        
        # 创建叠加图，增强对比度
        # 归一化CAM以增强可视化效果
        cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # 应用阈值，只显示强激活区域
        threshold = 0.3  # 只显示超过30%强度的激活
        cam_thresholded = cam_normalized.copy()
        cam_thresholded[cam_thresholded < threshold] = 0
        
        # 使用更鲜艳的颜色映射
        cam_colored = plt.cm.hot(cam_thresholded)[:, :, :3]
        
        # 创建叠加图
        overlay = 0.6 * np.stack([img_np]*3, axis=-1) + 0.4 * cam_colored
        overlay = np.clip(overlay, 0, 1)
        
        im = ax.imshow(overlay)
        
        # 添加节点编号（大字体）和激活强度
        ax.text(10, 30, f'{node_idx+1}', 
                fontsize=24, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
        
        # 显示激活强度值
        ax.text(10, overlay.shape[0]-10, f'Act: {activation_strength:.3f}', 
                fontsize=10, color='white',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
        
        ax.set_title(f'Node {node_idx+1} (Rank #{i+1})', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'{layer_name} - Top 6 Most Active Nodes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()




def create_node_activation_matrix(node_cams, layer_name, save_path):
    """
    创建节点激活强度矩阵热力图
    """
    if not node_cams:
        return
    
    # 将每个CAM划分为网格区域
    grid_size = 8  # 8x8网格
    h, w = node_cams[0].shape
    h_step, w_step = h // grid_size, w // grid_size
    
    # 计算每个节点在每个网格区域的平均激活
    num_nodes = len(node_cams)
    activation_matrix = np.zeros((num_nodes, grid_size * grid_size))
    
    for node_idx, cam in enumerate(node_cams):
        for i in range(grid_size):
            for j in range(grid_size):
                region = cam[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                activation_matrix[node_idx, i*grid_size + j] = region.mean()
    
    # 创建热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 节点-区域激活矩阵
    im1 = ax1.imshow(activation_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('Spatial Region (8x8 grid)', fontsize=11)
    ax1.set_ylabel('Node Index', fontsize=11)
    ax1.set_title(f'{layer_name} - Node Activation Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Activation Strength')
    
    # 添加网格线
    ax1.set_xticks(np.arange(0, grid_size*grid_size, grid_size))
    ax1.set_yticks(np.arange(num_nodes))
    ax1.set_yticklabels([f'N{i+1}' for i in range(num_nodes)])
    ax1.grid(True, which='major', alpha=0.3)
    
    # 2. 节点相似度矩阵
    # 计算节点间的相似度
    from scipy.spatial.distance import cosine
    similarity_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity = 1 - cosine(activation_matrix[i], activation_matrix[j])
                similarity_matrix[i, j] = max(0, similarity)
    
    im2 = ax2.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
    ax2.set_xlabel('Node Index', fontsize=11)
    ax2.set_ylabel('Node Index', fontsize=11)
    ax2.set_title(f'{layer_name} - Node Similarity Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Cosine Similarity')
    
    # 添加数值标注（仅对小矩阵）
    if num_nodes <= 12:
        for i in range(num_nodes):
            for j in range(num_nodes):
                text = ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def analyze_node_specialization(model, dataloader, save_dir, num_samples=10):
    """
    分析节点的专业化程度 - 哪些节点专注于特定类别
    """
    os.makedirs(save_dir, exist_ok=True)
    
    lgae_configs = {'lgae1': 12, 'lgae2': 16, 'lgae3': 20, 'lgae4': 16}
    
    # 收集所有样本的节点激活统计
    node_class_activations = {
        layer: {cls: [[] for _ in range(num_nodes)] 
               for cls in range(1, 4)}  # 3个类别
        for layer, num_nodes in lgae_configs.items()
    }
    
    model.eval()
    samples_processed = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        # 处理数据（同前）...
        if isinstance(batch_data, dict):
            images = batch_data['image']
            if len(images.shape) == 4 and images.shape[1] > 1:
                mid_slice = images.shape[1] // 2
                images = images[:, mid_slice:mid_slice+1, :, :]
            images = images.float().cuda()
        else:
            continue
        
        if images.shape[1] != 1:
            images = images[:, 0:1, :, :]
        
        # 为每个层和节点收集激活
        for lgae_name, num_nodes in lgae_configs.items():
            node_gradcam = NodeLevelGradCAM(model, target_layer_name=lgae_name)
            
            for node_idx in range(num_nodes):
                for target_class in range(1, 4):  # 3个类别
                    cam = node_gradcam.generate_node_cam(images, node_idx, target_class)
                    if cam is not None:
                        mean_activation = cam.detach().mean().item()
                        node_class_activations[lgae_name][target_class][node_idx].append(mean_activation)
            
            node_gradcam.cleanup()
        
        samples_processed += 1
    
    # 创建专业化分析图
    create_specialization_plots(node_class_activations, lgae_configs, save_dir)


def create_specialization_plots(node_class_activations, lgae_configs, save_dir):
    """
    创建节点专业化分析图
    """
    for layer_name, num_nodes in lgae_configs.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 节点-类别激活热力图
        activation_matrix = np.zeros((num_nodes, 3))
        for cls in range(1, 4):
            for node_idx in range(num_nodes):
                activations = node_class_activations[layer_name][cls][node_idx]
                if activations:
                    activation_matrix[node_idx, cls-1] = np.mean(activations)
        
        ax = axes[0, 0]
        im = ax.imshow(activation_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel('Node', fontsize=11)
        ax.set_title(f'{layer_name} - Node-Class Activation', fontsize=12)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
        ax.set_yticks(range(num_nodes))
        ax.set_yticklabels([f'N{i+1}' for i in range(num_nodes)])
        plt.colorbar(im, ax=ax)
        
        # 2. 节点专业化指数
        ax = axes[0, 1]
        specialization_index = []
        for node_idx in range(num_nodes):
            node_activations = activation_matrix[node_idx]
            if node_activations.sum() > 0:
                # 计算熵作为专业化指数
                probs = node_activations / (node_activations.sum() + 1e-8)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                specialization = 1 - entropy / np.log(3)  # 归一化
            else:
                specialization = 0
            specialization_index.append(specialization)
        
        bars = ax.bar(range(num_nodes), specialization_index, color='steelblue')
        ax.set_xlabel('Node Index', fontsize=11)
        ax.set_ylabel('Specialization Index', fontsize=11)
        ax.set_title(f'{layer_name} - Node Specialization', fontsize=12)
        ax.set_xticks(range(0, num_nodes, max(1, num_nodes//10)))
        ax.grid(True, alpha=0.3)
        
        # 3. 类别覆盖率
        ax = axes[1, 0]
        class_coverage = []
        for cls in range(1, 4):
            active_nodes = sum(1 for node_idx in range(num_nodes) 
                             if activation_matrix[node_idx, cls-1] > 0.1)
            class_coverage.append(active_nodes / num_nodes * 100)
        
        bars = ax.bar(['Class 1', 'Class 2', 'Class 3'], class_coverage, 
                      color=['red', 'green', 'blue'])
        ax.set_ylabel('Node Coverage (%)', fontsize=11)
        ax.set_title(f'{layer_name} - Class Coverage', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 4. 节点聚类（基于激活模式）
        ax = axes[1, 1]
        from sklearn.cluster import KMeans
        n_clusters = min(4, num_nodes // 3)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(activation_matrix)
            
            scatter = ax.scatter(range(num_nodes), specialization_index, 
                               c=clusters, cmap='viridis', s=100)
            ax.set_xlabel('Node Index', fontsize=11)
            ax.set_ylabel('Specialization Index', fontsize=11)
            ax.set_title(f'{layer_name} - Node Clusters', fontsize=12)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.suptitle(f'{layer_name} Node Specialization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'node_specialization_{layer_name}.png'), 
                   dpi=120, bbox_inches='tight')
        plt.close()



# ===============================================================================================






def visualize_gradcam_during_training(model, image, label, iter_num, save_dir, 
                                     target_layers=['lgae1', 'lgae2', 'lgae3', 'lgae4']):
    """
    训练过程中可视化单个批次的GradCAM（极简内存版本）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存当前训练状态
    was_training = model.training
    save_path = None
    
    try:
        # 创建一个简单的GradCAM可视化
        model.eval()
        
        # 使用checkpoint技术减少内存使用
        with torch.no_grad():
            # 降低分辨率以减少内存使用
            img_input = F.interpolate(image[:1].clone(), size=(128, 128), mode='bilinear', align_corners=False)
            img_input = img_input.detach().cuda()
        
        # 简单的前向传播获取特征图
        feature_maps = {}
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    feature_maps[name] = output[0].detach().cpu()
                else:
                    feature_maps[name] = output.detach().cpu()
            return hook
        
        # 注册钩子
        hooks = []
        for layer_name in target_layers:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                hooks.append(layer.register_forward_hook(get_activation(layer_name)))
        
        # 前向传播（不需要梯度）
        with torch.no_grad():
            _ = model(img_input)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 创建可视化
        num_layers = len([k for k in feature_maps.keys()])
        if num_layers == 0:
            return None
            
        fig, axes = plt.subplots(2, num_layers + 1, figsize=(4*(num_layers+1), 8))
        
        if len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        # 原始图像
        img_orig = image[0, 0].detach().cpu().numpy()
        img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min() + 1e-8)
        
        axes[0, 0].imshow(img_orig, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        if label is not None:
            lbl = label[0].detach().cpu().numpy()
            axes[1, 0].imshow(lbl, cmap='tab10')
            axes[1, 0].set_title('Label')
        else:
            axes[1, 0].imshow(img_orig, cmap='gray')
            axes[1, 0].set_title('Input')
        axes[1, 0].axis('off')
        
        # 显示特征图（简化版本，不是真正的GradCAM）
        for idx, (layer_name, features) in enumerate(feature_maps.items()):
            if idx + 1 >= axes.shape[1]:
                break
                
            # 取特征图的平均作为简单的注意力图
            if len(features.shape) == 4:
                attention = features[0].mean(dim=0).numpy()
            elif len(features.shape) == 3:
                attention = features.mean(dim=0).numpy()
            else:
                attention = features[0, 0].numpy() if len(features.shape) > 1 else features.numpy()
            
            # 归一化
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            # 上采样到原始大小
            attention_resized = cv2.resize(attention, (img_orig.shape[1], img_orig.shape[0]))
            
            # 显示注意力图
            im = axes[0, idx + 1].imshow(attention_resized, cmap='jet', vmin=0, vmax=1)
            axes[0, idx + 1].set_title(f'{layer_name} Features')
            axes[0, idx + 1].axis('off')
            
            # 叠加图
            attention_colored = plt.cm.jet(attention_resized)[:, :, :3]
            overlayed = 0.7 * np.stack([img_orig, img_orig, img_orig], axis=-1) + 0.3 * attention_colored
            overlayed = np.clip(overlayed, 0, 1)
            
            axes[1, idx + 1].imshow(overlayed)
            axes[1, idx + 1].set_title(f'{layer_name} Overlay')
            axes[1, idx + 1].axis('off')
        
        plt.suptitle(f'Feature Visualization at Iteration {iter_num}')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'features_iter_{iter_num:06d}.png')
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close('all')
        
        # 清理
        del feature_maps
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        plt.close('all')
        torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    
    return save_path



class ImprovedGradCAMForSegmentation:
    """改进的分割任务GradCAM - 使用引导式梯度和聚焦机制"""
    
    def __init__(self, model, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0]
                else:
                    self.activations[name] = output
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    self.gradients[name] = grad_output[0]
                else:
                    self.gradients[name] = grad_output
            return hook
        
        for name in self.target_layers:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                self.handles.append(layer.register_forward_hook(forward_hook(name)))
                self.handles.append(layer.register_full_backward_hook(backward_hook(name)))
    
    def generate_cam_for_class_guided(self, input_image, target_class=1, use_pred_mask=True):
        """
        生成引导式的类别特定CAM
        
        Args:
            input_image: 输入图像
            target_class: 目标类别
            use_pred_mask: 是否使用预测mask来引导
        """
        self.activations.clear()
        self.gradients.clear()
        
        input_image = input_image.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_image)  # (B, C, H, W)
        
        # 获取预测mask
        with torch.no_grad():
            pred_probs = F.softmax(output, dim=1)
            pred_mask = torch.argmax(output, dim=1)
            class_mask = (pred_mask == target_class).float()
        
        # 选择目标：使用三种策略的组合
        if len(output.shape) == 4:
            class_logits = output[:, target_class, :, :]
            
            if use_pred_mask and class_mask.sum() > 0:
                # 策略1：只关注预测为该类的区域
                masked_logits = class_logits * class_mask
                score = masked_logits.sum() / (class_mask.sum() + 1e-8)
                
                # 策略2：加上高置信度区域的额外权重
                high_conf_mask = (pred_probs[:, target_class, :, :] > 0.5).float()
                high_conf_score = (class_logits * high_conf_mask).mean()
                
                # 组合两种策略
                score = 0.7 * score + 0.3 * high_conf_score
            else:
                # 如果没有预测到该类，使用全局平均
                score = class_logits.mean()
        else:
            score = output.mean()
        
        # 反向传播
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # 生成CAM
        cam_dict = {}
        for layer_name in self.target_layers:
            if layer_name not in self.activations or layer_name not in self.gradients:
                continue
            
            activation = self.activations[layer_name]
            gradient = self.gradients[layer_name]
            
            # 处理梯度
            if isinstance(gradient, tuple):
                gradient = gradient[0]
            if gradient.dim() == 4 and gradient.shape[0] > 1:
                gradient = gradient[0:1]
            
            # Grad-CAM++风格的权重计算
            if gradient.dim() == 4:
                # 计算alpha (像素级权重)
                alpha_num = gradient.pow(2)
                alpha_denom = 2 * gradient.pow(2) + (gradient * activation).sum(dim=(2, 3), keepdim=True)
                alpha = alpha_num / (alpha_denom + 1e-8)
                
                # 应用ReLU到梯度
                positive_gradients = F.relu(gradient)
                
                # 计算权重
                weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
            else:
                weights = gradient.mean(dim=(1, 2), keepdim=True) if gradient.dim() == 3 else gradient
            
            # 加权求和
            if activation.dim() == 4:
                cam = (weights * activation).sum(dim=1, keepdim=True)
            else:
                cam = (weights * activation).sum(dim=0, keepdim=True)
            
            # ReLU激活
            cam = F.relu(cam)
            
            # 归一化
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = torch.zeros_like(cam)
            
            # 上采样到输入大小
            if cam.shape[-2:] != input_image.shape[-2:]:
                cam = F.interpolate(cam, size=input_image.shape[-2:], 
                                  mode='bilinear', align_corners=False)
            
            # 应用引导：使用预测mask来聚焦CAM
            if use_pred_mask and class_mask.sum() > 0:
                # 创建软mask（扩展预测区域）
                soft_mask = F.max_pool2d(class_mask.unsqueeze(1), 
                                        kernel_size=7, stride=1, padding=3)
                cam = cam * (0.3 + 0.7 * soft_mask)  # 保留一些背景信息
            
            cam_dict[layer_name] = cam.detach().cpu()
        
        return cam_dict, class_mask.cpu()
    
    def cleanup(self):
        """清理钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()



def visualize_improved_gradcam_segmentation(model, dataloader, save_dir, num_samples=5, 
                                           target_layers=['lgae1', 'lgae2', 'lgae3', 'lgae4']):
    """
    改进的分割GradCAM可视化
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 使用改进的GradCAM
    gradcam = ImprovedGradCAMForSegmentation(model, target_layers)
    
    samples_processed = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        # 处理数据
        if isinstance(batch_data, dict):
            images = batch_data['image']
            labels = batch_data.get('label')
            
            # 处理3D volume
            if len(images.shape) == 4 and images.shape[1] > 1:
                mid_slice = images.shape[1] // 2
                images = images[:, mid_slice:mid_slice+1, :, :]
                if labels is not None:
                    labels = labels[:, mid_slice, :, :]
            
            images = images.float().cuda()
            if labels is not None:
                labels = labels.cuda()
        else:
            continue
        
        # 确保是单通道
        if images.shape[1] != 1:
            images = images[:, 0:1, :, :]
        
        try:
            # 获取模型预测并计算性能
            with torch.no_grad():
                pred_output = model(images)
                pred_probs = F.softmax(pred_output, dim=1)
                pred_mask = torch.argmax(pred_output, dim=1)
                
                # 计算每个类别的IoU（如果有GT）
                if labels is not None:
                    num_classes = pred_output.shape[1]
                    ious = []
                    for c in range(1, num_classes):
                        pred_c = (pred_mask == c).float()
                        gt_c = (labels == c).float()
                        intersection = (pred_c * gt_c).sum()
                        union = pred_c.sum() + gt_c.sum() - intersection
                        iou = intersection / (union + 1e-8)
                        ious.append(iou.item())
                    print(f"Sample {samples_processed + 1} - IoUs: {ious}")
            
            # 准备可视化
            num_classes = pred_output.shape[1]
            target_classes = list(range(1, min(num_classes, 4)))
            
            # 创建更紧凑的布局
            fig = plt.figure(figsize=(20, 4 * len(target_classes) + 2))
            gs = fig.add_gridspec(len(target_classes) + 2, len(target_layers) + 3, 
                                 hspace=0.3, wspace=0.2)
            
            # 原始图像
            img_np = images[0, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # 第一行：总览
            ax_img = fig.add_subplot(gs[0, 0])
            ax_img.imshow(img_np, cmap='gray')
            ax_img.set_title('Input Image', fontsize=11, fontweight='bold')
            ax_img.axis('off')
            
            ax_gt = fig.add_subplot(gs[0, 1])
            if labels is not None:
                label_np = labels[0].detach().cpu().numpy() if labels.dim() == 3 else labels.detach().cpu().numpy()
                ax_gt.imshow(label_np, cmap='tab10', vmin=0, vmax=num_classes-1)
                ax_gt.set_title('Ground Truth', fontsize=11, fontweight='bold')
            else:
                ax_gt.imshow(pred_mask[0].cpu().numpy(), cmap='tab10', vmin=0, vmax=num_classes-1)
                ax_gt.set_title('Prediction', fontsize=11, fontweight='bold')
            ax_gt.axis('off')
            
            ax_pred = fig.add_subplot(gs[0, 2])
            ax_pred.imshow(pred_mask[0].cpu().numpy(), cmap='tab10', vmin=0, vmax=num_classes-1)
            ax_pred.set_title('Model Prediction', fontsize=11, fontweight='bold')
            ax_pred.axis('off')
            
            # 为每个类别生成CAM
            for class_idx, target_class in enumerate(target_classes):
                print(f"  Generating improved CAM for class {target_class}...")
                
                # 生成引导式CAM
                cam_dict, class_mask_np = gradcam.generate_cam_for_class_guided(
                    images, target_class, use_pred_mask=True
                )
                
                # 类别标题
                ax_title = fig.add_subplot(gs[class_idx + 2, 0])
                ax_title.text(0.5, 0.5, f'Class {target_class}', 
                            fontsize=12, fontweight='bold',
                            ha='center', va='center',
                            transform=ax_title.transAxes)
                ax_title.axis('off')
                
                # 预测mask
                ax_mask = fig.add_subplot(gs[class_idx + 2, 1])
                ax_mask.imshow(class_mask_np[0].numpy(), cmap='Reds', vmin=0, vmax=1)
                ax_mask.set_title('Predicted Region', fontsize=10)
                ax_mask.axis('off')
                
                # 置信度图
                ax_conf = fig.add_subplot(gs[class_idx + 2, 2])
                conf_map = pred_probs[0, target_class].detach().cpu().numpy()
                im_conf = ax_conf.imshow(conf_map, cmap='plasma', vmin=0, vmax=1)
                ax_conf.set_title('Confidence', fontsize=10)
                ax_conf.axis('off')
                
                # 各层CAM
                for layer_idx, layer_name in enumerate(target_layers):
                    ax_cam = fig.add_subplot(gs[class_idx + 2, layer_idx + 3])
                    
                    if layer_name in cam_dict:
                        cam = cam_dict[layer_name][0, 0].numpy()
                        
                        # 创建更好的叠加效果
                        cam_colored = plt.cm.jet(cam)[:, :, :3]
                        
                        # 自适应混合比例
                        alpha = 0.4 + 0.3 * cam  # CAM越强，显示越多
                        overlay = np.stack([img_np]*3, axis=-1)
                        for i in range(3):
                            overlay[:, :, i] = (1 - alpha) * overlay[:, :, i] + alpha * cam_colored[:, :, i]
                        overlay = np.clip(overlay, 0, 1)
                        
                        ax_cam.imshow(overlay)
                        ax_cam.set_title(f'{layer_name}', fontsize=9)
                    
                    ax_cam.axis('off')
            
            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            plt.colorbar(im_conf, cax=cbar_ax)
            cbar_ax.set_ylabel('Activation Strength', fontsize=10)
            
            plt.suptitle(f'Improved GradCAM Analysis - Sample {samples_processed + 1}', 
                        fontsize=14, fontweight='bold')
            
            save_path = os.path.join(save_dir, f'improved_gradcam_{samples_processed + 1:03d}.png')
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved improved GradCAM to {save_path}")
            
        except Exception as e:
            print(f"✗ Error processing sample {samples_processed + 1}: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')
        
        samples_processed += 1
    
    gradcam.cleanup()
    print(f"\nCompleted! Processed {samples_processed} samples.")





def check_data_format(dataloader, num_samples=2):
    """
    检查数据加载器返回的数据格式
    """
    for i, batch_data in enumerate(dataloader):
        if i >= num_samples:
            break
        
        print(f"\n--- Sample {i+1} ---")
        if isinstance(batch_data, dict):
            print("Data type: dict")
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value)}")
        else:
            print(f"Data type: {type(batch_data)}")
            if isinstance(batch_data, (list, tuple)):
                for j, item in enumerate(batch_data):
                    if isinstance(item, torch.Tensor):
                        print(f"  Item {j}: shape={item.shape}, dtype={item.dtype}")
                    else:
                        print(f"  Item {j}: type={type(item)}")


parser = argparse.ArgumentParser()
# parser.add_argument("--root_path", type=str, default=r"E:\1_master_degree_project\LiShuai\LGAELLH0622\datasets\ACDC", help="Name of Experiment")
parser.add_argument("--dataset", type=str, choices=["ACDC", "mscmrseg191", "mscmrseg192","Task051","Task052"],        default="Task051", help="select a dataset")
# 根目录由 dataset 自动推定
ROOT_MAP = {
    "ACDC":    r"E:\1_master_degree_project\LiShuai\LGAELLH0622\datasets\ACDC",
    "mscmrseg191": r"E:\1_master_degree_project\LiShuai\LGAELLH0622\datasets\mscmrseg19_split1",
    "Task051":  r"E:\1_master_degree_project\LiShuai\LGAELLH0622\datasets\Task05_split1",
    "mscmrseg192": r"E:\1_master_degree_project\LiShuai\LGAELLH0622\datasets\mscmrseg19_split2",
    "Task052":  r"E:\1_master_degree_project\LiShuai\LGAELLH0622\datasets\Task05_split2"
}
# 如果效果好的话记得剪切一下训练的目录，由 mscmrseg191/diffrect 到 Task051/diffrect
parser.add_argument("--exp", type=str, default="Task051/diffrect", help="experiment_name")
parser.add_argument("--model", type=str, default="UNet_LGAE1", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=3,help="output channel of network")
parser.add_argument("--img_channels", type=int, default=1, help="images channels, 1 if ACDC, 3 if GLAS")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)

parser.add_argument("--labeled_bs", type=int, default=3, help="labeled_batch_size per gpu")
parser.add_argument("--labeled_num", type=int, default=21, help="labeled data")
parser.add_argument("--refine_start", type=int, default=1000, help="start iter for rectification")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=250.0, help="consistency_rampup")
# rf
parser.add_argument("--base_chn_rf", type=int, default=64, help="rect model base channel")
parser.add_argument("--ldm_beta_sch", type=str, default='cosine', help="diffusion schedule beta")
parser.add_argument("--ts", type=int, default=10, help="ts")
parser.add_argument("--ts_sample", type=int, default=2, help="ts_sample")
parser.add_argument("--ref_consistency_weight", type=float, default=-1, help="consistency_rampup")
parser.add_argument("--no_color", default=False, action="store_true", help="no color image")
parser.add_argument("--no_blur", default=False, action="store_true", help="no blur image")
parser.add_argument("--rot", type=int, default=359, help="rotation angle")

# LLH visualization parameters
parser.add_argument("--llh_vis_freq", type=int, default=10000, help="Frequency of LLH visualization (in iterations)")
# parser.add_argument("--save_llh_evolution", default=True, action="store_true", help="Save full LLH evolution series at the end of training")
parser.add_argument("--noise_mode", type=str, choices=["none", "gaussian", "saltpepper"], default="none", help="noise type")
parser.add_argument("--noise_std",  type=float, default=0.03, help="gaussian std")
parser.add_argument("--noise_sp",   type=float, default=0.03, help="salt & pepper prob")
args = parser.parse_args()

def patients_to_slices(dataset_key, patiens_num):
    ref_dict = None
    if dataset_key == "ACDC":
        ref_dict = {
            "1": 32,
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "70": 1311,
        }
    elif dataset_key == "Task051":
        assert args.num_classes == 3, "Task05 only has 3 classes"
        ref_dict = {'2': 82,
                    '21':820}
    elif dataset_key == "Task052":
        assert args.num_classes == 3, "Task05 only has 3 classes"
        ref_dict = {'2': 40}
    elif dataset_key == "mscmrseg191":
        ref_dict = {'7': 110,
                    '35':529
                    }
    elif dataset_key == "mscmrseg192":
        ref_dict = {'7': 103}
    else:
        raise NotImplementedError
    return ref_dict[str(patiens_num)]
# ---------- 3. **新增通用可视化 5 张图 ---------
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F


import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quick_visualize_5(args, db, prefix="sample"):
    """
    为每个数据集随机抽取 5 份测试样本，展示中间切片 image & label
    """
    os.makedirs(prefix, exist_ok=True)
    for rank, idx in enumerate(torch.randperm(len(db))[:6]):
        data_dict = db[idx]

        # ----------- image ----------
        image = data_dict["image"]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # 如果是 3D volume 且 C>1，统一取通道 0
        if len(image.shape) == 3 and image.shape[0] > 1:
            image = image[0]

        # 归一到 [0,1]
        image -= image.min()
        image /= (image.max() + 1e-8)

        img_np = image.cpu().numpy()  # 现在一定是 numpy

        # ----------- label ----------
        # 处理 label
        label_np = data_dict.get("label")
        if label_np is None:
            label_np = np.zeros(img_np.shape, dtype=float)
        if isinstance(label_np, torch.Tensor):
            label_np = label_np.cpu().numpy()

        # 统一取中间切片
        if label_np.ndim == 3:
            label_np = label_np[label_np.shape[0] // 2]

        # ----------- 画图 ----------
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap="gray")
        plt.axis("off")
        plt.title("image")

        plt.subplot(1, 2, 2)
        plt.imshow(label_np, cmap="tab10", vmin=0, vmax=9)
        plt.axis("off")
        plt.title("label")

        save_name = os.path.join(prefix,
                                 f"{args.dataset}_case{rank:02d}_slice.png")
        print(save_name)
        plt.tight_layout()
        plt.savefig(save_name, dpi=150)
        plt.close()

        # 同时保存原始的 tensor / array
        torch.save({
            "image": torch.from_numpy(img_np) if isinstance(img_np, np.ndarray) else img_np,
            "label": torch.from_numpy(label_np).long() if isinstance(label_np, np.ndarray) else label_np.long()
        }, save_name.replace(".png", ".pth"))

    print(f"{prefix} done, saved 5 samples for dataset {args.dataset}")



def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(args, snapshot_path):
    args_dict = vars(args)
    for key, val in args_dict.items():
        logging.info("{}: {}".format(str(key), str(val)))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    gradcam_dir = os.path.join(snapshot_path, "gradcam_visualizations")
    os.makedirs(gradcam_dir, exist_ok=True)
    def create_model(ema=False, in_chns=1):
        model = net_factory(net_type=args.model, in_chns=in_chns, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def get_comp_loss(weak, strong, bs=args.batch_size):
        il_output = torch.reshape(
            strong,
            (bs, args.num_classes, args.patch_size[0] * args.patch_size[1])
        )
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.patch_size[0] * args.patch_size[1]))
        as_weight = torch.mean(as_weight)
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(torch.add(torch.negative(strong), 1), comp_labels)
        return comp_loss, as_weight
    # def get_comp_loss(weak, strong, bs=args.batch_size):
    #     batch_size = strong.shape[0]
    #     height = strong.shape[2]
    #     width = strong.shape[3]
    #     spatial_size = height * width
        
    #     il_output = torch.reshape(
    #         strong,
    #         (batch_size, args.num_classes, spatial_size)
    #     )
    #     as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(spatial_size))
    #     as_weight = torch.mean(as_weight)
    #     comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
    #     comp_loss = as_weight * ce_loss(torch.add(torch.negative(strong), 1), comp_labels)
    #     return comp_loss, as_weight
    

    def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment_Ours(args.patch_size, args)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_test = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    model = create_model(in_chns=args.img_channels)
    # print("\n" + "="*80)
    # print("MODEL ARCHITECTURE")
    # print("="*80)
    # print(model)
    # print("="*80)
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    # print("="*80 + "\n")
    if hasattr(model, 'set_training_stage'):
        model.set_training_stage("initialization")
    
    refine_model = UNet_LDMV2(
        in_chns=3+args.img_channels, 
        class_num=num_classes, 
        out_chns=num_classes, 
        ldm_method='replace', 
        ldm_beta_sch=args.ldm_beta_sch, 
        ts=args.ts, 
        ts_sample=args.ts_sample
    ).cuda()

    iter_num = 0
    start_epoch = 0

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    refine_optimizer = optim.SGD(refine_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.num_classes == 4:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    elif args.num_classes == 3:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}
    elif args.num_classes == 2:
        color_map = {0: (0, 0, 0), 1: (255, 255, 255)}

    model.train()
    refine_model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    current_performance = 0.0

    iter_num = int(iter_num)

    vis_dir = os.path.join(snapshot_path, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            # ---- 噪声注入（仅在训练阶段） ----
            if args.noise_mode != "none":
                weak_batch   = add_noise(weak_batch,   mode=args.noise_mode, gaussian_std=args.noise_std, sp_prob=args.noise_sp)
                strong_batch = add_noise(strong_batch, mode=args.noise_mode, gaussian_std=args.noise_std, sp_prob=args.noise_sp)
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            
            label_batch[args.labeled_bs:] = torch.zeros_like(label_batch[args.labeled_bs:])

            stage_name = f"Epoch_{epoch_num}_Batch_{i_batch}_Iter_{iter_num}"
            if hasattr(model, 'set_training_stage'):
                model.set_training_stage(stage_name)
            

            outputs_strong = model(strong_batch)
            outputs_weak = model(weak_batch)
                
            # outputs_strong_soft = torch.softmax(outputs_strong[0], dim=1)
            outputs_strong_soft = torch.softmax(outputs_strong,dim = 1)

            # outputs_weak_soft   = torch.softmax(outputs_weak[0],   dim=1)
            outputs_weak_soft   = torch.softmax(outputs_weak, dim = 1)

            pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
            outputs_weak_masked = outputs_weak_soft * pseudo_mask
            pseudo_outputs = torch.argmax(outputs_weak_masked.detach(), dim=1, keepdim=False)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)
            
            # sup_loss = ce_loss(outputs_weak[0][: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
            #     outputs_weak_soft[: args.labeled_bs],
            #     label_batch[: args.labeled_bs].unsqueeze(1),
            # )

            # unsup_loss = (
            #     ce_loss(outputs_strong[0][args.labeled_bs :], pseudo_outputs[args.labeled_bs :])
            #     + dice_loss(outputs_strong_soft[args.labeled_bs :], pseudo_outputs[args.labeled_bs :].unsqueeze(1))
            #     + as_weight * comp_loss
            # )


            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            unsup_loss = (
                ce_loss(outputs_strong[args.labeled_bs :], pseudo_outputs[args.labeled_bs :])
                + dice_loss(outputs_strong_soft[args.labeled_bs :], pseudo_outputs[args.labeled_bs :].unsqueeze(1))
                + as_weight * comp_loss
            )

            pseudo_mask_strong = (normalize(outputs_strong_soft) > args.conf_thresh).float()
            outputs_strong_masked = outputs_strong_soft * pseudo_mask_strong
            pseudo_outputs_strong = torch.argmax(outputs_strong_masked.detach(), dim=1, keepdim=False)

            pseudo_outputs_for_refine = pseudo_outputs.detach().clone()
            pseudo_outputs_numpy = pseudo_outputs_for_refine.clone().detach().cpu().numpy()
            pseudo_outputs_color = pl_weak_embed(color_map, pseudo_outputs_numpy)

            pseudo_outputs_strong_forrefine = pseudo_outputs_strong.detach().clone()
            pseudo_outputs_strong_numpy = pseudo_outputs_strong_forrefine.cpu().numpy()
            pseudo_outputs_strong_color = pl_strong_embed(color_map, pseudo_outputs_strong_numpy)

            label_batch_numpy = label_batch[:][: args.labeled_bs].cpu().numpy()
            label_batch_color = label_embed(color_map, label_batch_numpy)
            label_batch_color = torch.cat((label_batch_color.cuda(), pseudo_outputs_color[args.labeled_bs :].cuda()), dim=0)
                
            loss = sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t = dice_loss(pseudo_outputs_for_refine[:  args.labeled_bs].unsqueeze(1), label_batch[: args.labeled_bs].unsqueeze(1), oh_input=True)
            t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999
            lat_loss_sup, ref_outputs = refine_model(pseudo_outputs_color.cuda(), t, weak_batch.cuda(), training=True, good=label_batch_color.cuda())
            ref_outputs_soft = torch.softmax(ref_outputs, dim=1)

            sup_loss_cedice = ce_loss(ref_outputs[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                ref_outputs_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            sup_loss_ref = sup_loss_cedice + lat_loss_sup

            ref_soft = ref_outputs_soft
            ref_pseudo_mask = (normalize(ref_soft) > args.conf_thresh).float()
            ref_outputs_masked = ref_soft * ref_pseudo_mask
            ref_pseudo_outputs = torch.argmax(ref_outputs_masked.detach(), dim=1, keepdim=False)

            t2 = dice_loss(pseudo_outputs_strong_forrefine[args.labeled_bs :].unsqueeze(1), ref_pseudo_outputs[args.labeled_bs:].unsqueeze(1), oh_input=True)
            t2 = torch.ones((pseudo_outputs_strong_color.shape[0]), dtype=torch.float32, device='cuda') * t2 * 999
            lat_loss_unsup, ref_outputs_strong = refine_model(pseudo_outputs_strong_color.cuda(), t2, strong_batch.cuda(), training=True, good=pseudo_outputs_color.cuda())
            ref_outputs_strong_soft = torch.softmax(ref_outputs_strong, dim=1)

            ref_comp_loss, ref_as_weight = get_comp_loss(weak=ref_soft, strong=ref_outputs_strong_soft)
            unsup_loss_cedice = (
                ce_loss(ref_outputs_strong[args.labeled_bs :], ref_pseudo_outputs[args.labeled_bs :])
                + dice_loss(ref_outputs_strong_soft[args.labeled_bs :], ref_pseudo_outputs[args.labeled_bs :].unsqueeze(1))
                + ref_as_weight * ref_comp_loss
            ) 
            unsup_loss_ref = unsup_loss_cedice + lat_loss_unsup

            ref_consistency_weight = consistency_weight if args.ref_consistency_weight == -1 else args.ref_consistency_weight
            refine_loss = sup_loss_ref + ref_consistency_weight * unsup_loss_ref
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()
            # 在train函数中，修改GradCAM调用部分
            if iter_num % args.llh_vis_freq == 0 and args.model == "UNet_LGAE1":
                try:
                    # 先清理GPU缓存
                    torch.cuda.empty_cache()
                    
                    # 使用小批量进行GradCAM
                    with torch.no_grad():
                        weak_batch_for_cam = weak_batch[:1].clone().detach()
                        label_for_cam = label_batch[:1].clone().detach() if args.labeled_bs > 0 else None
                    
                    # 使用弱增强图像进行可视化
                    visualize_gradcam_during_training(
                        model=model,
                        image=weak_batch_for_cam,
                        label=label_for_cam,
                        iter_num=iter_num,
                        save_dir=gradcam_dir,
                        target_layers=['lgae1', 'lgae2', 'lgae3', 'lgae4']
                    )
                    logging.info(f"Saved GradCAM visualization at iteration {iter_num}")
                    
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logging.warning(f"Failed to generate GradCAM at iteration {iter_num}: {e}")
                    torch.cuda.empty_cache()
            if iter_num > args.refine_start:
                t = dice_loss(pseudo_outputs_for_refine[:  args.labeled_bs].unsqueeze(1), label_batch[: args.labeled_bs].unsqueeze(1), oh_input=True)
                t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999
                ref_outputs = refine_model(pseudo_outputs_color.cuda(), t, weak_batch.cuda(), training=False)

                ref_outputs_soft_for_refine = torch.softmax(ref_outputs, dim=1)
                pseudo_mask = (normalize(ref_outputs_soft_for_refine) > args.conf_thresh).float()
                ref_outputs_soft_masked = ref_outputs_soft_for_refine * pseudo_mask
                pseudo_outputs_ref = torch.argmax(ref_outputs_soft_masked.detach(), dim=1, keepdim=False)
                
                outputs_weak = model(weak_batch)
                # outputs_weak_soft = torch.softmax(outputs_weak[0], dim=1)
                # unsup_label_rect_loss = ce_loss(outputs_weak[0][args.labeled_bs :], pseudo_outputs_ref[args.labeled_bs :]) + dice_loss(
                #     outputs_weak_soft[args.labeled_bs :],
                #     pseudo_outputs_ref[args.labeled_bs :].unsqueeze(1),
                # )


                outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
                unsup_label_rect_loss = ce_loss(outputs_weak[args.labeled_bs :], pseudo_outputs_ref[args.labeled_bs :]) + dice_loss(
                    outputs_weak_soft[args.labeled_bs :],
                    pseudo_outputs_ref[args.labeled_bs :].unsqueeze(1),
                )

                optimizer.zero_grad()
                unsup_label_rect_loss.backward()
                optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            logging.info("iteration %d : mloss : %f, refsupce: %f, refsuplat: %f, refunsupce: %f, refunsuplat: %f, t: %f, t2: %f" % 
                         (iter_num, loss.item(), sup_loss_cedice.item(), lat_loss_sup.item(), unsup_loss_cedice.item(), lat_loss_unsup.item(), torch.mean(t).item(), torch.mean(t2).item()))



            if iter_num % 100 == 0:
                model.eval()
                refine_model.eval()
                metric_list = 0.0
                FLAGS = parser.parse_args()
                
                # # 创建可视化保存目录
                # vis_save_dir = os.path.join(snapshot_path, f"llh_visualization_iter_{iter_num}")
                # os.makedirs(vis_save_dir, exist_ok=True)
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                for i_batch_val, sampled_batch_val in enumerate(valloader):

                    
                    # 正常计算指标
                    metric_i = test_single_volume(
                        FLAGS,
                        sampled_batch_val["image"],
                        sampled_batch_val["label"],
                        model,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)

                current_performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_jaccard = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]

                if current_performance > best_performance:
                    best_performance = current_performance
                    logging.info("BEST PERFORMANCE UPDATED AT ITERATION %d: Dice: %f, HD95: %f" % (iter_num, current_performance, mean_hd95))
                    save_best = os.path.join(snapshot_path, "{}_{}_best_model0906.pth".format(args.model,args.noise_mode))
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)

                for class_i in range(num_classes - 1):
                    logging.info(
                        "iteration %d: model_val_%d_dice : %f model_val_%d_hd95 : %f model_val_%d_jaccard : %f model_val_%d_asd : %f"
                        % (iter_num, class_i + 1, metric_list[class_i, 0], class_i + 1, metric_list[class_i, 1], 
                        class_i + 1, metric_list[class_i, 2], class_i + 1, metric_list[class_i, 3])
                    )

                logging.info(
                    "iteration %d : model_mean_dice : %f model_mean_hd95 : %f model_mean_jaccard : %f model_mean_asd : %f"
                    % (iter_num, current_performance, mean_hd95, mean_jaccard, mean_asd)
                )
                
                # test_func1(num_classes, db_test, model, refine_model, iter_num, testloader)

                model.train()
                refine_model.train()

            if iter_num >= max_iterations:
                break
        # if args.save_llh_evolution and args.model == "UNet_LGAE1":
        #     try:
        #         # 生成GIF动画
        #         create_gradcam_evolution_gif(gradcam_dir, os.path.join(snapshot_path, "gradcam_evolution.gif"))
        #         logging.info("Generated GradCAM evolution GIF")
        #     except Exception as e:
        #         logging.error(f"Failed to generate GradCAM evolution GIF: {e}")
        if iter_num >= max_iterations:
            iterator.close()
            break
            
def create_gradcam_evolution_gif(image_dir, output_path, duration=200):
    """
    从保存的GradCAM图像创建演化GIF
    
    Args:
        image_dir: 包含GradCAM图像的目录
        output_path: 输出GIF路径
        duration: 每帧持续时间（毫秒）
    """
    from PIL import Image as PILImage
    import glob
    
    # 获取所有GradCAM图像
    image_files = sorted(glob.glob(os.path.join(image_dir, 'gradcam_iter_*.png')))
    
    if not image_files:
        logging.warning("No GradCAM images found for GIF creation")
        return
    
    # 读取图像
    images = []
    for img_file in image_files[::5]:  # 每5个迭代取一张，避免GIF太大
        img = PILImage.open(img_file)
        images.append(img)
    
    # 保存为GIF
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        logging.info(f"Saved GradCAM evolution GIF to {output_path}")

def pl_weak_embed(color_map, pseudo_outputs_numpy):
    pseudo_outputs_color = torch.zeros((pseudo_outputs_numpy.shape[0], 3, pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2]), dtype=torch.float32)
    for i in range(pseudo_outputs_numpy.shape[0]):
        color_data = np.zeros((pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_color[i] = color_tensor
    return pseudo_outputs_color

def pl_strong_embed(color_map, pseudo_outputs_strong_numpy):
    pseudo_outputs_strong_color = torch.zeros((pseudo_outputs_strong_numpy.shape[0], 3, pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2]), dtype=torch.float32)
    for i in range(pseudo_outputs_strong_numpy.shape[0]):
        color_data = np.zeros((pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_strong_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_strong_color[i] = color_tensor
    return pseudo_outputs_strong_color

def label_embed(color_map, label_batch_numpy):
    label_batch_color = torch.zeros((label_batch_numpy.shape[0], 3, label_batch_numpy.shape[1], label_batch_numpy.shape[2]), dtype=torch.float32, device='cuda')
    for i in range(label_batch_numpy.shape[0]):
        color_data = np.zeros((label_batch_numpy.shape[1], label_batch_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[label_batch_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        label_batch_color[i] = color_tensor
    return label_batch_color

def test_func1(num_classes, db_test, model, refine_model, iter_num, testloader):
    metric_list_test = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes,
                    )
        metric_list_test += np.array(metric_i)
    metric_list_test = metric_list_test / len(db_test)
    performance = np.mean(metric_list_test, axis=0)[0]
    mean_hd95 = np.mean(metric_list_test, axis=0)[1]
    mean_jaccard = np.mean(metric_list_test, axis=0)[2]
    mean_asd = np.mean(metric_list_test, axis=0)[3]
    
    for class_i in range(num_classes - 1):
        logging.info(
            "(Test) iteration %d: model_val_%d_dice : %f model_val_%d_hd95 : %f model_val_%d_jaccard : %f model_val_%d_asd : %f"
            % (iter_num, class_i + 1, metric_list_test[class_i, 0], class_i + 1, metric_list_test[class_i, 1], 
               class_i + 1, metric_list_test[class_i, 2], class_i + 1, metric_list_test[class_i, 3])
        )
    
    logging.info(
        "(Test) iteration %d : model_mean_dice : %f model_mean_hd95 : %f model_mean_jaccard : %f model_mean_asd : %f"
        % (iter_num, performance, mean_hd95, mean_jaccard, mean_asd)
    )

def test_func(FLAGS, test_save_path, num_classes, db_test, model, refine_model, iter_num, testloader):
    metric_list_test = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume_refinev3(
                        FLAGS,
                        test_save_path,
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        # refine_model,
                        classes=num_classes,
                        
                    )
        metric_list_test += np.array(metric_i)
    metric_list_test = metric_list_test / len(db_test)
    performance = np.mean(metric_list_test, axis=0)[0]
    mean_hd95 = np.mean(metric_list_test, axis=0)[1]
    mean_jaccard = np.mean(metric_list_test, axis=0)[2]
    mean_asd = np.mean(metric_list_test, axis=0)[3]

    for class_i in range(num_classes - 1):
        logging.info(
                        "(Test) iteration %d: model_val_%d_dice : %f model_val_%d_hd95 : %f model_val_%d_jaccard : %f model_val_%d_asd : %f"
                        % (iter_num, class_i + 1, metric_list_test[class_i, 0], class_i + 1, metric_list_test[class_i, 1], class_i + 1, metric_list_test[class_i, 2], class_i + 1, metric_list_test[class_i, 3])
                    )
    logging.info(
                    "(Test) iteration %d : model_mean_dice : %f model_mean_hd95 : %f model_mean_jaccard : %f model_mean_asd : %f"
                    % (iter_num, performance, mean_hd95, mean_jaccard, mean_asd)
                )

if __name__ == "__main__":
    import matplotlib
    matplotlib.set_loglevel("warning")
    import logging as mpl_logging
    mpl_logging.getLogger('matplotlib.font_manager').disabled = True
    mpl_logging.getLogger('matplotlib').setLevel(mpl_logging.WARNING)
    args.root_path = ROOT_MAP[args.dataset]
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    FLAGS = parser.parse_args()
    print(FLAGS)
    print(ROOT_MAP[args.dataset])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    print(snapshot_path + "/log.log")
    logging.getLogger('').handlers = []
    logging.basicConfig(
        filename=snapshot_path + "/log.log",
        level=logging.DEBUG,
        filemode="w", 
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if "brats" in args.root_path.lower():
        args.patch_size = [128, 128]
    logging.info(str(args))

    train_or_not = True
    train_or_not = False
    if train_or_not:
        train(args, snapshot_path)
    else:
        # 测试模式
        iter_num = 0
        db_test = BaseDataSets(base_dir=args.root_path, split="test")
        db_val = BaseDataSets(base_dir=args.root_path, split="val")
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
        # quick_visualize_5(args, db_test,prefix=f"visual_{args.dataset}") 
        import sys
        sys.path.append(r'E:\1_master_degree_project\LiShuai\20250520\networks')
        from unet import UNet_LGAE1,UNet,UNet_DS,UNet_CCT,UNet_URPC
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\mscmrseg191\diffrect_7_labeled\UNet_LGAE1\UNet_LGAE1_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_1_labeled\UNet_LGAE1\UNet_LGAE1_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_LGAE1\UNet_LGAE1_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_LGAE1\UNet_LGAE1_none_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_DS\unet_ds_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\unet_urpc\unet_urpc_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\mscmrseg191\diffrect_7_labeled\UNet_LGAE1\UNet_LGAE1_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\Task052\diffrect_2_labeled\UNet_LGAE1\UNet_LGAE1_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\unet_cct\unet_cct_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_LGAE1\UNet_LGAE1_gaussian_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\unet_cct\unet_cct_saltpepper_best_model.pth"
        # model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\mscmrseg191\diffrect_35_labeled\UNet_LGAE1\UNet_LGAE1_none_best_model0830.pth"
        model_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\Task051\diffrect_21_labeled\UNet_LGAE1\UNet_LGAE1_none_best_model0906.pth"
        model = UNet_LGAE1(in_chns=1, class_num=args.num_classes).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        refine_model = UNet_LDMV2(
            in_chns=3+args.img_channels, 
            class_num=args.num_classes, 
            out_chns=args.num_classes, 
            ldm_method='replace', 
            ldm_beta_sch=args.ldm_beta_sch, 
            ts=args.ts, 
            ts_sample=args.ts_sample
        ).cuda()
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\mscmrseg191\diffrect_7_labeled\UNet_LGAE1"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_1_labeled\UNet_LGAE1"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_LGAE1"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_DS"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\unet_urpc"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\mscmrseg191\diffrect_7_labeled\UNet_LGAE1"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\Task052\diffrect_2_labeled\UNet_LGAE1"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\unet_cct"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\ACDC\diffrect_28_labeled\UNet_LGAE1"
        # test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\mscmrseg191\diffrect_35_labeled\UNet_LGAE1"
        test_save_path = r"E:\1_master_degree_project\LiShuai\LGAELLH0622\logs\Task051\diffrect_21_labeled\UNet_LGAE1"
        test_func(FLAGS, test_save_path, args.num_classes, db_test, model, refine_model, iter_num, testloader)
        print('-----------------------------valloader-----------------------------')
        # test_func(FLAGS, test_save_path, args.num_classes, db_val, model, refine_model, iter_num, valloader)
        # 先检查数据格式
        print("Checking testloader data format:")
        # check_data_format(testloader, num_samples=2)
        
        # # 在if __name__ == "__main__": 的测试部分       
        # gradcam_test_dir = os.path.join(test_save_path, "improved_gradcam")
        # try:
        #     visualize_improved_gradcam_segmentation(
        #         model=model,
        #         dataloader=testloader,
        #         save_dir=gradcam_test_dir,
        #         num_samples=10,
        #         target_layers=['lgae1', 'lgae2', 'lgae3', 'lgae4']
        #     )
        #     print(f"Improved GradCAM completed. Results saved in {gradcam_test_dir}")
        # except Exception as e:
        #     print(f"Error during GradCAM visualization: {e}")
        #     import traceback
        #     traceback.print_exc()

        # node_gradcam_dir = os.path.join(test_save_path, "node_gradcam_analysis")

        # # 使用新的综合可视化
        # try:
        #     visualize_all_layers_top_nodes(
        #         model=model,
        #         dataloader=testloader,
        #         save_dir=node_gradcam_dir,
        #         num_samples=5,
        #         lgae_configs={'lgae1': 6, 'lgae2': 6, 'lgae3': 6, 'lgae4': 6}
        #     )
        #     print(f"Top nodes visualization completed!")
        # except Exception as e:
        #     print(f"Error: {e}")
        #     import traceback
        #     traceback.print_exc()
# -*- coding: utf-8 -*-
"""
MinkUNet with HFE, CDAG, HCL - 集成本文三大创新模块的完整网络

架构说明:
1. 编码器: 标准MinkUNet编码器，提取多尺度特征
2. HFE模块: 在编码器输出后，生成三种专门化特征
   - 全局上下文特征 → 树木解码器
   - 语义特征 → 语义解码器  
   - 局部细节特征 → 标准实例解码器
3. CDAG模块: 在解码器跳跃连接处，自适应选择编码器特征
4. 三解码器: 语义、标准实例、树木实例
5. HCL损失: 在训练时约束层次一致性
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from .resnet import ResNetBase


class MinkUNetWithModules(ResNetBase):
    """
    集成HFE、CDAG的MinkUNet骨干网络
    
    根据配置文件动态启用/禁用各模块
    """
    
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels, out_channels, cfg=None, instance_decoder=True, D=3, use_tanh=False):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数 (语义类别数)
            cfg: 配置字典或ConfigManager
            instance_decoder: 是否使用实例解码器
            D: 空间维度
            use_tanh: 是否对偏移向量使用tanh
        """
        self.instance_decoder = instance_decoder
        self.use_tanh = use_tanh
        self.cfg = cfg
        
        # 解析配置
        self._parse_config(cfg)
        
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def _parse_config(self, cfg):
        """解析配置文件，获取模块开关"""
        if cfg is None:
            # 默认全部禁用
            self.use_hfe = False
            self.use_cdag = False
            self.hfe_cfg = {}
            self.cdag_cfg = {}
        elif isinstance(cfg, dict):
            # 原始字典
            network_cfg = cfg.get('network', {})
            hfe_cfg = network_cfg.get('hfe', {}) or {}
            cdag_cfg = network_cfg.get('cdag', {}) or {}
            self.use_hfe = hfe_cfg.get('enabled', False)
            self.use_cdag = cdag_cfg.get('enabled', False)
            self.hfe_cfg = hfe_cfg
            self.cdag_cfg = cdag_cfg
        elif hasattr(cfg, 'get'):
            # ConfigManager对象
            self.use_hfe = cfg.get('network.hfe.enabled', False)
            self.use_cdag = cfg.get('network.cdag.enabled', False)
            print(self.use_cdag)
            self.hfe_cfg = cfg.get('network.hfe', {}) or {}
            self.cdag_cfg = cfg.get('network.cdag', {}) or {}
        else:
            self.use_hfe = False
            self.use_cdag = False
            self.hfe_cfg = {}
            self.cdag_cfg = {}

    def network_initialization(self, in_channels, out_channels, D):
        """网络初始化"""
        
        # ==================== 编码器 ====================
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])
        
        self.inplanes_enc = self.inplanes  # 编码器输出通道数
        
        # ==================== HFE模块（可选）====================
        if self.use_hfe:
            from .hfe import HFE
            self.hfe = HFE(
                in_channels=self.inplanes_enc,
                out_channels=self.inplanes_enc,
                cfg=self.hfe_cfg
            )
            print(f"[INFO] HFE模块已启用")
        
        # ==================== CDAG模块（可选）====================
        if self.use_cdag:
            from .cdag import CDAG
            # 获取CDAG配置
            cdag_apply_to = self.cdag_cfg.get('apply_to', {})
            self.cdag_apply_sem = cdag_apply_to.get('semantic_decoder', True)
            self.cdag_apply_tree = cdag_apply_to.get('tree_decoder', True)
            self.cdag_apply_inst = cdag_apply_to.get('instance_decoder', True)
            
            # 为每个解码器阶段创建CDAG模块
            # 语义解码器CDAG
            if self.cdag_apply_sem:
                self.cdag_sem_s8 = CDAG(self.PLANES[4], self.PLANES[2] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_sem_s4 = CDAG(self.PLANES[5], self.PLANES[1] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_sem_s2 = CDAG(self.PLANES[6], self.PLANES[0] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_sem_s1 = CDAG(self.PLANES[7], self.INIT_DIM, cfg=self.cdag_cfg)
            
            # 树木解码器CDAG
            if self.cdag_apply_tree:
                self.cdag_tree_s8 = CDAG(self.PLANES[4], self.PLANES[2] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_tree_s4 = CDAG(self.PLANES[5], self.PLANES[1] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_tree_s2 = CDAG(self.PLANES[6], self.PLANES[0] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_tree_s1 = CDAG(self.PLANES[7], self.INIT_DIM, cfg=self.cdag_cfg)
            
            # 标准实例解码器CDAG
            if self.cdag_apply_inst:
                self.cdag_inst_s8 = CDAG(self.PLANES[4], self.PLANES[2] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_inst_s4 = CDAG(self.PLANES[5], self.PLANES[1] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_inst_s2 = CDAG(self.PLANES[6], self.PLANES[0] * self.BLOCK.expansion, cfg=self.cdag_cfg)
                self.cdag_inst_s1 = CDAG(self.PLANES[7], self.INIT_DIM, cfg=self.cdag_cfg)
            
            print(f"[INFO] CDAG模块已启用: sem={self.cdag_apply_sem}, tree={self.cdag_apply_tree}, inst={self.cdag_apply_inst}")
        
        # ==================== 语义解码器 ====================
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes_enc, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
        
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion, out_channels,
            kernel_size=1, bias=True, dimension=D)
        
        self.relu = ME.MinkowskiReLU(inplace=True)

        # ==================== 实例解码器 ====================
        if self.instance_decoder:
            # 树木实例解码器 (ins2)
            self.convtr4p16s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes_enc, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
            self.bntr4_ins2 = ME.MinkowskiBatchNorm(self.PLANES[4])

            inplanes_ins2 = 2*self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
            self.inplanes = inplanes_ins2
            self.block5_ins2 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
            
            self.convtr5p8s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
            self.bntr5_ins2 = ME.MinkowskiBatchNorm(self.PLANES[5])

            inplanes_ins2 = 2*self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
            self.inplanes = inplanes_ins2
            self.block6_ins2 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
            
            self.convtr6p4s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
            self.bntr6_ins2 = ME.MinkowskiBatchNorm(self.PLANES[6])

            inplanes_ins2 = 2*self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
            self.inplanes = inplanes_ins2
            self.block7_ins2 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
            
            self.convtr7p2s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
            self.bntr7_ins2 = ME.MinkowskiBatchNorm(self.PLANES[7])

            inplanes_ins2 = 2*self.PLANES[7] + self.INIT_DIM
            self.inplanes = inplanes_ins2
            self.block8_ins2 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

            self.final_ins2 = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion, 3,
                kernel_size=1, bias=True, dimension=D)

            # 标准实例解码器 (ins1)
            self.convtr4p16s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes_enc, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
            self.bntr4_ins1 = ME.MinkowskiBatchNorm(self.PLANES[4])

            inplanes_ins1 = 2*self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
            self.inplanes = inplanes_ins1
            self.block5_ins1 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
            
            self.convtr5p8s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
            self.bntr5_ins1 = ME.MinkowskiBatchNorm(self.PLANES[5])

            inplanes_ins1 = 2*self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
            self.inplanes = inplanes_ins1
            self.block6_ins1 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
            
            self.convtr6p4s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
            self.bntr6_ins1 = ME.MinkowskiBatchNorm(self.PLANES[6])

            inplanes_ins1 = 2*self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
            self.inplanes = inplanes_ins1
            self.block7_ins1 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
            
            self.convtr7p2s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
            self.bntr7_ins1 = ME.MinkowskiBatchNorm(self.PLANES[7])

            inplanes_ins1 = 2*self.PLANES[7] + self.INIT_DIM
            self.inplanes = inplanes_ins1
            self.block8_ins1 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

            self.final_ins1 = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion, 3,
                kernel_size=1, bias=True, dimension=D)

    def _apply_cdag(self, cdag_module, decoder_feat, encoder_feat):
        """应用CDAG模块"""
        if cdag_module is not None:
            return cdag_module(decoder_feat, encoder_feat)
        return encoder_feat

    def forward(self, x):
        """前向传播"""
        # ==================== 编码器 ====================
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)  # stride=1, 初始特征

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)  # stride=2

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)  # stride=4

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)  # stride=8

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_encoder = self.block4(out)  # stride=16, 编码器最终输出
        
        # ==================== HFE分支（可选）====================
        if self.use_hfe:
            # HFE输出: (实例特征, 语义特征, 树木特征)
            feat_inst, feat_sem, feat_tree = self.hfe(out_encoder)
        else:
            # 不使用HFE，三个解码器共享相同输入
            feat_inst = out_encoder
            feat_sem = out_encoder
            feat_tree = out_encoder
        
        # ==================== 语义解码器 ====================
        out_sem = self.convtr4p16s2(feat_sem)
        out_sem = self.bntr4(out_sem)
        out_skip_sem1 = self.relu(out_sem)  # stride=8

        # CDAG门控（可选）
        if self.use_cdag and self.cdag_apply_sem:
            out_b3p8_gated = self._apply_cdag(self.cdag_sem_s8, out_skip_sem1, out_b3p8)
        else:
            out_b3p8_gated = out_b3p8
        
        out_sem = ME.cat(out_skip_sem1, out_b3p8_gated)
        out_sem = self.block5(out_sem)

        out_sem = self.convtr5p8s2(out_sem)
        out_sem = self.bntr5(out_sem)
        out_skip_sem2 = self.relu(out_sem)  # stride=4

        if self.use_cdag and self.cdag_apply_sem:
            out_b2p4_gated = self._apply_cdag(self.cdag_sem_s4, out_skip_sem2, out_b2p4)
        else:
            out_b2p4_gated = out_b2p4
            
        out_sem = ME.cat(out_skip_sem2, out_b2p4_gated)
        out_sem = self.block6(out_sem)

        out_sem = self.convtr6p4s2(out_sem)
        out_sem = self.bntr6(out_sem)
        out_skip_sem3 = self.relu(out_sem)  # stride=2

        if self.use_cdag and self.cdag_apply_sem:
            out_b1p2_gated = self._apply_cdag(self.cdag_sem_s2, out_skip_sem3, out_b1p2)
        else:
            out_b1p2_gated = out_b1p2
            
        out_sem = ME.cat(out_skip_sem3, out_b1p2_gated)
        out_sem = self.block7(out_sem)

        out_sem = self.convtr7p2s2(out_sem)
        out_sem = self.bntr7(out_sem)
        out_skip_sem4 = self.relu(out_sem)  # stride=1

        if self.use_cdag and self.cdag_apply_sem:
            out_p1_gated = self._apply_cdag(self.cdag_sem_s1, out_skip_sem4, out_p1)
        else:
            out_p1_gated = out_p1
            
        out_sem = ME.cat(out_skip_sem4, out_p1_gated)
        out_sem = self.block8(out_sem)

        out_ins1, out_ins2 = None, None
        
        if self.instance_decoder:
            # ==================== 树木实例解码器 (ins2) ====================
            out_ins2 = self.convtr4p16s2_ins2(feat_tree)
            out_ins2 = self.bntr4_ins2(out_ins2)
            out_skip_tree1 = self.relu(out_ins2)

            if self.use_cdag and self.cdag_apply_tree:
                out_b3p8_tree = self._apply_cdag(self.cdag_tree_s8, out_skip_tree1, out_b3p8)
            else:
                out_b3p8_tree = out_b3p8
                
            out_ins2 = ME.cat(out_skip_tree1, out_b3p8_tree, out_skip_sem1)
            out_ins2 = self.block5_ins2(out_ins2)

            out_ins2 = self.convtr5p8s2_ins2(out_ins2)
            out_ins2 = self.bntr5_ins2(out_ins2)
            out_skip_tree2 = self.relu(out_ins2)

            if self.use_cdag and self.cdag_apply_tree:
                out_b2p4_tree = self._apply_cdag(self.cdag_tree_s4, out_skip_tree2, out_b2p4)
            else:
                out_b2p4_tree = out_b2p4
                
            out_ins2 = ME.cat(out_skip_tree2, out_b2p4_tree, out_skip_sem2)
            out_ins2 = self.block6_ins2(out_ins2)

            out_ins2 = self.convtr6p4s2_ins2(out_ins2)
            out_ins2 = self.bntr6_ins2(out_ins2)
            out_skip_tree3 = self.relu(out_ins2)

            if self.use_cdag and self.cdag_apply_tree:
                out_b1p2_tree = self._apply_cdag(self.cdag_tree_s2, out_skip_tree3, out_b1p2)
            else:
                out_b1p2_tree = out_b1p2
                
            out_ins2 = ME.cat(out_skip_tree3, out_b1p2_tree, out_skip_sem3)
            out_ins2 = self.block7_ins2(out_ins2)

            out_ins2 = self.convtr7p2s2_ins2(out_ins2)
            out_ins2 = self.bntr7_ins2(out_ins2)
            out_skip_tree4 = self.relu(out_ins2)

            if self.use_cdag and self.cdag_apply_tree:
                out_p1_tree = self._apply_cdag(self.cdag_tree_s1, out_skip_tree4, out_p1)
            else:
                out_p1_tree = out_p1
                
            out_ins2 = ME.cat(out_skip_tree4, out_p1_tree, out_skip_sem4)
            out_ins2 = self.block8_ins2(out_ins2)
            out_ins2 = self.final_ins2(out_ins2)
            
            if self.use_tanh:
                out_ins2 = ME.MinkowskiTanh()(out_ins2)

            # ==================== 标准实例解码器 (ins1) ====================
            out_ins1 = self.convtr4p16s2_ins1(feat_inst)
            out_ins1 = self.bntr4_ins1(out_ins1)
            out_skip_inst1 = self.relu(out_ins1)

            if self.use_cdag and self.cdag_apply_inst:
                out_b3p8_inst = self._apply_cdag(self.cdag_inst_s8, out_skip_inst1, out_b3p8)
            else:
                out_b3p8_inst = out_b3p8
                
            out_ins1 = ME.cat(out_skip_inst1, out_b3p8_inst, out_skip_tree1)
            out_ins1 = self.block5_ins1(out_ins1)

            out_ins1 = self.convtr5p8s2_ins1(out_ins1)
            out_ins1 = self.bntr5_ins1(out_ins1)
            out_skip_inst2 = self.relu(out_ins1)

            if self.use_cdag and self.cdag_apply_inst:
                out_b2p4_inst = self._apply_cdag(self.cdag_inst_s4, out_skip_inst2, out_b2p4)
            else:
                out_b2p4_inst = out_b2p4
                
            out_ins1 = ME.cat(out_skip_inst2, out_b2p4_inst, out_skip_tree2)
            out_ins1 = self.block6_ins1(out_ins1)

            out_ins1 = self.convtr6p4s2_ins1(out_ins1)
            out_ins1 = self.bntr6_ins1(out_ins1)
            out_skip_inst3 = self.relu(out_ins1)

            if self.use_cdag and self.cdag_apply_inst:
                out_b1p2_inst = self._apply_cdag(self.cdag_inst_s2, out_skip_inst3, out_b1p2)
            else:
                out_b1p2_inst = out_b1p2
                
            out_ins1 = ME.cat(out_skip_inst3, out_b1p2_inst, out_skip_tree3)
            out_ins1 = self.block7_ins1(out_ins1)

            out_ins1 = self.convtr7p2s2_ins1(out_ins1)
            out_ins1 = self.bntr7_ins1(out_ins1)
            out_skip_inst4 = self.relu(out_ins1)

            if self.use_cdag and self.cdag_apply_inst:
                out_p1_inst = self._apply_cdag(self.cdag_inst_s1, out_skip_inst4, out_p1)
            else:
                out_p1_inst = out_p1
                
            out_ins1 = ME.cat(out_skip_inst4, out_p1_inst, out_skip_tree4)
            out_ins1 = self.block8_ins1(out_ins1)
            out_ins1 = self.final_ins1(out_ins1)
            
            if self.use_tanh:
                out_ins1 = ME.MinkowskiTanh()(out_ins1)

        return self.final(out_sem), out_ins1, out_ins2


# ==================== 具体网络变体 ====================

class MinkUNet14A_Ours(MinkUNetWithModules):
    """MinkUNet14A with HFE, CDAG modules"""
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18A_Ours(MinkUNetWithModules):
    """MinkUNet18A with HFE, CDAG modules"""
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34A_Ours(MinkUNetWithModules):
    """MinkUNet34A with HFE, CDAG modules"""
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet14B_Ours(MinkUNetWithModules):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B_Ours(MinkUNetWithModules):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet34B_Ours(MinkUNetWithModules):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)

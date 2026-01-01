#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation.py - 统一的消融实验运行脚本

支持两类消融实验:
1. 模块级消融 (Tab. ablation_modules): 验证HFE/CDAG/HCL各模块的贡献
2. 模块内消融: 
   - CDAG组件消融 (Tab. ablation_cdag): 空间→通道→多尺度→像素
   - HFE分支消融 (Tab. ablation_hfe): 膨胀卷积→完整
   - CDAG位置消融 (Tab. ablation_cdag_pos): 语义/树木/实例/全部解码器
   - HCL权重消融 (Tab. ablation_hcl): λ=0, 0.05, 0.1, 0.2, 0.5

用法:
    # 查看所有可用实验
    python run_ablation.py --list
    
    # 运行模块级消融 (对应论文Tab. ablation_modules)
    python run_ablation.py --module all
    python run_ablation.py --module baseline hfe_only ours
    
    # 运行CDAG组件消融 (对应论文Tab. ablation_cdag)
    python run_ablation.py --cdag all
    
    # 运行HFE分支消融 (对应论文Tab. ablation_hfe)
    python run_ablation.py --hfe all
    
    # 运行CDAG位置消融 (对应论文Tab. ablation_cdag_pos)
    python run_ablation.py --cdag-pos all
    
    # 运行HCL权重消融 (对应论文Tab. ablation_hcl)
    python run_ablation.py --hcl all
    
    # 运行所有消融实验
    python run_ablation.py --all
"""

import click
import os
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================================
# 消融实验定义
# ============================================================================

# 1. 模块级消融 (Tab. ablation_modules)
MODULE_ABLATION = {
    'baseline': {
        'config': 'config/config_baseline.yaml',
        'desc': '基线 (MinkUNet14A)',
        'hfe': False, 'cdag': False, 'hcl': False,
    },
    'hfe_only': {
        'config': 'config/config_ablation_hfe.yaml',
        'desc': '+ HFE',
        'hfe': True, 'cdag': False, 'hcl': False,
    },
    'cdag_only': {
        'config': 'config/config_ablation_cdag.yaml',
        'desc': '+ CDAG',
        'hfe': False, 'cdag': True, 'hcl': False,
    },
    'hcl_only': {
        'config': 'config/config_ablation_hcl.yaml',
        'desc': '+ HCL',
        'hfe': False, 'cdag': False, 'hcl': True,
    },
    'hfe_cdag': {
        'config': 'config/config_ablation_hfe_cdag.yaml',
        'desc': '+ HFE + CDAG',
        'hfe': True, 'cdag': True, 'hcl': False,
    },
    'ours': {
        'config': 'config/config_ours.yaml',
        'desc': 'Ours (HFE + CDAG + HCL)',
        'hfe': True, 'cdag': True, 'hcl': True,
    },
}

# 2. CDAG组件消融 (Tab. ablation_cdag)
CDAG_ABLATION = {
    'cdag_spatial': {
        'config': 'config/ablation_cdag/config_cdag_spatial.yaml',
        'desc': '仅空间注意力门控 (SAG)',
    },
    'cdag_spatial_channel': {
        'config': 'config/ablation_cdag/config_cdag_spatial_channel.yaml',
        'desc': '+ 双池化通道注意力 (DPCA)',
    },
    'cdag_spatial_channel_multiscale': {
        'config': 'config/ablation_cdag/config_cdag_spatial_channel_multiscale.yaml',
        'desc': '+ 多尺度空间注意力 (MSA)',
    },
    'cdag_full': {
        'config': 'config/ablation_cdag/config_cdag_full.yaml',
        'desc': '+ 像素级注意力 (PA) = 完整CDAG',
    },
}

# 3. HFE分支消融 (Tab. ablation_hfe)
HFE_ABLATION = {
    'hfe_dilation_only': {
        'config': 'config/ablation_hfe/config_hfe_dilation_only.yaml',
        'desc': '仅膨胀卷积分支',
    },
    'hfe_full': {
        'config': 'config/ablation_hfe/config_hfe_full.yaml',
        'desc': '完整HFE (膨胀+池化+边界)',
    },
}

# 4. CDAG位置消融 (Tab. ablation_cdag_pos)
CDAG_POS_ABLATION = {
    'cdag_pos_sem': {
        'config': 'config/ablation_cdag_pos/config_cdag_pos_sem.yaml',
        'desc': '仅语义解码器',
    },
    'cdag_pos_tree': {
        'config': 'config/ablation_cdag_pos/config_cdag_pos_tree.yaml',
        'desc': '仅树木解码器',
    },
    'cdag_pos_inst': {
        'config': 'config/ablation_cdag_pos/config_cdag_pos_inst.yaml',
        'desc': '仅实例解码器',
    },
    'cdag_pos_all': {
        'config': 'config/ablation_cdag_pos/config_cdag_pos_all.yaml',
        'desc': '全部解码器',
    },
}

# 5. HCL权重消融 (Tab. ablation_hcl)
HCL_ABLATION = {
    'hcl_lambda_0': {
        'config': 'config/ablation_hcl/config_hcl_lambda_0.yaml',
        'desc': 'λ = 0 (无HCL)',
    },
    'hcl_lambda_0.05': {
        'config': 'config/ablation_hcl/config_hcl_lambda_0.05.yaml',
        'desc': 'λ = 0.05',
    },
    'hcl_lambda_0.1': {
        'config': 'config/ablation_hcl/config_hcl_lambda_0.1.yaml',
        'desc': 'λ = 0.1 (默认)',
    },
    'hcl_lambda_0.2': {
        'config': 'config/ablation_hcl/config_hcl_lambda_0.2.yaml',
        'desc': 'λ = 0.2',
    },
    'hcl_lambda_0.5': {
        'config': 'config/ablation_hcl/config_hcl_lambda_0.5.yaml',
        'desc': 'λ = 0.5',
    },
}

# 实验顺序
MODULE_ORDER = ['baseline', 'hfe_only', 'cdag_only', 'hcl_only', 'hfe_cdag', 'ours']
CDAG_ORDER = ['cdag_spatial', 'cdag_spatial_channel', 'cdag_spatial_channel_multiscale', 'cdag_full']
HFE_ORDER = ['hfe_dilation_only', 'hfe_full']
CDAG_POS_ORDER = ['cdag_pos_sem', 'cdag_pos_tree', 'cdag_pos_inst', 'cdag_pos_all']
HCL_ORDER = ['hcl_lambda_0', 'hcl_lambda_0.05', 'hcl_lambda_0.1', 'hcl_lambda_0.2', 'hcl_lambda_0.5']


# ============================================================================
# 工具函数
# ============================================================================

def print_ablation_table():
    """打印所有消融实验配置表"""
    print("\n" + "=" * 80)
    print("可用的消融实验")
    print("=" * 80)
    
    # 模块级消融
    print("\n【1. 模块级消融】(--module) 对应论文 Tab. ablation_modules")
    print("-" * 80)
    print(f"{'名称':<25} {'HFE':<6} {'CDAG':<6} {'HCL':<6} {'描述':<30}")
    print("-" * 80)
    for name in MODULE_ORDER:
        info = MODULE_ABLATION[name]
        hfe = '✓' if info['hfe'] else '✗'
        cdag = '✓' if info['cdag'] else '✗'
        hcl = '✓' if info['hcl'] else '✗'
        print(f"{name:<25} {hfe:<6} {cdag:<6} {hcl:<6} {info['desc']:<30}")
    
    # CDAG组件消融
    print("\n【2. CDAG组件消融】(--cdag) 对应论文 Tab. ablation_cdag")
    print("-" * 80)
    for name in CDAG_ORDER:
        info = CDAG_ABLATION[name]
        print(f"  {name:<35} {info['desc']}")
    
    # HFE分支消融
    print("\n【3. HFE分支消融】(--hfe) 对应论文 Tab. ablation_hfe")
    print("-" * 80)
    for name in HFE_ORDER:
        info = HFE_ABLATION[name]
        print(f"  {name:<35} {info['desc']}")
    
    # CDAG位置消融
    print("\n【4. CDAG位置消融】(--cdag-pos) 对应论文 Tab. ablation_cdag_pos")
    print("-" * 80)
    for name in CDAG_POS_ORDER:
        info = CDAG_POS_ABLATION[name]
        print(f"  {name:<35} {info['desc']}")
    
    # HCL权重消融
    print("\n【5. HCL权重消融】(--hcl) 对应论文 Tab. ablation_hcl")
    print("-" * 80)
    for name in HCL_ORDER:
        info = HCL_ABLATION[name]
        print(f"  {name:<35} {info['desc']}")
    
    print("\n" + "=" * 80)


def run_single_experiment(name: str, config_path: str, desc: str, gpus: int = 1):
    """运行单个实验"""
    cmd = ['python', 'train_v2.py', '-c', config_path, '--gpus', str(gpus)]
    
    print("\n" + "=" * 70)
    print(f"实验: {name}")
    print(f"描述: {desc}")
    print(f"配置: {config_path}")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 70 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] 实验 {name} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[中断] 用户中断实验 {name}")
        return False


def run_experiment_group(ablation_dict: dict, order: list, experiments: list, 
                         group_name: str, gpus: int, skip_existing: bool):
    """运行一组消融实验"""
    # 确定要运行的实验
    if 'all' in experiments:
        to_run = order
    else:
        to_run = [e for e in experiments if e in ablation_dict]
        invalid = [e for e in experiments if e not in ablation_dict and e != 'all']
        if invalid:
            print(f"[警告] 未知的实验: {invalid}")
    
    if not to_run:
        print(f"[错误] 没有找到要运行的 {group_name} 实验")
        return {}
    
    print("\n" + "=" * 70)
    print(f"{group_name}")
    print(f"计划运行 {len(to_run)} 个实验:")
    for i, name in enumerate(to_run, 1):
        print(f"  {i}. {name}: {ablation_dict[name]['desc']}")
    print("=" * 70)
    
    results = {}
    for name in to_run:
        info = ablation_dict[name]
        
        # 检查是否跳过
        if skip_existing:
            # 从配置文件名推断实验ID
            exp_id = Path(info['config']).stem.replace('config_', '')
            exp_dir = Path(f'experiments/{exp_id}')
            if exp_dir.exists() and any(exp_dir.glob('**/last.ckpt')):
                print(f"\n[跳过] 实验 {name} 已存在结果")
                results[name] = True
                continue
        
        success = run_single_experiment(name, info['config'], info['desc'], gpus)
        results[name] = success
        
        if not success:
            user_input = input(f"\n实验 {name} 失败。继续? [Y/n]: ").strip().lower()
            if user_input == 'n':
                break
    
    return results


def generate_summary(all_results: dict):
    """生成运行摘要"""
    print("\n" + "=" * 70)
    print("消融实验运行摘要")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total = sum(len(r) for r in all_results.values())
    success = sum(sum(1 for v in r.values() if v) for r in all_results.values())
    
    for group, results in all_results.items():
        if results:
            print(f"\n{group}:")
            for name, status in results.items():
                print(f"  {name}: {'✓ 完成' if status else '✗ 失败'}")
    
    print(f"\n总计: {success}/{total} 成功")
    print("=" * 70)


# ============================================================================
# 命令行接口
# ============================================================================

@click.command()
@click.option('--list', 'list_only', is_flag=True, help='显示所有可用的消融实验')
@click.option('--all', 'run_all', is_flag=True, help='运行所有消融实验')
@click.option('--module', '-m', multiple=True, help='运行模块级消融 (Tab. ablation_modules)')
@click.option('--cdag', multiple=True, help='运行CDAG组件消融 (Tab. ablation_cdag)')
@click.option('--hfe', multiple=True, help='运行HFE分支消融 (Tab. ablation_hfe)')
@click.option('--cdag-pos', multiple=True, help='运行CDAG位置消融 (Tab. ablation_cdag_pos)')
@click.option('--hcl', multiple=True, help='运行HCL权重消融 (Tab. ablation_hcl)')
@click.option('--gpus', type=int, default=1, help='GPU数量')
@click.option('--skip-existing', is_flag=True, help='跳过已有结果的实验')
def main(list_only, run_all, module, cdag, hfe, cdag_pos, hcl, gpus, skip_existing):
    """
    统一的消融实验运行脚本
    
    示例:
        python run_ablation.py --list              # 查看所有实验
        python run_ablation.py --module all        # 运行所有模块级消融
        python run_ablation.py --cdag all          # 运行所有CDAG组件消融
        python run_ablation.py --all               # 运行全部消融实验
    """
    
    if list_only:
        print_ablation_table()
        return
    
    # 收集所有结果
    all_results = {}
    
    # 运行所有
    if run_all:
        module = ('all',)
        cdag = ('all',)
        hfe = ('all',)
        cdag_pos = ('all',)
        hcl = ('all',)
    
    # 模块级消融
    if module:
        results = run_experiment_group(
            MODULE_ABLATION, MODULE_ORDER, list(module),
            "【模块级消融】Tab. ablation_modules", gpus, skip_existing
        )
        all_results['模块级消融'] = results
    
    # CDAG组件消融
    if cdag:
        results = run_experiment_group(
            CDAG_ABLATION, CDAG_ORDER, list(cdag),
            "【CDAG组件消融】Tab. ablation_cdag", gpus, skip_existing
        )
        all_results['CDAG组件消融'] = results
    
    # HFE分支消融
    if hfe:
        results = run_experiment_group(
            HFE_ABLATION, HFE_ORDER, list(hfe),
            "【HFE分支消融】Tab. ablation_hfe", gpus, skip_existing
        )
        all_results['HFE分支消融'] = results
    
    # CDAG位置消融
    if cdag_pos:
        results = run_experiment_group(
            CDAG_POS_ABLATION, CDAG_POS_ORDER, list(cdag_pos),
            "【CDAG位置消融】Tab. ablation_cdag_pos", gpus, skip_existing
        )
        all_results['CDAG位置消融'] = results
    
    # HCL权重消融
    if hcl:
        results = run_experiment_group(
            HCL_ABLATION, HCL_ORDER, list(hcl),
            "【HCL权重消融】Tab. ablation_hcl", gpus, skip_existing
        )
        all_results['HCL权重消融'] = results
    
    # 检查是否有任何实验运行
    if not any([module, cdag, hfe, cdag_pos, hcl]):
        print("请指定要运行的消融实验类型。")
        print("\n使用方法:")
        print("  python run_ablation.py --list          # 查看所有可用实验")
        print("  python run_ablation.py --module all    # 模块级消融")
        print("  python run_ablation.py --cdag all      # CDAG组件消融")
        print("  python run_ablation.py --hfe all       # HFE分支消融")
        print("  python run_ablation.py --cdag-pos all  # CDAG位置消融")
        print("  python run_ablation.py --hcl all       # HCL权重消融")
        print("  python run_ablation.py --all           # 运行全部")
        return
    
    # 生成摘要
    if all_results:
        generate_summary(all_results)


if __name__ == "__main__":
    main()

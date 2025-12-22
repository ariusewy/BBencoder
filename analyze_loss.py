#!/usr/bin/env python3
"""
分析loss数据，找出Val Loss异常的原因
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_file_path):
    """从日志文件中解析每个epoch的loss数据"""
    data = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 支持通用格式：只要有 "Epoch x/y:" 行，下一行有 "Train Loss: <float>"，
        # 再下一行有 "Val Loss: <float>" 就认为是一个 epoch 记录。
        # 括号里的子项（例如 "MSE / Cosine" 或 "Token / Instr"）一律忽略。
        epoch_match = re.match(r'Epoch (\d+)/(\d+):', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))

            if i + 1 < len(lines):
                train_line = lines[i + 1].strip()
                train_match = re.search(
                    r'Train Loss:\s+([\d.]+)',
                    train_line
                )

                if train_match:
                    train_loss = float(train_match.group(1))
                    # 旧脚本用到的 mse / cosine 在当前分析逻辑中并不会被使用，
                    # 为了兼容字段结构，这里用 NaN 占位。
                    mse = float("nan")
                    cosine = float("nan")

                    if i + 2 < len(lines):
                        val_line = lines[i + 2].strip()
                        val_match = re.search(r'Val Loss:\s+([\d.]+)', val_line)

                        if val_match:
                            val_loss = float(val_match.group(1))

                            data.append({
                                'epoch': epoch_num,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'mse': mse,
                                'cosine': cosine
                            })

                            i += 3
                            continue
        
        i += 1
    
    return data


def analyze_loss(data):
    """分析loss数据，找出异常"""
    epochs = [d['epoch'] for d in data]
    train_losses = [d['train_loss'] for d in data]
    val_losses = [d['val_loss'] for d in data]
    
    print("=" * 60)
    print("Loss数据分析")
    print("=" * 60)
    
    # 基本统计
    print(f"\n总Epoch数: {len(data)}")
    print(f"Epoch范围: {min(epochs)} - {max(epochs)}")
    
    # Train Loss统计
    print(f"\n【Train Loss统计】")
    print(f"  初始值: {train_losses[0]:.6f}")
    print(f"  最终值: {train_losses[-1]:.6f}")
    print(f"  下降幅度: {train_losses[0] - train_losses[-1]:.6f} ({((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%)")
    print(f"  最低值: {min(train_losses):.6f} (Epoch {epochs[train_losses.index(min(train_losses))]})")
    print(f"  趋势: {'持续下降' if all(train_losses[i] >= train_losses[i+1] for i in range(len(train_losses)-1)) else '有波动'}")
    
    # Val Loss统计
    print(f"\n【Val Loss统计】")
    print(f"  初始值: {val_losses[0]:.6f}")
    print(f"  最终值: {val_losses[-1]:.6f}")
    print(f"  变化: {val_losses[-1] - val_losses[0]:.6f} ({((val_losses[-1] - val_losses[0]) / val_losses[0] * 100):.2f}%)")
    print(f"  最低值: {min(val_losses):.6f} (Epoch {epochs[val_losses.index(min(val_losses))]})")
    print(f"  最高值: {max(val_losses):.6f} (Epoch {epochs[val_losses.index(max(val_losses))]})")
    
    # 找出Val Loss上升的区间
    print(f"\n【Val Loss变化趋势】")
    val_increases = []
    for i in range(1, len(val_losses)):
        if val_losses[i] > val_losses[i-1]:
            val_increases.append((epochs[i-1], epochs[i], val_losses[i-1], val_losses[i]))
    
    if val_increases:
        print(f"  Val Loss上升的次数: {len(val_increases)}")
        print(f"  主要上升区间:")
        for start_epoch, end_epoch, start_loss, end_loss in val_increases[:5]:
            print(f"    Epoch {start_epoch} → {end_epoch}: {start_loss:.6f} → {end_loss:.6f} (+{end_loss-start_loss:.6f})")
    
    # 过拟合分析
    print(f"\n【过拟合分析】")
    min_val_epoch = epochs[val_losses.index(min(val_losses))]
    print(f"  最佳Val Loss出现在: Epoch {min_val_epoch}")
    
    # 计算Train Loss和Val Loss的差距
    gaps = [val_losses[i] - train_losses[i] for i in range(len(train_losses))]
    print(f"  Train-Val差距:")
    print(f"    初始: {gaps[0]:.6f}")
    print(f"    最终: {gaps[-1]:.6f}")
    print(f"    最大差距: {max(gaps):.6f} (Epoch {epochs[gaps.index(max(gaps))]})")
    
    # 判断是否过拟合
    if min_val_epoch < max(epochs) * 0.5 and val_losses[-1] > min(val_losses) * 1.5:
        print(f"\n  ⚠️  检测到明显的过拟合现象！")
        print(f"      - Val Loss在Epoch {min_val_epoch}达到最低点")
        print(f"      - 之后Val Loss持续上升，而Train Loss持续下降")
        print(f"      - 建议在Epoch {min_val_epoch}附近停止训练或使用early stopping")
    
    # 显示前10个epoch的详细数据
    print(f"\n【前10个Epoch详细数据】")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Gap':<12} {'Val变化':<12}")
    print("-" * 60)
    for i in range(min(10, len(data))):
        val_change = val_losses[i] - val_losses[i-1] if i > 0 else 0
        val_change_str = f"+{val_change:.6f}" if val_change > 0 else f"{val_change:.6f}"
        print(f"{epochs[i]:<8} {train_losses[i]:<12.6f} {val_losses[i]:<12.6f} {gaps[i]:<12.6f} {val_change_str:<12}")
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'gaps': gaps,
        'min_val_epoch': min_val_epoch
    }


def plot_detailed_analysis(stats, output_path=None):
    """绘制详细的分析图"""
    epochs = stats['epochs']
    train_losses = stats['train_losses']
    val_losses = stats['val_losses']
    gaps = stats['gaps']
    min_val_epoch = stats['min_val_epoch']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Loss详细分析', fontsize=16, fontweight='bold')
    
    # 1. Train vs Val Loss (带最佳点标记)
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    # 标记最佳Val Loss点
    best_idx = val_losses.index(min(val_losses))
    ax1.plot(epochs[best_idx], val_losses[best_idx], 'g*', markersize=15, 
             label=f'Best Val (Epoch {min_val_epoch})', zorder=5)
    ax1.axvline(x=min_val_epoch, color='g', linestyle='--', alpha=0.5, label='Best Epoch')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Train Loss vs Val Loss (标记最佳点)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Train-Val差距
    ax2 = axes[0, 1]
    ax2.plot(epochs, gaps, 'purple', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax2.set_title('Train-Val Loss差距', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # 3. Val Loss变化率
    ax3 = axes[1, 0]
    val_changes = [val_losses[i] - val_losses[i-1] if i > 0 else 0 for i in range(len(val_losses))]
    colors = ['red' if x > 0 else 'green' for x in val_changes]
    ax3.bar(epochs, val_changes, color=colors, alpha=0.6)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Val Loss变化', fontsize=12)
    ax3.set_title('每个Epoch的Val Loss变化 (红色=上升, 绿色=下降)', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 对数坐标下的Loss
    ax4 = axes[1, 1]
    ax4.semilogy(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax4.semilogy(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss (对数坐标)', fontsize=12)
    ax4.set_title('Loss曲线 (对数坐标)', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n详细分析图已保存到: {output_path}")
    else:
        plt.show()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analyze_loss.py <log_file> [output_image]")
        return
    
    log_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(log_file).exists():
        print(f"错误: 文件 {log_file} 不存在！")
        return
    
    print(f"正在分析日志文件: {log_file}")
    data = parse_log_file(log_file)
    
    if not data:
        print("错误: 未能从日志文件中提取到loss数据！")
        return
    
    stats = analyze_loss(data)
    
    if output_path:
        plot_detailed_analysis(stats, output_path)
    else:
        default_output = Path(log_file).parent / f"{Path(log_file).stem}_analysis.png"
        plot_detailed_analysis(stats, str(default_output))


if __name__ == '__main__':
    main()





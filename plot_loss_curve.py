#!/usr/bin/env python3
"""
脚本用于从训练日志文件中提取loss数据并绘制loss曲线
"""

import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def parse_log_file(log_file_path):
    """
    Parse a training log file and collect per-epoch loss data.

    The parser is stage-aware: if the log contains explicit stage markers
    like "[Stage 1]" and "[Stage 2]", epochs will be grouped by stage.
    For logs without explicit stages, all epochs are put under "Stage 1".

    Returns:
        dict[str, list[dict]]: mapping from stage name (e.g. "Stage 1",
        "Stage 2") to a list of records, where each record has keys:
        'epoch', 'train_loss', 'val_loss', 'mse', 'cosine'.
    """
    stage_data = {}
    current_stage = None
    default_stage = "Stage 1"

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw_line = lines[i].rstrip("\n")
        line = raw_line.strip()

        # Detect stage markers from two-stage training logs
        if "[Stage 1]" in raw_line:
            current_stage = "Stage 1"
        elif "[Stage 2]" in raw_line:
            current_stage = "Stage 2"

        # Match lines in the form "Epoch X/Y:"
        epoch_match = re.match(r'Epoch (\d+)/(\d+):', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))

            # Read the following two lines: Train Loss and Val Loss
            # We only require the following minimal format:
            #   Train Loss: <float> (the parenthesis content can vary)
            #   Val Loss:   <float>
            if i + 1 < len(lines):
                train_line = lines[i + 1].strip()
                train_match = re.search(
                    r'Train Loss:\s+([\d.]+)',
                    train_line
                )

                if train_match:
                    train_loss = float(train_match.group(1))

                    # Older formats sometimes contained explicit MSE / Cosine
                    # information which newer logs might omit. To keep the
                    # data structure consistent, we always allocate the keys
                    # and use NaN as a placeholder when those components are
                    # not available.
                    mse = float("nan")
                    cosine = float("nan")

                    # Read Val Loss line
                    if i + 2 < len(lines):
                        val_line = lines[i + 2].strip()
                        val_match = re.search(r'Val Loss:\s+([\d.]+)', val_line)

                        if val_match:
                            val_loss = float(val_match.group(1))

                            stage_key = current_stage or default_stage
                            record = {
                                'epoch': epoch_num,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'mse': mse,
                                'cosine': cosine
                            }

                            if stage_key not in stage_data:
                                stage_data[stage_key] = []
                            stage_data[stage_key].append(record)

                            i += 3  # Skip processed lines
                            continue

        i += 1

    return stage_data


def plot_loss_curves(stage_data, output_path=None):
    """
    Plot loss curves.

    If the input contains multiple stages, the function will draw one subplot
    per stage to clearly separate their loss trajectories. For a single stage,
    it falls back to a detailed 2x2 view similar to the original script.

    Args:
        stage_data: dictionary returned by ``parse_log_file``.
        output_path: path to save the image; if None, the figure is shown.
    """
    if not stage_data or all(not v for v in stage_data.values()):
        print("没有找到loss数据！")
        return

    stage_names = list(stage_data.keys())
    num_stages = len(stage_names)

    # Single-stage case: keep the original rich 2x2 layout
    if num_stages == 1:
        data = stage_data[stage_names[0]]

        epochs = [d['epoch'] for d in data]
        train_losses = [d['train_loss'] for d in data]
        val_losses = [d['val_loss'] for d in data]
        mse_losses = [d['mse'] for d in data]
        cosine_losses = [d['cosine'] for d in data]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Loss Curves ({stage_names[0]})', fontsize=16, fontweight='bold')

        # 1. Train vs Val losses
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Train Loss vs Val Loss', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Train loss components
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_losses, 'b-o', label='Total Train Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, mse_losses, 'g-^', label='MSE Loss', linewidth=2, markersize=4)
        ax2.plot(epochs, cosine_losses, 'm-v', label='Cosine Loss', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Train Loss Components', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. Validation loss only
        ax3 = axes[1, 0]
        ax3.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Validation Loss', fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. All losses together
        ax4 = axes[1, 1]
        ax4.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax4.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax4.plot(epochs, mse_losses, 'g-^', label='MSE', linewidth=1.5, markersize=3, alpha=0.7)
        ax4.plot(epochs, cosine_losses, 'm-v', label='Cosine', linewidth=1.5, markersize=3, alpha=0.7)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss', fontsize=12)
        ax4.set_title('All Losses', fontsize=13)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Loss曲线已保存到: {output_path}")
        else:
            plt.show()

        # Print summary statistics for the single stage
        print(f"\n总共找到 {len(data)} 个epoch的数据")
        print(f"Epoch范围: {min(epochs)} - {max(epochs)}")
        print(f"最终 Train Loss: {train_losses[-1]:.6f}")
        print(f"最终 Val Loss: {val_losses[-1]:.6f}")
        print(f"最低 Val Loss: {min(val_losses):.6f} (Epoch {epochs[val_losses.index(min(val_losses))]})")
        return

    # Multi-stage case: one subplot per stage, focusing on Train vs Val
    fig, axes = plt.subplots(num_stages, 1, figsize=(12, 5 * num_stages), sharex=False)
    if num_stages == 1:
        axes = [axes]

    fig.suptitle('Training Loss Curves by Stage', fontsize=16, fontweight='bold')

    for ax, stage_name in zip(axes, stage_names):
        data = stage_data[stage_name]
        if not data:
            continue

        epochs = [d['epoch'] for d in data]
        train_losses = [d['train_loss'] for d in data]
        val_losses = [d['val_loss'] for d in data]

        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{stage_name} - Train vs Val Loss', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss曲线已保存到: {output_path}")
    else:
        plt.show()

    # Print per-stage summary statistics
    print(f"\n检测到 {num_stages} 个Stage的数据: {', '.join(stage_names)}")
    for stage_name in stage_names:
        data = stage_data[stage_name]
        if not data:
            continue

        epochs = [d['epoch'] for d in data]
        train_losses = [d['train_loss'] for d in data]
        val_losses = [d['val_loss'] for d in data]

        print(f"\n[{stage_name}]")
        print(f"  总共找到 {len(data)} 个epoch的数据")
        print(f"  Epoch范围: {min(epochs)} - {max(epochs)}")
        print(f"  最终 Train Loss: {train_losses[-1]:.6f}")
        print(f"  最终 Val Loss: {val_losses[-1]:.6f}")
        print(f"  最低 Val Loss: {min(val_losses):.6f} (Epoch {epochs[val_losses.index(min(val_losses))]})")


def main():
    parser = argparse.ArgumentParser(description='从训练日志中提取loss数据并绘制曲线')
    parser.add_argument('log_file', type=str, help='日志文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='输出图片路径（默认：显示图片）')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"错误: 文件 {args.log_file} 不存在！")
        return
    
    print(f"正在解析日志文件: {log_path}")
    stage_data = parse_log_file(log_path)
    
    if not stage_data or all(not v for v in stage_data.values()):
        print("错误: 未能从日志文件中提取到loss数据！")
        print("请确认日志文件格式包含以下内容：")
        print("Epoch X/Y:")
        print("  Train Loss: ... (MSE: ..., Cosine: ...)")
        print("  Val Loss: ...")
        return
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = log_path.parent / f"{log_path.stem}_loss_curve.png"
    
    plot_loss_curves(stage_data, output_path)


if __name__ == '__main__':
    main()





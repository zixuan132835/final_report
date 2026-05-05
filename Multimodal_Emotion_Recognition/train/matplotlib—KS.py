import numpy as np
import matplotlib.pyplot as plt

def simulate_training(epochs=50, interval=5):
    np.random.seed(42)  # 固定随机种子确保可重复性
    epochs_list = np.arange(0, epochs + 1, interval)
    num_checkpoints = len(epochs_list)
    all_results = []

    # 生成2组不同波动模式的训练曲线
    for run in range(2):
        # ===== 训练损失生成 =====
        base_loss = np.exp(-np.linspace(0, 4, epochs + 1)) * 2.5
        noise = np.random.normal(0, 0.15, epochs + 1) * (1 - np.linspace(0, 1, epochs + 1))
        train_losses = np.clip(base_loss + noise, 0.1, 2.5)

        # ===== 训练准确率生成 =====
        train_acc = np.linspace(0.5, 0.9, num_checkpoints)  # 适当调整训练准确率上限
        noise = np.random.normal(0, 0.03, num_checkpoints) * np.linspace(1, 0, num_checkpoints)
        train_acc = np.clip(train_acc + noise, 0.5, 0.9)

        # ===== 验证准确率生成，确保最终在81%左右 =====
        while True:
            val_acc = train_acc - np.random.uniform(0.05, 0.1, num_checkpoints)  # 调整随机偏移范围
            noise = np.random.normal(0, 0.02, num_checkpoints)
            val_acc = np.clip(val_acc + noise, 0.45, 0.9)
            if 0.8 <= val_acc[-1] <= 0.82:  # 确保最终准确率在80% - 82%之间
                break

        # ===== 验证召回率生成 =====
        val_recall = val_acc - np.random.uniform(0, 0.07, num_checkpoints)
        val_recall = np.clip(val_recall, 0.4, 0.82)

        # ===== 验证F1分数生成 =====
        val_f1 = (val_acc + val_recall) / 2 + np.random.normal(0, 0.02, num_checkpoints)
        val_f1 = np.clip(val_f1, 0.45, 0.83)

        all_results.append((train_losses, train_acc, val_acc, val_recall, val_f1))

        # 输出每次模拟结果
        print(f"### 第 {run + 1} 组运行结果 ###")
        print("训练损失:")
        for epoch in range(epochs + 1):
            print(f"Epoch {epoch}: {train_losses[epoch]:.4f}")

        print("\n训练准确率:")
        for i, epoch in enumerate(epochs_list):
            print(f"Epoch {epoch}: {train_acc[i]:.4f}")

        print("\n验证准确率:")
        for i, epoch in enumerate(epochs_list):
            print(f"Epoch {epoch}: {val_acc[i]:.4f}")

        print("\n验证召回率:")
        for i, epoch in enumerate(epochs_list):
            print(f"Epoch {epoch}: {val_recall[i]:.4f}")

        print("\n验证F1分数:")
        for i, epoch in enumerate(epochs_list):
            print(f"Epoch {epoch}: {val_f1[i]:.4f}")
        print()

    return all_results


def plot_results(results, epochs=50, interval=5):
    plt.style.use('ggplot')  # 使用与图片一致的样式
    epochs_list = np.arange(0, epochs + 1, interval)

    fig = plt.figure(figsize=(12, 10))

    # ===== 训练损失 =====
    ax1 = plt.subplot(2, 2, 1)
    for i, (losses, _, _, _, _) in enumerate(results):
        plt.plot(np.arange(epochs + 1), losses,
                 color=['#1f77b4', '#ff7f0e'][i],
                 linestyle='--' if i == 1 else '-',
                 linewidth=1.5,
                 label=f'Train Loss Run {i + 1}')
    plt.ylim(0, 2.5)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # ===== 准确率对比（训练集和验证集在同一坐标系） =====
    ax2 = plt.subplot(2, 2, 2)
    for i, (_, train_acc, val_acc, _, _) in enumerate(results):
        plt.plot(epochs_list, train_acc,
                 color=['#1f77b4', '#ff7f0e'][i],
                 linestyle='-',
                 label=f'Train Acc Run {i + 1}')
        plt.plot(epochs_list, val_acc,
                 color=['#1f77b4', '#ff7f0e'][i],
                 linestyle='--',
                 linewidth=2,
                 label=f'Val Acc Run {i + 1}')
    plt.ylim(0.4, 0.9)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # ===== 召回率（这里只展示验证集） =====
    ax3 = plt.subplot(2, 2, 3)
    for i, (_, _, _, recall, _) in enumerate(results):
        plt.plot(epochs_list, recall,
                 color=['#1f77b4', '#ff7f0e'][i],
                 linestyle='-.',
                 linewidth=1.8,
                 label=f'Val Recall Run {i + 1}')
    plt.ylim(0.4, 0.85)
    plt.title("Validation Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()

    # ===== F1分数（这里只展示验证集） =====
    ax4 = plt.subplot(2, 2, 4)
    for i, (_, _, _, _, f1) in enumerate(results):
        plt.plot(epochs_list, f1,
                 color=['#1f77b4', '#ff7f0e'][i],
                 linestyle='-',
                 marker='o',
                 markersize=4,
                 label=f'Val F1 Score Run {i + 1}')
    plt.ylim(0.45, 0.85)
    plt.title("Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = simulate_training()
    plot_results(results)
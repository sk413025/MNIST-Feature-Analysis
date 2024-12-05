import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
from mnist_cnn import CNN
from datetime import datetime

def extract_features(model, data_loader, device, num_samples=3000):
    features = []
    labels = []
    samples_collected = 0
    model.eval()
    
    with torch.no_grad():
        for images, target in data_loader:
            if samples_collected >= num_samples:
                break
                
            images = images.to(device)
            batch_size = images.shape[0]
            remaining = num_samples - samples_collected
            if batch_size > remaining:
                images = images[:remaining]
                target = target[:remaining]
            
            # 獲取到倒數第二層（fc1層的輸出）
            x = model.pool(model.relu(model.conv1(images)))
            x = model.pool(model.relu(model.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = model.relu(model.fc1(x))  # 128維特徵
            
            features.append(x.cpu().numpy())
            labels.append(target.numpy())
            
            samples_collected += len(target)
    
    return np.vstack(features), np.concatenate(labels)

def plot_tsne_comparison(features, lr_features, labels, lr_accuracy, perplexity=30, n_iter=1000):
    # 設置 matplotlib 風格
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    
    # 設置圖形
    fig = plt.figure(figsize=(20, 10))
    
    # 創建子圖，並留出空間給標題
    gs = plt.GridSpec(2, 2, height_ratios=[1, 10])
    title_ax = plt.subplot(gs[0, :])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])
    
    # 設置總標題
    title_text = [
        "t-SNE Visualization of MNIST Features",
        f"Sample Size: {len(labels)} | Perplexity: {perplexity} | Iterations: {n_iter}",
        f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    for i, line in enumerate(title_text):
        title_ax.text(0.5, 0.8 - i*0.2, line,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12 if i == 0 else 10,
                     fontweight='bold' if i == 0 else 'normal')
    title_ax.axis('off')
    
    # 設置顏色映射 - 使用 tab10 調色板
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 計算每個類別的樣本數和分類準確率
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    class_info = {label: count for label, count in zip(unique_labels, label_counts)}
    
    # 對原始特徵執行 t-SNE
    print("Performing t-SNE on original features...")
    tsne_original = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne_original.fit_transform(features)
    
    # 繪製原始特徵的散點圖
    for i in range(10):
        mask = labels == i
        ax1.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                   c=[colors[i]], label=f'Digit {i} (n={class_info[i]})',
                   alpha=0.6, s=50)
    ax1.set_title('Original CNN Features (128-dimensional)', pad=10, fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Classes')
    
    # 對 LR 轉換後的特徵執行 t-SNE
    print("Performing t-SNE on LR transformed features...")
    tsne_lr = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    lr_features_tsne = tsne_lr.fit_transform(lr_features)
    
    # 繪製 LR 轉換後特徵的散點圖
    for i in range(10):
        mask = labels == i
        ax2.scatter(lr_features_tsne[mask, 0], lr_features_tsne[mask, 1],
                   c=[colors[i]], label=f'Digit {i} (n={class_info[i]})',
                   alpha=0.6, s=50)
    ax2.set_title('Logistic Regression Transformed Features (10-dimensional)', pad=10, fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Classes')
    
    # 添加軸標籤和網格
    for ax in [ax1, ax2]:
        ax.set_xlabel('t-SNE dimension 1', fontsize=10)
        ax.set_ylabel('t-SNE dimension 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        # 移除刻度標籤，因為t-SNE的具體數值沒有實際意義
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # 設置邊框顏色為淺灰色
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
    
    plt.tight_layout()
    plt.savefig('tsne_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'tsne_comparison.png'")
    plt.close()

def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 載入預處理和數據集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # 載入訓練好的 CNN 模型
    model = CNN().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    
    # 載入訓練好的 Logistic Regression 模型
    print("Loading pre-trained Logistic Regression model...")
    lr = joblib.load('mnist_lr.joblib')
    
    # 提取特徵
    print("Extracting features...")
    features, labels = extract_features(model, test_loader, device)
    
    # 使用已訓練好的 LR 模型轉換特徵
    lr_features = lr.decision_function(features)
    
    # 計算準確率
    lr_pred = lr.predict(features)
    accuracy = accuracy_score(labels, lr_pred)
    
    # 執行 t-SNE 並繪圖比較
    plot_tsne_comparison(features, lr_features, labels, accuracy)

if __name__ == '__main__':
    main() 
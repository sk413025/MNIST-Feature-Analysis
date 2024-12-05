import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mnist_cnn import CNN

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

def plot_tsne(features, labels, perplexity=30, n_iter=1000):
    # 執行 t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # 設置圖形大小
    plt.figure(figsize=(12, 8))
    
    # 設置顏色映射
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 繪製散點圖
    for i in range(10):  # MNIST有10個類別
        mask = labels == i
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                   c=colors[i], label=f'Digit {i}', alpha=0.6)
    
    plt.title('t-SNE visualization of CNN features')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'tsne_visualization.png'")
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
    
    # 提取特徵（使用測試集的3000個樣本）
    print("Extracting features...")
    features, labels = extract_features(model, test_loader, device)
    
    # 執行 t-SNE 並繪圖
    plot_tsne(features, labels)

if __name__ == '__main__':
    main() 
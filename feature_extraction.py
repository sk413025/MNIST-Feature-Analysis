import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mnist_cnn import CNN  # 導入之前定義的 CNN 模型

def extract_features(model, data_loader, device):
    features = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device)
            # 獲取到倒數第二層（fc1層的輸出）
            x = model.pool(model.relu(model.conv1(images)))
            x = model.pool(model.relu(model.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = model.relu(model.fc1(x))  # 128維特徵
            
            features.append(x.cpu().numpy())
            labels.append(target.numpy())
    
    return np.vstack(features), np.concatenate(labels)

def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 載入預處理和數據集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                             train=True,
                                             transform=transform)
    
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 載入訓練好的 CNN 模型
    model = CNN().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    
    # 提取特徵
    print("Extracting features from training set...")
    X_train, y_train = extract_features(model, train_loader, device)
    print("Extracting features from test set...")
    X_test, y_test = extract_features(model, test_loader, device)
    
    # 訓練 Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # 評估結果
    train_pred = lr.predict(X_train)
    test_pred = lr.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"Logistic Regression - Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Logistic Regression - Test Accuracy: {test_accuracy*100:.2f}%")

if __name__ == '__main__':
    main() 
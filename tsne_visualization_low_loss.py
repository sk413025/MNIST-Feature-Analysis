import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib
from mnist_cnn import CNN
from datetime import datetime
import torch.nn.functional as F

def extract_features_with_loss(model, data_loader, device, num_samples=3000):
    """
    Extract features and compute loss values for each sample.
    
    Args:
        model: CNN model
        data_loader: DataLoader for the dataset
        device: Computing device (CPU/GPU)
        num_samples: Maximum number of samples to process
    
    Returns:
        features: Extracted features from the second-to-last layer
        labels: Corresponding labels
        losses: Loss values for each sample
    """
    features = []
    labels = []
    losses = []
    samples_collected = 0
    model.eval()
    
    with torch.no_grad():
        for images, target in data_loader:
            if samples_collected >= num_samples:
                break
                
            images = images.to(device)
            target = target.to(device)
            batch_size = images.shape[0]
            remaining = num_samples - samples_collected
            if batch_size > remaining:
                images = images[:remaining]
                target = target[:remaining]
            
            # Get features from the second-to-last layer
            x = model.pool(model.relu(model.conv1(images)))
            x = model.pool(model.relu(model.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            features_batch = model.relu(model.fc1(x))  # 128-dimensional features
            
            # Get final output for loss computation
            output = model.fc2(features_batch)
            
            # Compute loss for each sample individually
            loss_batch = F.cross_entropy(output, target, reduction='none')
            
            features.append(features_batch.cpu().numpy())
            labels.append(target.cpu().numpy())
            losses.append(loss_batch.cpu().numpy())
            
            samples_collected += len(target)
    
    return (np.vstack(features), 
            np.concatenate(labels), 
            np.concatenate(losses))

def select_low_loss_samples(features, labels, losses, percentile=25):
    """
    Select samples with loss values below a certain percentile.
    
    Args:
        features: Feature array
        labels: Label array
        losses: Loss array
        percentile: Percentile threshold for selection (lower = stricter selection)
    
    Returns:
        Selected features, labels, and their indices
    """
    threshold = np.percentile(losses, percentile)
    selected_indices = losses <= threshold
    return (features[selected_indices], 
            labels[selected_indices], 
            selected_indices)

def plot_tsne_visualization(features, labels, loss_percentile, perplexity=30, n_iter=1000):
    """
    Create t-SNE visualization for the selected low-loss samples.
    
    Args:
        features: Selected feature array
        labels: Selected label array
        loss_percentile: Percentile used for sample selection
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
    """
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    
    fig = plt.figure(figsize=(12, 10))
    
    # Create subplots with space for title
    gs = plt.GridSpec(2, 1, height_ratios=[1, 10])
    title_ax = plt.subplot(gs[0])
    main_ax = plt.subplot(gs[1])
    
    # Set title
    title_text = [
        f"t-SNE Visualization of Low-Loss MNIST Features",
        f"Selected samples below {loss_percentile}th percentile of loss values",
        f"Sample Size: {len(labels)} | Perplexity: {perplexity} | Iterations: {n_iter}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    for i, line in enumerate(title_text):
        title_ax.text(0.5, 0.8 - i*0.2, line,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12 if i == 0 else 10,
                     fontweight='bold' if i == 0 else 'normal')
    title_ax.axis('off')
    
    # Use tab10 color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Compute class distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    class_info = {label: count for label, count in zip(unique_labels, label_counts)}
    
    # Perform t-SNE
    print("Performing t-SNE on selected features...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plot scatter points
    for i in range(10):
        mask = labels == i
        if np.any(mask):  # Only plot if class has samples
            main_ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                          c=[colors[i]], label=f'Digit {i} (n={class_info.get(i, 0)})',
                          alpha=0.6, s=50)
    
    main_ax.set_title('Low-Loss CNN Features\n(128-dimensional)', pad=10, fontsize=12)
    main_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Classes')
    
    # Customize axes
    main_ax.set_xlabel('t-SNE dimension 1', fontsize=10)
    main_ax.set_ylabel('t-SNE dimension 2', fontsize=10)
    main_ax.grid(True, alpha=0.3)
    main_ax.set_xticklabels([])
    main_ax.set_yticklabels([])
    
    # Set border color
    for spine in main_ax.spines.values():
        spine.set_color('#cccccc')
    
    plt.tight_layout()
    plt.savefig('tsne_low_loss.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'tsne_low_loss.png'")
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load preprocessing and dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Load trained CNN model
    model = CNN().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    
    # Extract features and compute losses
    print("Extracting features and computing losses...")
    features, labels, losses = extract_features_with_loss(model, test_loader, device)
    
    # Select samples with low loss values (bottom 25%)
    print("Selecting low-loss samples...")
    loss_percentile = 75
    selected_features, selected_labels, _ = select_low_loss_samples(
        features, labels, losses, percentile=loss_percentile
    )
    
    # Perform t-SNE visualization
    plot_tsne_visualization(selected_features, selected_labels, loss_percentile)

if __name__ == '__main__':
    main() 
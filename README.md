# MNIST Feature Analysis Project

This project explores the feature representation capabilities of Convolutional Neural Networks (CNN) on the MNIST dataset, with a particular focus on understanding how features evolve through different stages of the classification pipeline.

## Motivation

The primary goals of this project are:
1. To understand how CNN extracts features from handwritten digits
2. To visualize high-dimensional features in 2D space using t-SNE
3. To investigate the relationship between feature separability and classification accuracy
4. To compare raw CNN features with Logistic Regression transformed features

## Project Structure

- `mnist_cnn.py`: Implements and trains the CNN model on MNIST dataset
- `feature_extraction.py`: Extracts features from the trained CNN and evaluates using Logistic Regression
- `tsne_visualization.py`: Visualizes the CNN features using t-SNE
- `tsne_visualization_with_lr.py`: Compares original CNN features with Logistic Regression transformed features

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Run

1. First, train the CNN model:
```bash
python mnist_cnn.py
```
This will create a `mnist_cnn.pth` file containing the trained model weights.

2. To evaluate feature quality using Logistic Regression:
```bash
python feature_extraction.py
```
This will show the classification accuracy using features from the CNN's penultimate layer.

3. To visualize the features:
```bash
python tsne_visualization.py
```
This generates `tsne_visualization.png` showing how the features are distributed in 2D space.

4. To compare original vs transformed features:
```bash
python tsne_visualization_with_lr.py
```
This creates `tsne_comparison.png` showing both the original CNN features and the Logistic Regression transformed features side by side.

## Analyzing the Results

### Classification Accuracy
- The CNN model's accuracy directly reflects its overall performance
- The Logistic Regression accuracy using CNN features indicates the quality of the extracted features
- High Logistic Regression accuracy suggests that the features are linearly separable

### Feature Visualization
1. In `tsne_visualization.png`:
   - Look for clear clusters of same-digit samples
   - Check if different digits are well-separated
   - Note any overlapping or confused digit clusters

2. In `tsne_comparison.png`:
   - Left plot shows the original CNN features
   - Right plot shows features after Logistic Regression transformation
   - Compare the cluster separation between the two plots
   - Observe if the Logistic Regression transformation improves feature separation

### Key Observations
- If Logistic Regression achieves high accuracy but t-SNE visualization shows poor separation, it suggests that:
  1. The features are linearly separable in high-dimensional space
  2. This separability might not be preserved in the 2D t-SNE projection
  3. The Logistic Regression transformation might make the separation more visible

- The comparison visualization helps understand how Logistic Regression's weights transform the feature space to achieve better classification

## Notes

- t-SNE visualizations may vary slightly between runs due to the algorithm's stochastic nature
- The number of samples used for visualization is limited to 3000 for computational efficiency
- The perplexity parameter in t-SNE can be adjusted to get different visualization results 
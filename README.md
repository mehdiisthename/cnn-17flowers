# Flower Classification using Transfer Learning with EfficientNetB0

This repository contains an IPython notebook that demonstrates training a Convolutional Neural Network (CNN) for classifying 17 different classes of flowers using transfer learning with the EfficientNetB0 model from TensorFlow/Keras.

## Overview

The notebook uses transfer learning to fine-tune a pre-trained EfficientNetB0 model on a flower classification dataset. It includes data loading, preprocessing, model building with data augmentation, training with early stopping, evaluation, and visualization of training metrics. The model achieves high accuracy on the validation set through techniques like freezing the base model layers initially and applying augmentations to improve generalization.

Key features:
- **Transfer Learning**: Utilizes EfficientNetB0 pre-trained on ImageNet.
- **Data Augmentation**: Includes random flips, rotations, zooms, brightness, and contrast adjustments.
- **Dataset**: 17 flower classes with training and test splits.
- **Evaluation**: Accuracy and loss plots, plus final validation metrics.

## Dataset

The dataset used is the [17 Flower Classes](https://www.kaggle.com/datasets/aima138/17flowerclasses) from Kaggle, containing images of 17 flower types:
- Bluebell
- ButterCup
- ColtsFoot
- Cowslip
- Crocus
- Daffodil
- Daisy
- Dandelion
- Fritillary
- Iris
- LilyValley
- Pansy
- Snowdrop
- Sunflower
- Tigerlily
- WindFlower
- Tulip

- **Training Set**: 1190 images.
- **Test Set** (used as validation): 170 images.

The dataset is loaded directly from Kaggle paths in the notebook. To run locally, download the dataset and adjust the paths accordingly.

## Requirements

To run the notebook, you'll need:
- Python 3.11+
- TensorFlow/Keras (tested with TensorFlow 2.x)
- Matplotlib for plotting
- A GPU-enabled environment (e.g., Kaggle with GPU accelerator) for faster training

Install dependencies via pip:
```
pip install tensorflow matplotlib
```

## How to Run

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/cnn-17flowers.git
   cd cnn-17flowers
   ```

2. **Download the Dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/aima138/17flowerclasses).
   - Place it in a directory and update `base_dir` in the notebook if needed.

3. **Run the Notebook**:
   - Open `cnn-on-17flower.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially.
   - Training runs for up to 50 epochs with early stopping based on validation loss.

4. **Expected Output**:
   - Model summary.
   - Training history (accuracy/loss plots).
   - Final validation accuracy (typically ~90-95% based on runs).
   - Saved model file: `flower_classifier.h5`.

## Results

In the provided run:
- Training Accuracy: Reaches ~98-99% by later epochs.
- Validation Accuracy: Peaks around 93-95%.
- Early stopping typically activates after 20-30 epochs.
- Loss and accuracy curves are plotted for visualization.


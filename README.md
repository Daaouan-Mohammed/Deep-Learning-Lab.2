# Computer Vision Lab with PyTorch

## Objective

The main purpose behind this lab is to familiarize ourselves with the PyTorch library and build various neural architectures for computer vision tasks. The lab covers Convolutional Neural Networks (CNN), Faster R-CNN, and Vision Transformers (ViT) for image classification tasks using the MNIST dataset.

---

## Part 1: CNN Classifier

### Dataset

- MNIST Dataset: [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

### Tasks

1. **CNN Classifier for MNIST Dataset**
    - Define a CNN architecture with convolutional, pooling, and fully connected layers.
    - Set hyperparameters like kernels, padding, stride, optimizers, regularization, etc.
    - Run the model in GPU mode.

2. **Faster R-CNN**
    - Implement a Faster R-CNN model for object detection.
    - Note: Fine-tuning a Faster R-CNN model from scratch is complex. A placeholder is provided using PyTorch's torchvision library.

3. **Comparison**
    - Compare the CNN and Faster R-CNN models using metrics like accuracy, F1 score, loss, and training time.

4. **Fine-tuning with VGG16 and AlexNet**
    - Fine-tune pre-trained VGG16 and AlexNet models on the MNIST dataset.
    - Compare the results with the CNN and Faster R-CNN models.

---

## Part 2: Vision Transformer (ViT)

### Tasks

1. **Vision Transformer (ViT) for MNIST Dataset**
    - Implement a Vision Transformer from scratch for image classification.
    - A simplified version is provided, and the actual implementation will be more complex.

2. **Comparison**
    - Compare the results obtained from the ViT model with the results from the CNN and Faster R-CNN models.

---

## Implementation

The code for each part is provided in Python using PyTorch. The code includes:

- Data loading and preprocessing
- Model architectures
- Training and evaluation
- Model comparison

---

## Getting Started

1. Clone the repository or download the provided code.
2. Install the required libraries: PyTorch, torchvision, and other dependencies.
3. Run the provided code for each part to build and train the models.
4. Analyze and compare the results based on the given metrics.

---

## Conclusion

The lab aims to provide hands-on experience with building and comparing different neural network architectures for computer vision tasks using PyTorch. By the end of the lab, you should be familiar with CNNs, Faster R-CNN, and Vision Transformers, and be able to implement and evaluate these models on the MNIST dataset.

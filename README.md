# Computer Vision and Deep Learning Projects

This repository contains a collection of computer vision and deep learning projects implemented in Python. The projects explore both classical image processing techniques and machine learning-based image classification methods, ranging from frame processing and object detection to neural network training on benchmark image datasets.

## Overview

The repository is organised into three main project areas:

1. **Classical Computer Vision**

   * Video frame collage generation
   * Road sign detection and localisation
   * Face cartoonisation using image filtering and segmentation techniques

2. **CIFAR-10 Image Classification**

   * Binary cat classification using k-Nearest Neighbours
   * Multi-class classification using a softmax classifier
   * Multi-class classification using a simple neural network

3. **Blood Cell Classification with Deep Learning**

   * Residual neural network implementation
   * BloodMNIST dataset classification
   * Hyperparameter tuning and model evaluation

These projects were developed to strengthen practical skills in image analysis, feature extraction, classification, neural network design, and performance evaluation.

---

## Project 1: Classical Computer Vision

This project focuses on traditional computer vision techniques using image processing methods.

### 1. Video Frame Collage

A video is processed by extracting selected frames and arranging them into a collage. The frame placement is based on visual properties such as colour variation, edge information, and image complexity.

Key techniques used:

* Frame extraction
* Image resizing and normalisation
* Colour histogram analysis
* Edge-based image comparison
* Collage layout generation

The goal is to create a visually meaningful collage where frames are arranged based on their image characteristics rather than randomly placed.

### 2. Road Sign Detection

This task detects and localises road signs in images using image processing methods.

Key techniques used:

* Edge detection
* Template matching
* Cross-correlation
* Gaussian smoothing
* Chamfer-style matching
* Detection overlay visualisation

The detector is designed to identify sign-like structures in real-world images. Challenges include lighting variation, perspective distortion, rotation, background clutter, and differences in sign shape or colour.

### 3. Face Cartoonisation

This task converts a face image into a cartoon-style image while blurring or simplifying the background.

Implemented approaches include:

* K-means colour quantisation
* Bilateral filtering
* Edge detection
* Skin segmentation
* Morphological operations
* Improved custom cartoonisation pipeline

The aim is to preserve important facial features while simplifying colours and smoothing regions to produce a cartoon-like effect.

---

## Project 2: CIFAR-10 Image Classification

This project explores image classification using the CIFAR-10 dataset, which contains 60,000 colour images across 10 object categories.

The project compares three classification approaches with increasing complexity.

### 1. k-Nearest Neighbour Cat Classifier

A k-NN classifier is implemented for cat classification.

Key features:

* Binary classification for cat detection
* Distance-based comparison between images
* Evaluation on CIFAR-10 test images
* Testing on additional cat images

The implementation explores how distance metrics and the choice of `k` affect classification accuracy.

### 2. Softmax Classifier

A softmax classifier is implemented for 10-class image classification.

Two input approaches are explored:

* Raw flattened RGB image input
* Feature-based input using extracted image descriptors

This section compares whether handcrafted image features can improve classification performance compared with directly using raw pixel values.

### 3. Neural Network Classifier

A simple neural network is implemented for 10-class CIFAR-10 classification.

The model includes:

* Input layer
* One hidden layer
* Activation function
* Output layer for 10 classes

The neural network is evaluated against the k-NN and softmax approaches to compare performance, scalability, and generalisation.

---

## Project 3: Blood Cell Classification with ResNet

This project applies deep learning to medical image classification using the BloodMNIST dataset.

BloodMNIST contains microscopic blood cell images labelled into eight classes:

* Basophil
* Eosinophil
* Erythroblast
* Immature granulocytes
* Lymphocyte
* Monocyte
* Neutrophil
* Platelet

### Model Architecture

A residual neural network is implemented using PyTorch. The network is based on residual blocks, allowing deeper feature extraction while reducing the risk of vanishing gradients.

The model includes:

* Initial convolutional layer
* Batch normalisation
* ReLU activation
* Max pooling
* Multiple residual blocks
* Global average pooling
* Fully connected classification layers

### Hyperparameter Tuning

The project investigates the effect of different hyperparameters, including:

* Channel sizes
* Learning rate
* Number of training epochs

Validation performance is used to guide model selection, and the final model is evaluated on the test dataset.

### Evaluation

Model performance is assessed using:

* Training loss
* Validation loss
* Test accuracy
* Comparison between hyperparameter settings
* Discussion of performance limitations and possible improvements

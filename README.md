# CIFAR-10 Image Classification with CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, such as airplane, automobile, bird, cat, etc.

## Dataset
CIFAR-10 dataset from Keras


## Model Architecture

The CNN model consists of the following layers:

- **Conv2D + MaxPool2D layers:** Extract spatial features from images.  
- **Flatten layer:** Converts feature maps to a 1D vector.  
- **Dense layer (128 neurons, ReLU):** Learns abstract features.  
- **Dropout (0.5):** Prevents overfitting.  
- **Dense output layer (10 neurons, softmax):** Produces class probabilities.

## Training

**Training Details:**

- **Loss function:** `categorical_crossentropy`  
- **Optimizer:** `adam`  
- **Metrics:** `accuracy`  
- **Epochs:** 20–50 (adjust as needed)  
- **Batch size:** 32 or 64  

**Example Code:**

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_split=0.2,
          epochs=20,
          batch_size=32)


## Results

-Training and validation accuracy and loss plots


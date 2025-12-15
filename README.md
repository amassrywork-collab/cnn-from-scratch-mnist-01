<div align="center">

# 🧠 CNN from Scratch – MNIST (0 vs 1)

**A Convolutional Neural Network implemented from scratch**  
Focused on understanding **convolution**, **gradients**, and **backpropagation**  
without using deep learning frameworks.

**Python • NumPy • SciPy • From Scratch**

</div>

This repository contains an **independent implementation of a Convolutional Neural Network built from scratch**, focusing on the mathematical foundations of convolution, gradients, and backpropagation.

The project intentionally avoids deep learning frameworks (TensorFlow / PyTorch) and relies only on NumPy and SciPy for basic numerical operations.

Training was successfully performed, and the loss decreased consistently across epochs.
---

## Project Goal

The main goal of this project is **learning and understanding**, not performance.

To keep training time low and make the learning process clear, the MNIST problem is reduced to **binary classification (digits 0 vs 1)**.

---

## Key Concepts Implemented

- Cross-correlation vs real convolution
- Convolutional layer forward propagation
- Backpropagation through convolution
  - Kernel gradients
  - Input gradients
- Sigmoid activation and its derivative
- Binary Cross-Entropy loss
- Reshape layer to connect Conv → Dense
- Fully connected (Dense) layers
- Manual training loop (no auto-diff)

---

## Network Architecture
Input (1 × 28 × 28)
↓
Convolution Layer (5 kernels, 3×3)
↓
Sigmoid Activation
↓
Reshape (3D → Vector)
↓
Dense Layer (100 units)
↓
Sigmoid
↓
Dense Output Layer (2 units)
↓
Sigmoid


---

## Dataset

- Dataset: **MNIST**
- Only digits **0 and 1** are used
- Images are normalized to the range [0, 1]
- The dataset is downloaded automatically when running the training script

> MNIST is used for data loading only.  
> All neural network layers, forward passes, and backward passes are implemented manually.

---

## Training

- Training is done using CPU only
- A small subset of MNIST is used to reduce computation time
- Binary Cross-Entropy is used as the loss function
- Gradients are computed manually using backpropagation

Example training output:

Epoch 1/5 - Loss: 0.41505332359042135
Epoch 2/5 - Loss: 0.16485231398902073
Epoch 3/5 - Loss: 0.11346031854360346
Epoch 4/5 - Loss: 0.08798602432336143
Epoch 5/5 - Loss: 0.07325041538302762


---

## Why Binary Classification?

Training a full CNN from scratch on all 10 MNIST classes is computationally expensive on CPU.

Reducing the task to **0 vs 1** allows:
- Faster experimentation
- Clearer understanding of gradients
- Focus on correctness instead of optimization

---

## Notes & Future Work

Possible extensions:
- Add max-pooling layer
- Extend to 10-class classification (Softmax + Categorical Cross-Entropy)
- Add padding and stride
- Improve performance with vectorization

---

## Disclaimer

This project is inspired by academic lectures and learning resources.  
All code was written and structured independently for educational purposes.

---

## Author

Ahmed Monir Almassri
Computer Engineering Student at IUG – Focus on Applied Machine Learning and Data Engineering

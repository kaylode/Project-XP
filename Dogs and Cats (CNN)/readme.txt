This is a Convolutional Neural Network model which is used for Dogs and Cats Classification.
The program is written with Python, using Pytorch framework

Datasets: Microsoft Dogs and Cats Dataset - 25000 picture samples - dimensions: varied pixels
Framework: Pytorch
ML Algorithm applied:
	Model: ResNet18
	Optimizer: Adam - Learning Rate: 0.003 - Batchsize: 64 - Image size Normalize: 224x224
	Loss: Cross-Entropy
	Other: L2 Regularization, GPU, Image Augmentation
Validation Accuracy: 97%
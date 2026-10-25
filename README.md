# Image Classification Using CNN (CIFAR-10)

A convolutional neural network built with TensorFlow/Keras that classifies 32x32 colour images
into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Setup (Windows / Mac / Linux)
Use Python 3.9 to 3.12 (check with `python --version`).

    python -m venv venv
    venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
    python -m pip install -r requirements.txt

## Train
    python train.py --epochs 5     # quick test run
    python train.py                # full run (up to 30 epochs, early stopping)

The first run downloads CIFAR-10 (about 170 MB). Training on a CPU takes a while; a few minutes
per epoch is normal, and a GPU or Google Colab is much faster.

Outputs: `cifar10_cnn.keras` (trained model), `training_curves.png`, `confusion_matrix.png`,
and a per-class precision/recall report in the terminal.

## Predict on your own images
    python predict.py my_photo.jpg

## How it works
- Input: 32x32 RGB images; pixel scaling is built into the model
- Augmentation (random flip, shift, small rotation) reduces overfitting; active only in training
- 3 blocks of [Conv-BatchNorm-ReLU] x2 + MaxPool + Dropout with 32, 64, 128 filters
- Global average pooling, a 128-unit dense layer, and a 10-way softmax
- Adam optimiser, early stopping on validation accuracy, learning-rate reduction on plateau
- 45,000 images train, 5,000 validate, 10,000 test (the official CIFAR-10 test split)

## Ideas to improve it
Train longer, try a deeper network or transfer learning (MobileNetV2/ResNet with upscaled images),
tune dropout and learning rate, or swap in your own image dataset with `image_dataset_from_directory`.

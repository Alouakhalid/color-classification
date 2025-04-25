# 🎨 Color Classification with CNN (TensorFlow)

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify images based on their dominant color. The model was trained on a custom dataset containing 10 color classes: `black`, `blue`, `brown`, `green`, `grey`, `orange`, `red`, `violet`, `white`, and `yellow`.

## 📂 Dataset

Images are organized in subfolders under `/content/training_dataset/`, one for each color class. Each image is resized to **128x128** and normalized before feeding it to the model.

## 🧠 Model Architecture

- **Conv2D** (32 filters, 3x3, ReLU)
- **MaxPooling2D**
- **Conv2D** (64 filters, 3x3, ReLU)
- **MaxPooling2D**
- **Conv2D** (128 filters, 3x3, ReLU)
- **MaxPooling2D**
- **Flatten**
- **Dense** (128 units, ReLU + L2 Regularization + Dropout)
- **Dense** (10 units, Softmax)

### 🔧 Compilation & Training

- Optimizer: `Adam`
- Loss: `Sparse Categorical Crossentropy`
- Epochs: `10`
- Validation Split: `20%`
- Callback: `CSVLogger` for monitoring training progress

## 📈 Results

Training and validation loss and accuracy are plotted for visualization. The model achieved competitive accuracy on the test set.

## 🔍 Prediction Function

A utility function `predict_color(image_path)` allows prediction of the dominant color in any new image.

```python
predicted_color = predict_color('/path/to/image.png')
print(predicted_color)

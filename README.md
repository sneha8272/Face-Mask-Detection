# Face Mask Detection using Deep Learning

A deep learning project for real-time face mask detection utilizing the MobileNetV2 architecture with transfer learning. The model is trained to classify images into two categories: 'with mask' or 'without mask'. This project covers the entire machine learning pipeline—from data acquisition and preprocessing to model training, fine-tuning, evaluation, and deployment readiness.

## 🚀 Features
* **Efficient Data Loading:** Uses `ImageDataGenerator` for on-the-fly image augmentation and batching.
* **Transfer Learning:** Leverages pre-trained MobileNetV2 weights to accelerate training and improve overall accuracy.
* **Fine-tuning Phase:** Includes an advanced fine-tuning phase by unfreezing base model layers with a lower learning rate.
* **Comprehensive Evaluation:** Visualizes training history (accuracy and loss plots) and generates a confusion matrix.
* **User-friendly Prediction:** Allows uploading custom images for instant mask detection with confidence scores.
* **Model Persistence:** Provides clean functionality to save and load the trained model for future inference.

---

## 📂 Dataset
The dataset used for training is the **`omkargurav/face-mask-dataset`** from Kaggle. It contains images categorized into two classes: `with_mask` and `without_mask`. The project automatically handles downloading this dataset using `kagglehub`.

---

## 🛠️ Tech Stack & Dependencies
Ensure you have the necessary libraries installed. If you are running the project locally, install the dependencies using pip:

```bash
pip install tensorflow matplotlib scikit-learn kagglehub
```

---

## 💻 Getting Started & Execution

1. Clone this repository or open the notebook file (`.ipynb`) in Google Colab.
2. Run all cells sequentially. The execution pipeline will perform the following steps:
   * **Import Libraries:** Loads TensorFlow, OpenCV, and evaluation modules.
   * **Download Dataset:** Automatically fetches the dataset via `kagglehub`.
   * **Preprocess Images:** Resizes images and sets up the training data generators.
   * **Build Model:** Compiles the MobileNetV2-based transfer learning network.
   * **Train Model:** Executes two training phases (initial training + fine-tuning).
   * **Evaluate:** Plots the performance metrics and displays the confusion matrix.
   * **Predict:** Prompts you to upload a custom image for live testing.
   * **Save:** Exports the finalized weights to your storage.

---

## ⚙️ Model Architecture
The custom neural network architecture consists of:
* A pre-trained **MobileNetV2 base model** (initialized with `imagenet` weights).
* A **GlobalAveragePooling2D** layer to reduce spatial dimensionality.
* A **Dense layer** with ReLU activation for feature extraction.
* A **Dropout layer** for regularization and preventing overfitting.
* A final **Dense layer** with Sigmoid activation for binary classification.

### Training Details:
* **Initial Training:** The MobileNetV2 base model is frozen; only the newly added top layers are trained for **8 epochs**.
* **Fine-tuning:** The last **30 layers** of MobileNetV2 are unfrozen. The entire model is trained with a lower learning rate (`0.0001`) for an additional **8 epochs**.
* **Optimizer:** Adam Optimizer.
* **Loss Function:** Binary Cross-Entropy.

---

## 📊 Evaluation
The model's performance is evaluated and visualized through two main methods:
1. **Training & Validation Plots:** Visual representations tracking loss and accuracy across all epochs.
2. **Confusion Matrix:** Provides a detailed numerical breakdown of true positives, true negatives, false positives, and false negatives on the validation subset.

---

## 💾 Saving and Loading the Model
The trained model is saved in Keras's native format (`.keras`) and can be easily reloaded for future production use without needing retraining.

```python
from tensorflow.keras.models import load_model

# Save the model
model.save('mask_detector_model.keras')

# Load the model back for inference
loaded_model = load_model('mask_detector_model.keras')
```
## 🖼️ Prediction Preview
![Prediction Example](prediction_screenshot.png)


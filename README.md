# Automated Fruit Classification System

## Project Overview

This project focuses on developing a **Computer Vision system** to assist local farmers in automating the sorting of produce. Manual classification is time-consuming and prone to human error. By leveraging Deep Learning, specifically **Convolutional Neural Networks (CNNs)**, this system identifies different fruit varieties with high precision, enabling efficient pallet stacking and inventory management.

The model was trained on a pre-processed dataset containing **14 distinct fruit categories** and achieved a final testing accuracy of **99.85%**.

## Key Features

* **Custom CNN Architecture:** A 3-block convolutional network designed for optimal feature extraction (edges → shapes → textures).
* **Data Augmentation:** Real-time generation of image variations (rotation, zoom, shifts) to prevent overfitting and improve generalization.
* **Hyperparameter Tuning:** Optimized learning rates and dropout layers to ensure stable convergence.
* **Early Stopping:** Automated training monitoring to prevent wasted computational resources.
* **Robust Evaluation:** Validated using an 80/20 train/validation split and tested on a completely unseen test dataset.

## Dataset

The dataset is a curated subset of the Kaggle Fruit Recognition data.
* **Input Shape:** 112x112x3 (RGB Images)
* **Classes (14):** Apple, Apricot, Avocado, Banana, Blueberry, Cactus Fruit, Cherry, Corn, Kiwi, Mango, Orange, Pineapple, Strawberry, Watermelon.
* **Structure:**
    * `Train`: Used for training and validation.
    * `Test`: Used exclusively for the final performance evaluation.

## Model Architecture

The solution implements a Sequential CNN with the following structure:

1.  **Convolutional Blocks (x3):**
    * Filters: 32 → 64 → 128
    * Kernel Size: 3x3
    * Activation: ReLU
    * Pooling: MaxPooling2D
2.  **Global Average Pooling:** efficient dimensionality reduction.
3.  **Dense Layers:**
    * 128 Neurons (ReLU)
    * Dropout (0.5) for regularization.
    * Output Layer (Softmax) for 14-class probability.

## Performance & Results

The model was evaluated on unseen test data with the following results:

* **Final Test Accuracy:** `99.85%`
* **Loss:** `0.0057`

### Learning Curves
*The model demonstrates stable learning with validation metrics closely tracking training metrics, indicating no significant overfitting.*

!Machine-Learning-Applications/learning curves.png

### Confusion Matrix
*The system shows distinct separation between classes, with negligible confusion even among visually similar fruits.*

!Machine-Learning-Applications/confusion matrix.png

## Installation & Usage

### Prerequisites
* Python 3.x
* TensorFlow / Keras
* Pandas, NumPy, Matplotlib, Seaborn, OpenCV

### Running the Notebook
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/fruit-classification-system.git](https://github.com/yourusername/fruit-classification-system.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib opencv-python seaborn scikit-learn
    ```
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Fruit_Classification_System.ipynb
    ```

## Future Improvements

* **Quality Control:** Expand the dataset to include damaged/rotten fruits to detect quality issues.
* **Mobile Deployment:** Convert the model to TensorFlow Lite for use in a mobile app for farmers.
* **Real-time Processing:** Integrate with a camera feed for conveyor belt sorting.

## Credits

* **Course:** Machine Learning Applications (TECH3300)
* **Data Source:** [Kaggle Fruit Recognition Dataset](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition/data)
* **Libraries:** TensorFlow, Keras, Matplotlib, Pandas.

---
*Created by Jeisson Nino - 2024*

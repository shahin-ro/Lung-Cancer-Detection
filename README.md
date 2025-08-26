# Lung Cancer Detection🔬

A deep learning system for classifying lung cancer from CT scans using
CNN and Xception transfer learning. 📊 Clones dataset from GitHub,
preprocesses images, trains a 4-class model (normal, adenocarcinoma,
large cell, squamous cell), and predicts with visualizations. 🚀 Ideal
for medical image analysis!

## 📋 Overview

This project implements a lung cancer prediction system using
Convolutional Neural Networks (CNN) with transfer learning based on the
Xception model pre-trained on ImageNet. It classifies chest CT scan
images into four categories:

-   Normal
-   Adenocarcinoma
-   Large Cell Carcinoma
-   Squamous Cell Carcinoma

The codebase is a Jupyter notebook designed for Google Colab, leveraging
GPU acceleration for efficient training. It includes data preprocessing,
model training, evaluation, and prediction with random test image
selection or user-uploaded images.

## 🛠️ Features

-   **Dataset Handling**: Clones dataset from your GitHub repository and
    ensures correct folder structure for 4 classes.
-   **Data Augmentation**: Applies transformations (flips, rotations,
    zooms) to enhance training data.
-   **Transfer Learning**: Uses Xception model with frozen pre-trained
    layers for robust feature extraction.
-   **Model Training**: Includes callbacks for early stopping, learning
    rate reduction, and model checkpointing.
-   **Evaluation & Visualization**: Plots accuracy/loss curves and
    supports predictions with true/predicted label display.
-   **Flexible Prediction**: Allows random test image selection or
    user-uploaded images for inference.

## 📂 Dataset

The dataset is sourced from your GitHub repository: YOUR_REPOSITORY_URL.
It is organized into three directories:

-   `dataset/train/`: Training images
-   `dataset/valid/`: Validation images
-   `dataset/test/`: Test images

Each directory contains subfolders for the four classes:

-   `normal`
-   `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`
-   `large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa`
-   `squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa`

**Note**: If the dataset is incomplete (e.g., empty folders), you may
need to populate it with images matching the class names. Ensure images
are in PNG/JPG format.

## 🚀 Getting Started

### Prerequisites

-   Google Colab with GPU runtime
-   Google Drive account (for saving models)
-   Python 3.x with required libraries (see below)

### Installation

1.  Open the Jupyter notebook `Lung_Tumor_Detection.ipynb` in Google
    Colab.

2.  Install dependencies by running the first cell:

    ``` bash
    !pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
    ```

### Setup

1.  **Clone the Repository**: Run the cell to clone the dataset and code
    from your repository:

    ``` bash
    !git clone https://github.com/shahin-ro/Lung-Cancer-Detection.git
    %cd https://github.com/shahin-ro/Lung-Cancer-Detection.git
    ```

2.  **Mount Google Drive**: Mount your Google Drive to save the trained
    model:

    ``` python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3.  **Verify Dataset**: Ensure the dataset in `./dataset/` has the
    correct structure. The code automatically removes unexpected folders
    and ensures the four class directories exist.

### Usage

1.  **Run the Notebook**: Execute the cells sequentially in Google
    Colab:

    -   **Cell 1**: Installs dependencies.
    -   **Cell 2**: Clones your repository.
    -   **Cell 3**: Mounts Google Drive.
    -   **Cell 4**: Verifies and cleans dataset structure.
    -   **Cell 5**: Loads and preprocesses data with
        `ImageDataGenerator`.
    -   **Cell 6**: Defines and compiles the Xception-based model.
    -   **Cell 7**: Trains the model with callbacks and saves it to
        Google Drive.
    -   **Cell 8**: Evaluates the model and plots accuracy/loss curves.
    -   **Cell 9**: Randomly selects a test image or allows uploading an
        image for prediction.

2.  **Prediction**:

    -   For random test image prediction, select option 1 in the
        prediction cell.
    -   For custom image prediction, select option 2 and upload a
        PNG/JPG CT scan image via the Colab file picker.

### Example Prediction

``` python
# Output for a random test image
Selected image: ./dataset/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/image1.png
True class: adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
Predicted class: adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib (Probability: 0.9273)
[Image displayed with true and predicted labels]
```

## 📈 Results

-   **Training Accuracy**: \~93% (as reported in the original
    repository, varies with dataset size).
-   **Validation Accuracy**: Monitored during training with early
    stopping to prevent overfitting.
-   **Test Accuracy**: Evaluated on the test set, displayed in Cell 8.
-   **Visualizations**: Training/validation accuracy and loss curves are
    plotted for performance analysis.

## 🛠️ Troubleshooting

-   **Empty Dataset Folders**: If `./dataset/test/` or other directories
    are empty, populate them with images matching the class names. Check
    folder contents with `!ls -l dataset/test/*`.
-   **Model Not Found**: Ensure the trained model is saved at
    `/content/drive/MyDrive/trained_lung_cancer_model.h5`. Re-run
    training if needed.
-   **Class Mismatch**: The code ensures only 4 classes are used. Verify
    folder names match `expected_classes`.
-   **Image Format**: Use PNG/JPG images for compatibility. Update the
    prediction code if other formats are used.
-   **Colab Memory Issues**: Reduce `BATCH_SIZE` (e.g., to 4) or
    `IMAGE_SIZE` (e.g., to 224x224) if out-of-memory errors occur.

## 🙏 Acknowledgements

-   Thanks to hallowshaw for the original project inspiration and
    dataset structure.
-   The Xception model is based on TensorFlow/Keras implementations
    pre-trained on ImageNet.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for
details.

## 📬 Contact

For issues or suggestions, open an issue on this repository or contact
the maintainer.

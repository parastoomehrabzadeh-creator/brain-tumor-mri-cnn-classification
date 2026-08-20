# Brain Tumor MRI Classification with a Custom CNN

This project presents a deep-learning pipeline for brain tumor MRI image classification using a custom convolutional neural network (CNN) built with TensorFlow and Keras.

The project was developed as part of a practical university lab in Biomedical Engineering. It focuses on RGB medical image classification, data preprocessing, data augmentation, CNN model training, early stopping, TensorBoard logging, and clinically relevant model evaluation.

## Project Overview

Medical image classification is an important application of artificial intelligence in healthcare. In this project, a CNN is trained to classify brain MRI images into four categories:

- glioma
- meningioma
- no tumor
- pituitary tumor

The workflow includes loading image datasets, cleaning invalid image files, applying augmentation to the training set, training a custom CNN, and evaluating the model using accuracy, precision, recall, F1-score, sensitivity, specificity, and confusion matrix.

## Why This Project Matters

In biomedical AI applications, classification performance must be evaluated carefully. Accuracy alone is not enough, especially in medical problems where false negatives can have serious clinical consequences.

This project demonstrates a practical deep-learning workflow for medical image classification and highlights the importance of sensitivity and specificity in biomedical model evaluation.

## Key Features

- Brain MRI image classification with TensorFlow and Keras
- Four-class classification: glioma, meningioma, no tumor, and pituitary tumor
- Automated dataframe creation from folder-based image datasets
- Invalid image detection and reporting
- Image resizing and pixel normalization
- Data augmentation for training images
- Custom CNN architecture
- Early stopping to reduce overfitting
- TensorBoard logging
- Model checkpointing
- Confusion matrix and classification report generation
- Sensitivity and specificity calculation for each class
- Clean GitHub-ready project structure

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- Convolutional Neural Networks
- Medical image classification

## Dataset Notice

The original MRI dataset is not included in this repository due to file size limitations and course data-sharing restrictions.

The project expects the dataset to be placed locally in the following structure:

    data/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/

The data folders are included only as placeholders using .gitkeep files.

## Reported Dataset Summary

The original notebook reported the following dataset sizes:

- Training samples: 5714
- Testing samples: 1312

Training class distribution:

- notumor: 1595 images
- pituitary: 1458 images
- meningioma: 1340 images
- glioma: 1321 images

The original notebook also reported that some invalid image files were detected and ignored during loading.

## Model Architecture

The implemented CNN follows this general structure:

    Conv2D
    MaxPooling2D
    Conv2D
    MaxPooling2D
    Conv2D
    MaxPooling2D
    Flatten
    Dense
    Dropout
    Dense Softmax Output

The model uses categorical cross-entropy loss and the Adamax optimizer.

## Machine-Learning Workflow

1. Load image paths and labels from folder structure
2. Verify image files and remove invalid entries
3. Split the training set into training and validation subsets
4. Apply data augmentation to training images
5. Normalize image pixels
6. Train a custom CNN model
7. Use early stopping and model checkpointing
8. Evaluate the model on the test set
9. Save metrics, confusion matrix, classification report, and training curves

## Reported Results from Original Notebook

The following results were reported in the original notebook output:

- Test accuracy: 0.9504
- Macro precision: 0.9514
- Macro recall: 0.9461
- Macro F1-score: 0.9475

Per-class sensitivity and specificity:

| Class | Sensitivity | Specificity |
|---|---:|---:|
| glioma | 0.8633 | 0.9941 |
| meningioma | 0.9379 | 0.9592 |
| notumor | 1.0000 | 0.9845 |
| pituitary | 0.9833 | 0.9960 |

Reported confusion matrix:

| True / Predicted | glioma | meningioma | notumor | pituitary |
|---|---:|---:|---:|---:|
| glioma | 259 | 38 | 2 | 1 |
| meningioma | 5 | 287 | 11 | 3 |
| notumor | 0 | 0 | 405 | 0 |
| pituitary | 1 | 3 | 1 | 295 |

Note: These results are reported from the original notebook output. They were not re-run in this repository because the original dataset is not included.

## Result Figures

Reported confusion matrix:

![Reported Confusion Matrix](results/confusion_matrix_reported.png)

Training metrics from the original notebook:

![Training Metrics](results/training_metrics_from_notebook.png)

Example image batch from the original notebook:

![Sample Batch](results/sample_batch_from_notebook.png)

## Project Structure

    brain-tumor-mri-cnn-classification/
    ├── data/
    │   ├── Training/
    │   │   └── .gitkeep
    │   └── Testing/
    │       └── .gitkeep
    ├── models/
    │   └── .gitkeep
    ├── logs/
    │   └── .gitkeep
    ├── results/
    │   ├── confusion_matrix_reported.csv
    │   ├── confusion_matrix_reported.png
    │   ├── reported_metrics_from_notebook.json
    │   ├── sample_batch_from_notebook.png
    │   └── training_metrics_from_notebook.png
    ├── src/
    │   └── brain_tumor_mri_cnn.py
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── LICENSE

## Installation

Clone the repository:

    git clone https://github.com/your-username/brain-tumor-mri-cnn-classification.git
    cd brain-tumor-mri-cnn-classification

Create a virtual environment:

    python -m venv .venv

Activate the virtual environment.

On Windows:

    .venv\Scripts\activate

On macOS/Linux:

    source .venv/bin/activate

Install the dependencies:

    pip install -r requirements.txt

## Usage

Place the MRI dataset locally inside the data folder using the expected folder structure.

Then run:

    python src/brain_tumor_mri_cnn.py

You can also specify custom paths and training settings:

    python src/brain_tumor_mri_cnn.py --train-dir data/Training --test-dir data/Testing --epochs 30 --batch-size 8 --image-size 128

Generated outputs will be saved in:

    results/
    models/
    logs/

## Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Sensitivity
- Specificity

For biomedical classification problems, sensitivity is especially important because it measures how well the model detects positive cases. Specificity is also important because it measures how well the model avoids false alarms.

## Important Note on AlexNet

The original lab task mentioned repeating the experiment with an AlexNet model. However, the uploaded notebook does not contain a completed AlexNet implementation.

For that reason, this GitHub version focuses on the completed custom CNN pipeline only. Adding AlexNet or transfer-learning models would be a useful future improvement.

## Future Improvements

Possible next steps include:

- Adding an AlexNet-inspired CNN implementation
- Comparing the custom CNN with pretrained transfer-learning models
- Adding Grad-CAM visualizations for explainability
- Adding cross-validation or repeated train/test experiments
- Adding model inference for single MRI images
- Improving class imbalance handling
- Adding a lightweight demo dataset if redistribution is allowed

## Repository Status

The code, structure, and reported outputs are prepared for GitHub portfolio presentation.

The original dataset is not included due to file size and course data-sharing restrictions.

## Author

Biomedical Engineering Master's student with interests in medical imaging, artificial intelligence, computer vision, machine learning, and medical device development.

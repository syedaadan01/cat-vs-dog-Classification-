# cat-vs-dog-Classification-


# Image Classification with Multiple Models

## Project Overview

This project involves training and evaluating multiple machine learning models for image classification. The primary goal is to assess the performance of various models on a dataset of images and visualize the results using confusion matrices, classification reports, and sample images with predictions.

## Objectives

1. **Train and Evaluate Models**: Implement and train several classification models including Logistic Regression, SVM, Random Forest, K-Nearest Neighbors, Naive Bayes, and Decision Tree on a given dataset.
2. **Performance Metrics**: Evaluate model performance using accuracy, confusion matrix, and classification report.
3. **Visualize Results**: Provide visual insights into model performance through plots of confusion matrices, classification reports, and sample images with predictions.

## Dataset

- **Images**: The dataset consists of 557 images categorized into two classes: 'cats' and 'dogs'.
- **Image Size**: The images are resized to 64x64 pixels and are converted into a flat vector of 12288 features (64x64x3).

## Models Used

1. **Logistic Regression**: A basic linear model used for binary classification.
2. **Support Vector Machine (SVM)**: A model that finds the optimal hyperplane for classification.
3. **Random Forest**: An ensemble method that uses multiple decision trees to improve classification accuracy.
4. **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies based on the majority class of nearest neighbors.
5. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
6. **Decision Tree**: A model that splits the data into branches to make classifications.

## Results

### Model Performance

- **Logistic Regression**:
  - Validation Accuracy: 0.55
  - Test Accuracy: 0.55

- **SVM**:
  - Validation Accuracy: 0.49
  - Test Accuracy: 0.56

- **Random Forest**:
  - Validation Accuracy: 0.57
  - Test Accuracy: 0.65

- **K-Nearest Neighbors**:
  - Validation Accuracy: 0.48
  - Test Accuracy: 0.63

- **Naive Bayes**:
  - Validation Accuracy: 0.48
  - Test Accuracy: 0.46

- **Decision Tree**:
  - Validation Accuracy: 0.58
  - Test Accuracy: 0.51

### Visualizations

- **Confusion Matrix**: Shows the number of true positive, true negative, false positive, and false negative predictions for each model.
- **Classification Report**: Includes precision, recall, and F1-score for each class.
- **Sample Images with Predictions**: Displays sample images with the true and predicted labels to visually assess model performance.

## Code Explanation

### Training Models

The code trains each model on the training set and evaluates it on the test set. Here is a summary of the key steps:

1. **Data Preparation**: Load and preprocess the dataset. Split the data into training, validation, and test sets.
2. **Model Training**: Train each model using the training data.
3. **Predictions**: Generate predictions for the test set using each trained model.
4. **Evaluation**: Compute and plot confusion matrices, classification reports, and sample images with predictions.

### Functions

- **`plot_confusion_matrix`**: Plots the confusion matrix for a given set of true and predicted labels.
- **`plot_classification_report`**: Displays the classification report as a heatmap.
- **`plot_sample_images`**: Shows sample images with their true and predicted labels.

## How to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. **Install Dependencies**:
   Make sure you have the required Python packages installed. You can use `requirements.txt` to install them:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code**:
   Execute the Python script or Jupyter Notebook to train the models and visualize the results.

   ```bash
   python your_script.py
   ```

   or open and run the Jupyter Notebook:
   ```bash
   jupyter notebook your_notebook.ipynb
   ```

## Future Work

- **Model Improvement**: Experiment with additional models and hyperparameter tuning.
- **Data Augmentation**: Apply data augmentation techniques to improve model performance.
- **User Interface**: Develop a web interface to allow users to interact with the model and upload their own images for classification.


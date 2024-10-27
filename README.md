# MachineLearningProject_MacqurieUniversity
This is a machine learning project to predict the age group of the person from the image. 

# Age Group Classification using Machine Learning and Deep Learning

## Project Overview

This project focuses on predicting the age group of individuals from images of human faces using both conventional machine learning and deep learning techniques. The dataset contains images of various individuals, with their corresponding age labels grouped into three categories:
- Ages 6-20
- Ages 35-40
- Ages 55-98

The primary objective is to develop high-performing classification models using:
1. Conventional machine learning approaches (including HOG features)
2. Deep learning models (potentially CNN-based)

### Datasets

The datasets for this project are provided in `.npy` format and are split into:
- `train_images.npy`
- `val_images.npy`
- `test_images.npy`
- A separate `test_private_images.npy` released later for final evaluation.

Labels are also provided in `.npy` format for training and validation:
- `train_labels.npy`
- `val_labels.npy`

Each label is formatted as `x-y`, where:
- `x` is an individual ID.
- `y` is the corresponding age of the person.

### Task

The main task is to predict the age group of individuals in the images. This is a classification problem where the age groups are divided into three classes:
- Class 0: Ages 6-20
- Class 1: Ages 35-40
- Class 2: Ages 55-98

### Project Goals
- **Model Development**: Implement and compare conventional machine learning models and deep learning models.
- **Feature Engineering**: Apply HOG (Histogram of Oriented Gradients) features to at least one conventional ML model.
- **Multiple Instances Consideration**: Incorporate multiple images of the same individual into at least one model.
- **Evaluation**: Perform model evaluation on a public test set, and final evaluation on a private test set to assess generalization.

## Project Structure

The repository contain the following components:

```plaintext
├── data/                     # Directory for datasets
├── notebooks/                # Jupyter Notebooks for model development and analysis
│   ├── ConventionalML.ipynb  # Notebook for conventional machine learning approaches
│   ├── DeepLearning.ipynb    # Notebook for deep learning models
├── results/                  # CSV files with predicted labels for public and private test sets
├── README.md                 # Project documentation (this file)

```

## Kaggle InClass Submissions

All results is submitted through the Kaggle InClass competition portal. The submission format is as follows:
- CSV file with two columns: `ID` and `Prediction`.
  - `ID`: Row number corresponding to the image.
  - `Prediction`: Predicted age group (0, 1, or 2).

The CSV filenames follow this format:
- Conventional ML submission: `46864512-conv-YYY.csv`
- Deep learning submission: `46864512-deep-YYY.csv`


## [Leaderboard with Private Dataset](https://github.com/ghatanisuresh/MachineLearningProject_MacqurieUniversity/blob/main/KaggleLeaderboard.md)

The private leaderboard reflects the final standings of the competition, calculated over the same rows as the public leaderboard.

**Summary**:

In this competition, my submission (under the name **Suresh Ghatani**) ranked 12th with a private score of **0.58003**. This was achieved using various approaches including deep learning models (CNN) and conventional machine learning techniques (SVM, Random Forest). The highest-performing model, a deep CNN, secured a score of **0.58003** on both the public and private test sets. Other methods, such as an SVM model with normalized features, also yielded competitive results with a score of **0.53706**.

## Evaluation Criteria

The project is evaluated based on:
- Accuracy of predicted labels on public and private test datasets.
- Quality of code and documentation.
- Thoroughness of the discussion on model performance, feature selection, and comparison between conventional ML and deep learning approaches.
  
## Future Work
- Experiment with additional feature extraction methods.
- Improve deep learning models using data augmentation and transfer learning techniques.
- Analyze the effect of noisy data on model performance.


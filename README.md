# CODSOFT
Internship




# Task-1 :


## Movie Description Classification

This GitHub repository contains code for a text classification task focused on movie descriptions. The goal of this project is to demonstrate two different approaches for classifying movie descriptions. You can choose either Part-1 or Part-2 based on your requirements.

## Note
In practical scenarios, you should choose either Part-1 or Part-2, not both. When you isolate one, you will not encounter any errors.

## Part-1: Naive Bayes Model
This part of the notebook focuses on using the Naive Bayes model for classifying movie descriptions.

- 1.0 - Setup And Import The Requirements


Ensure you have the required libraries installed and import them.

- 2.0 - Load Data And Process it


In this section, we cover data loading, cleaning, feature engineering, and TFIDF transformation.

- 2.1 - Load Data


Load the movie description dataset.

- 2.2 - Cleaning The Data


Clean and preprocess the data for further analysis.

- 2.3 - Feature Engineering


Perform any necessary feature engineering on the dataset.

- 2.4 - Do TFIDF Method


Transform the text data into TF-IDF vector representations.

- 3.0 - Load Model And Make Prediction


This section involves loading models and making predictions.

- 3.1 - Load Models And Choose The Best


Load the Naive Bayes models and choose the best-performing one.

- 3.2 - Make Prediction


Use the selected model to make predictions on movie descriptions.

## Part-2: FLAN-T5 Large Language Model


This part of the notebook focuses on using the FLAN-T5 large language model for movie description classification.

- ### 1 - Load Required Dependencies, Dataset, and LLM


Set up the environment and load the necessary dependencies, dataset, and the FLAN-T5 model.

- 1.1 - Test the Model with Zero Shot Inference


Demonstrate zero-shot inferencing using the FLAN-T5 model.

- 1.2 - Using One Shot and Few Shot Inference


Show how to use one-shot and few-shot inferencing techniques.

- 2 - Perform Parameter Efficient Fine-Tuning (PEFT)


In this section, we fine-tune the FLAN-T5 model for movie description classification.

- 2.1 - Preprocess the Classification Dataset


Prepare the dataset for fine-tuning.

- 2.2 - Setup the PEFT/LoRA model for Fine-Tuning


Configure the PEFT/LoRA model for fine-tuning.

- 2.3 - Train PEFT Adapter


Train the PEFT adapter on the classification task.

- 2.4 - Evaluate the Model Quantitatively (with ROUGE Metric)


Evaluate the fine-tuned model's performance using the ROUGE metric.

- 3 - Submission






# Task-2

# Transaction Fraud Detection

This GitHub repository contains a project focused on analyzing the Credit Card Transactions Fraud Detection Dataset, which includes both training and testing datasets. The primary objective of this project is to develop a fraud detection system for credit card transactions. We will conduct exploratory data analysis on the training dataset to identify potential correlations between features and fraudulent activities. Subsequently, we will build predictive models using these significant features and evaluate their effectiveness in detecting fraudulent transactions.

## Notebook Content

The project is organized into several sections within a Jupyter Notebook. Here's a brief overview of each section:

### 1.0 Load Requirements and Imports
In this section, we import the necessary libraries and dependencies required to run the project. This includes Python libraries for data manipulation, visualization, and machine learning.

### 2.0 Load Data and Analyze It
In this section, we load the Credit Card Transactions Fraud Detection Dataset and perform an initial analysis. This analysis includes exploring the dataset's structure, summary statistics, and data visualization to gain insights into the data.

### 3.0 Data Modeling and Prediction
This section is dedicated to building and evaluating predictive models for fraud detection.

#### 3.1 Preprocessing Train and Test Data
Before building models, we preprocess the training and testing data. This includes data cleaning, feature selection, and splitting the data into training and testing sets.

#### 3.2 SMOTE Method
We employ the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance issues in the dataset. SMOTE helps create synthetic samples of the minority class to balance the dataset and improve model performance.

#### 3.3 Preparing the Model and Making Predictions
In this final subsection, we prepare a Logistic Regression Model for fraud detection. This involves selecting appropriate algorithms, training the models on the preprocessed data, and making predictions. We evaluate the model's performance using relevant metrics to assess its ability to detect fraudulent transactions.




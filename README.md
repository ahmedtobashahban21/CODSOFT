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



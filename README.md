# Sentiment Analysis with LSTM

This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) neural networks. The model is built using PyTorch and trained on a dataset of movie reviews to classify them as positive or negative.

## Project Overview

The sentiment analysis model consists of the following components:

1. **LSTM Model Definition**: The `LSTM_model` class defines the architecture of the LSTM neural network. It includes an embedding layer, LSTM layers, and a fully connected output layer. The model computes the forward pass and calculates the loss using binary cross-entropy with logits.

2. **Data Preprocessing**: The data preprocessing script reads training and testing data from CSV files using pandas. It splits the data into features (reviews) and labels (sentiments) and further divides them into training, validation, and testing sets. The text data is tokenized and padded to ensure uniform length sequences.

3. **Training the Model**: The training script initializes the LSTM model with hyperparameters and optimizer settings. It then trains the model for a fixed number of epochs, monitoring training and validation performance. Early stopping and learning rate adjustment are used to prevent overfitting.

4. **Hyperparameter Tuning**: Hyperparameter tuning is performed using k-fold cross-validation. Various combinations of learning rates, weight decays, and dropout probabilities are explored to find the optimal configuration. The best hyperparameters and validation accuracy are reported.

5.**Final Training**: The model is trained again using the best hyperparameters obtained from hyperparameter tuning, and a learning rate scheduler is employed to dynamically adjust the learning rate during training, enhancing model convergence and performance.

## Usage

To train the sentiment analysis model:

1. Run the data preprocessing script to prepare the training and testing data.
2. Execute the training script to train the LSTM model and tune hyperparameters using k-fold cross-validation.
3. After training, evaluate the model on the test data to measure its performance.

## Files Included

- `kfold_hpt.ipynb` & `final_training.ipynb`: Defines the LSTM model architecture.
- `kfold_hpt.ipynb` & `final_training.ipynb`: Preprocesses the training and testing data.
- `kfold_hpt.ipynb`: Trains the LSTM model and performs hyperparameter tuning.
- `final_training.ipynb`: Trains the model with best hyperparameters and hyperparameter tuning , also includes data visualization to understand how each model interprets and classifies sentiment within the reviews.This is done by evaluating the performance of LSTM on a held-out testset of Yelp reviews using metrics like accuracy, precision, recall, and F1 score.

## Results
**kfold_hpt.ipynb**
The best hyperparameters found during hyperparameter tuning are:

- Learning Rate: 0.001
- Weight Decay: 1e-06
- Dropout Probability: 0.3

With these hyperparameters, the model achieves a validation accuracy of 84.53%.

**final_training.ipynb**

The model is training again with the best hyperparameters and learning rate scheduler
Results 
Testing Accuracy: 92.05%
Testing Loss: 0.2619
Precision: 0.9148
Recall: 0.9172
F1 Score: 0.9160

This also has visualization techniques  to understand how each model interprets and classifies sentiment within the reviews.
The following is incorporated - Confusion matrix, Histogram , learning curve ,Precision Recall curve , ROC curve,Word frequency using Histogram,Word Cloud for Positive and Negative reviews.

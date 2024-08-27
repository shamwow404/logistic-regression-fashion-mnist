#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:40:07 2024

@author: Sharmelle-Leonie Connell u3195672
"""

## PRML Assignment 1 Part B logistic regression using FASHION MNIST
# Import dataset from csv files
import pandas as pd
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')
# As the data is already split for test and train, 



# Showing images and data
# Extracting target and features to be used in plotting (labels, image) and to train the model
y_train = train_data.iloc[:, 0] 
X_train = train_data.iloc[:, 1:] 

y_test = test_data.iloc[:, 0]  # 1st column label, target variable
X_test = test_data.iloc[:, 1:] 



# Reshape and display some sample images
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 2))
for idx in range(5):
    image = X_train.iloc[idx].values.reshape(28, 28)  # Reshape the flat image to 28x28
    label = y_train.iloc[idx]
    plt.subplot(1, 5, idx + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(f'Label: {label}', fontsize=10)
    plt.axis('off') 

plt.show()


## Show correlaltion matrix
##correlation_matrix = test_data.corr()

# Visualize the correlation matrix
##import seaborn as sns
##plt.figure(figsize=(12, 8))
##sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
##plt.title("Correlation Matrix of Features")
##plt.show()


## Building the logistic regregression model to classify the images

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# Normalize feature values
X_train = X_train / 255.0  
X_test =  X_test / 255.0  # Normalize pixel values to [0, 1]


# Initialize the logistic regression model
# increase regularization (lower values of c is stronger reg)
# change solver from lbfgs to saga to deal with large dataset size and not reaching convergence within 300 iterations
lr = LogisticRegression(solver='saga', max_iter=300, C=0.1) 

# Fit the model
lr.fit(X_train, y_train)

# Predict on unseen data
y_pred = lr.predict(X_test)

# Measure performance
score = lr.score(X_test, y_test)
print(f'Accuracy: {score:.4f}')

# Print detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
## Visualise

# Display correct predictions
images_and_prediction = list(zip(combined_data, lr.predict(X)))

plt.figure(figsize=(10, 2))
for idx in range(5):
    image = X.iloc[idx].values.reshape(28, 28)  # Reshape the flat image to 28x28
    label = lr.predict(X)[idx]  # Predict the label for each image
    plt.subplot(1, 5, idx + 1)
    plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.title(f'Prediction: {label}', fontsize=10)  # Use the 'label' variable for the prediction
    plt.axis('off') 

plt.show()


## Display Misclassified images With predicted labels
index = 0
misclassifiedIndexes = []
for label, predict in zip(y_test, y_pred):
    if label != predict:
        misclassifiedIndexes.append(index)
    index +=1
    
#missclassifiedIndexes[:5]
#y_pred
#np.array(y_test)[:5]


plt.figure(figsize=(20,3))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.axis("off")
    plt.imshow(np.array(X_test.iloc[badIndex, :]).reshape(28, 28), cmap=plt.cm.
gray, interpolation='nearest')
    plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex], np.
array(y_test)[badIndex]), fontsize = 20)

## Saving the model
import pickle
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(lr, file)
    
## Loading the model later
import pickle

with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Verify predictions are the same
y_pred_loaded = loaded_model.predict(X_test)

# Optionally, you can print the accuracy or classification report again
loaded_score = loaded_model.score(X_test, y_test)
print(f'Accuracy of loaded model: {loaded_score:.4f}')
print(f'Accuracy of original model: {score:.4f}')

# If the accuracy matches the original, the model was saved and loaded correctly
# If the accuracy matches the original, the model was saved and loaded correctly


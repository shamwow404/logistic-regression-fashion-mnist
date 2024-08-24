#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:40:07 2024

@author: Sharmelle-Leonie Connell u3195672
"""

## PRML Assignment 1 Part B logistic regression using FASHION MNIST

## Import the data from csv
import pandas as pd
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')
# As the data is already split for test and train, 
# Concatenate the two DataFrames (just to observe the data as a whole, 
#and for future convenience normalising data
combined_data = pd.concat([train_data, test_data], ignore_index=True)


### Explore the data
#combined_data.shape
#combined_data.head()
#train_data.shape
#test_data.shape
#labels.value_counts()

# Showing images and data
# Extracting target and features to be used in plotting (labels, image) and to train the model
y = combined_data.iloc[:, 0]  # 1st column label, target variable
X = combined_data.iloc[:, 1:]  # Rest are features

# Reshape and display some sample images
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 2))
for idx in range(5):
    image = X.iloc[idx].values.reshape(28, 28)  # Reshape the flat image to 28x28
    label = y.iloc[idx]
    plt.subplot(1, 5, idx + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(f'Label: {label}', fontsize=10)
    plt.axis('off') 

plt.show()


## Building the logistic regregression model to classify the images

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# Normalize feature values
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

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
#Display correct predictions
images_and_prediction = list(zip(combined_data, lr.predict(X)))

plt.figure(figsize=(10,2))
for idx in range(5):
    image = X.iloc[idx].values.reshape(28, 28) 
    prediction = lr.predict(X)[idx]
    plt.subplot(1,5,idx+1)
    plt.axis("off")
    plt.imshow(np.array(image), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % int(prediction))
plt.show

## Display Misclassified images With predicted labels
# Create the dataframe
index = 0
misclassifiedIndexes = []
for label, predict in zip(y_test, y_pred):
    if label != predict:
        misclassifiedIndexes.append(index)
    index +=1
    
#missclassifiedIndexes[:5]
#y_pred
#np.array(y_test)[:5]

# Plot
plt.figure(figsize=(20,3))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.axis("off")
    plt.imshow(np.array(X_test.iloc[badIndex, :]).reshape(28, 28), cmap=plt.cm.
gray, interpolation='nearest')
    plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex], np.
array(y_test)[badIndex]), fontsize = 20)


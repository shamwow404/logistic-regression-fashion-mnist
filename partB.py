#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:40:07 2024

@author: Sharmelle-Leonie Connell u3195672
"""

## PRML Assignment 1 Part B logistic regression using FASHION MNIST

import pandas as pd
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')
# As the data is already split for test and train, 
# Concatenate the two DataFrames (just to observe the data as a whole, 
#and for future convenience normalising data
combined_data = pd.concat([train_data, test_data], ignore_index=True)



# Explore the data
#combined_data.shape
#combined_data.head()
#train_data.shape
#test_data.shape
#labels.value_counts()

# Showing images and data
# Extract features and labels
labels = combined_data.iloc[:, 0]  # Assuming the first column is the label
features = combined_data.iloc[:, 1:]  # Rest are features

# Reshape and display some sample images
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 2))
for idx in range(5):
    image = features.iloc[idx].values.reshape(28, 28)  # Reshape the flat image to 28x28
    label = labels.iloc[idx]
    plt.subplot(1, 5, idx + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(f'Label: {label}', fontsize=10)
    plt.axis('off')  # Hide axis for better visualization

plt.show()


## Building the logistic regregression model to classify the images

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

#Extract features and labels
y = combined_data.iloc[:, 0]  # first column is the label
X = combined_data.iloc[:, 1:]  # Rest are features

# Normalize feature values
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Initialize the logistic regression model
lr = LogisticRegression(solver='lbfgs', max_iter=200)

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
        


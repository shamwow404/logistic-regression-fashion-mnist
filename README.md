# Fashion MNIST - Logistic Regression

This project demonstrates how to train a **Logistic Regression** classifier on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset using `scikit-learn`. It includes image preprocessing, model training, evaluation, visualization of predictions (including misclassified images), and saving/loading the trained model with `pickle`.

---

## 📂 Dataset Setup

To run this project, you must first download the required CSV datasets:

👉 [**Download from Google Drive**](http://drive.google.com/file/d/1iX9mlb2VXOIPV5-0BSe-C1pBwoqt-k7X/view?pli=1)

Place the downloaded files (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`) in the **root folder of this project** (same directory as the script).

---

## 🚀 Features

- Loads Fashion-MNIST CSV datasets
- Visualizes sample images and their pixel matrices
- Trains a **Logistic Regression** model with normalization and regularization
- Evaluates model using accuracy, classification report, and confusion matrix
- Displays correct and misclassified predictions
- Saves and reloads the trained model with `pickle`

---

## 🧪 Requirements

Install the following Python libraries if you don't have them:

```bash
pip install pandas scikit-learn matplotlib numpy

```
## 📝 Running the Code
Make sure the CSV files are in the same folder as the script, then run:

```bash
python3 fashion_logistic_regression.py
```
You’ll see:

Sample image displays

Pixel matrix printouts

Model training output

Accuracy score

Classification report and confusion matrix

Visualization of correct and misclassified predictions

---

## 💾 Model Saving
The trained model is saved as:
```
logistic_regression_model.pkl
```
The script automatically tests that loading it back gives identical predictions.

## 📁 Project Structure
```
├── fashion_logistic_regression.py
├── fashion-mnist_train.csv         <-- Download from Google Drive
├── fashion-mnist_test.csv          <-- Download from Google Drive
└── logistic_regression_model.pkl   <-- Saved after training
```
---

## 🛠️ Tweak the Model & View Results

- To adjust the model’s performance, change the **regularization strength** by modifying the `C` parameter:
  ```python
  lr = LogisticRegression(solver='saga', max_iter=300, C=0.1)  # Lower C = more regularization

  ```
  Try different values like C=1.0, C=0.01, or C=0.001 and re-run the script.
---
- To allow more training time, increase the max number of iterations:

  ```python
  max_iter=500  # or higher if you see convergence warnings
  ```
---
- To view more sample correct predictions, change this block:
  ```python
  for idx in range(5):
    ...
  ```
  Increase range(5) to something like range(10) to see more results.
---
- To see more misclassified images, modify:
  ```python
  for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    ...
  ```
  You can change 0:5 to 0:10 or more.
---
- After making changes, re-run:
  ```bash
  python3 fashion_logistic_regression.py
  ```
  and check the accuracy, classification report, and new visual outputs.

HAVE FUN :3

✍️ Author
Made with 🖤 by Sham

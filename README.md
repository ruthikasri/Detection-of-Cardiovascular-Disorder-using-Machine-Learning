# Heart Disease Detection Using Machine Learning

## Overview
This project focuses on predicting cardiovascular disorders using a machine learning model. It utilizes the **Heart Disease Dataset** from Kaggle and employs **Logistic Regression** for classification. The project includes **data preprocessing, visualization, training, evaluation, and user input-based prediction.**

---

## Block Diagram

Below is a high-level block diagram of the system workflow:

```
          +---------------------------+
          |  Input Health Parameters  |
          +------------+--------------+
                       |
                       v
          +---------------------------+
          |     Data Preprocessing     |
          +------------+--------------+
                       |
                       v
          +---------------------------+
          |      Model Training        |
          +------------+--------------+
                       |
                       v
          +---------------------------+
          |      Model Evaluation      |
          +------------+--------------+
                       |
                       v
          +---------------------------+
          |  Predict Heart Disease?    |
          +------------+--------------+
                       |
              Yes             No
               |               |
     +-----------------+  +----------------+
     | Disease Detected |  | No Disease     |
     +-----------------+  +----------------+
```

---

## Flowchart

The following flowchart explains the step-by-step process of the project:

```
   Start
     |
     v
  Load Dataset
     |
     v
  Preprocess Data
     |
     v
  Split Data (Train/Test)
     |
     v
  Train Logistic Regression Model
     |
     v
  Evaluate Model Performance
     |
     v
  Get User Input for Prediction
     |
     v
  Predict Heart Disease (Yes/No)
     |
     v
   End
```

---

## Dataset
The dataset contains multiple medical attributes such as **age, sex, chest pain type, blood pressure, cholesterol levels, and heart rate** to predict the presence of heart disease (target variable: `0 = No Disease, 1 = Disease`).

---

## Code Explanation

### 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub
```
- Imports necessary libraries for data processing, visualization, and machine learning.

### 2. Downloading and Loading the Dataset
```python
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
csv_file = path + "/heart.csv"
df = pd.read_csv(csv_file)
```
- Downloads and loads the dataset from Kaggle.

### 3. Exploratory Data Analysis (EDA)
```python
print(df.head())
df.info()
print(df.isnull().sum())
print(df.describe())
```
- Displays the dataset structure, checks for missing values, and provides a statistical summary.

### 4. Data Visualization
```python
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```
- Shows a correlation heatmap to understand feature relationships.

### 5. Data Preprocessing
```python
X = df.drop(columns=['target'])  # Features
Y = df['target']  # Target Variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- Splits data into **training and testing sets**.
- Applies **StandardScaler** for feature normalization.

### 6. Model Training
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```
- Initializes and trains a **Logistic Regression model**.

### 7. Model Evaluation
```python
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print("Test Accuracy:", test_accuracy)
```
- Predicts training and test data and computes accuracy scores.

### 8. Confusion Matrix & Classification Report
```python
conf_matrix = confusion_matrix(Y_test, test_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(Y_test, test_predictions))
```
- Displays confusion matrix and classification report for model performance.

### 9. User Input Prediction
```python
def predict_heart_disease():
    print("Enter the following health parameters:")
    user_input = []
    input_ranges = {
        "age": "(29-77)",
        "sex": "(0: Female, 1: Male)",
        "cp": "(0-3, Chest pain type)",
        "trestbps": "(94-200, Resting BP in mm Hg)",
        "chol": "(126-564, Serum cholesterol mg/dl)",
        "fbs": "(0: <120 mg/dl, 1: >120 mg/dl, Fasting BS)",
        "restecg": "(0-2, Resting ECG results)",
        "thalach": "(71-202, Max heart rate)",
        "exang": "(0: No, 1: Yes, Exercise-induced angina)",
        "oldpeak": "(0.0-6.2, ST depression)",
        "slope": "(0-2, Slope of peak ST segment)",
        "ca": "(0-4, Major vessels colored by fluoroscopy)",
        "thal": "(0-3, Thalassemia type)"
    }
    for col in X.columns:
        value = float(input(f"{col} {input_ranges.get(col, '')}: "))
        user_input.append(value)

    user_array = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)
    prediction = model.predict(user_scaled)

    if prediction[0] == 1:
        print("The model predicts that the patient has heart disease.")
    else:
        print("The model predicts that the patient does NOT have heart disease.")
```
- Accepts user input for health parameters.
- Uses the trained model to predict heart disease presence.

---

## Deployment
- The model can be deployed using **Flask** for a web-based interface.
- The frontend can be built using **React.js** for user-friendly input.
- Deployment options include **Heroku, AWS, or Docker**.

---

## How to Run the Project
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/heart-disease-detection.git
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script
   ```bash
   python heart_disease_detection.py
   ```

---

## Conclusion
This project provides an effective **ML-based heart disease detection system** using **Logistic Regression**. Future improvements may include advanced models like **Random Forest, SVM, or Neural Networks** for better accuracy.

---

**Author:** Your Name

**GitHub Repository:** [Link to Repo]


# Machine-learning-model-implementation

NAME:RAJESH.R

PROGRAM:

Import necessary libraries
import pandas as pd import numpy as np import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Step 1: Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'] df = pd.read_csv(url, names=columns)

Step 2: Split the data
X = df.drop('Outcome', axis=1) # Features y = df['Outcome'] # Target label

80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 3: Train the Logistic Regression model
model = LogisticRegression(max_iter=200) model.fit(X_train, y_train)

Step 4: Make predictions
y_pred = model.predict(X_test)

Step 5: Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred)) print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) print("\nClassification Report:\n", classification_report(y_test, y_pred))

Step 6: (Optional) Plot actual vs predicted
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.6) plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.6) plt.title("Actual vs Predicted") plt.xlabel("Sample Index") plt.ylabel("Diabetes (1=Yes, 0=No)") plt.legend() plt.show()

🔍Code Description:

✅ 1. Import Necessary Libraries

import pandas as pd import numpy as np import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pandas: Used to load and manage the dataset.

numpy: Used for numerical operations.

matplotlib.pyplot: Used to plot graphs for visualization.

sklearn.model_selection.train_test_split: Used to split the data into training and test sets.

sklearn.linear_model.LogisticRegression: ML model used for binary classification.

sklearn.metrics: Provides tools to evaluate model performance.

📥 2. Load the Dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'] df = pd.read_csv(url, names=columns)

Loads the Pima Indians Diabetes Dataset from a URL.

Assigns appropriate column names to the dataset using names=columns.

📊 3. Prepare the Features and Target

X = df.drop('Outcome', axis=1) # Features y = df['Outcome'] # Target label

X contains all input features (pregnancies, glucose level, etc.).

y is the target/output column: 1 means diabetic, 0 means non-diabetic.

🧪 4. Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Splits the dataset into:

80% training data

20% testing data

random_state=42 ensures reproducibility.

🤖 5. Train the Logistic Regression Model

model = LogisticRegression(max_iter=200) model.fit(X_train, y_train)

Creates and trains a logistic regression classifier.

max_iter=200 ensures the model has enough iterations to converge.

📈 6. Make Predictions

y_pred = model.predict(X_test)

Predicts diabetes status (0 or 1) on the unseen test data.

📊 7. Evaluate the Model

print("Accuracy Score:", accuracy_score(y_test, y_pred)) print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) print("\nClassification Report:\n", classification_report(y_test, y_pred))

accuracy_score: Shows the overall accuracy of predictions.

confusion_matrix: Displays true positives, false positives, etc.

classification_report: Gives precision, recall, and F1-score.

📉 8. Visualize Predictions

plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.6) plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.6) plt.title("Actual vs Predicted") plt.xlabel("Sample Index") plt.ylabel("Diabetes (1=Yes, 0=No)") plt.legend() plt.show()

Creates a scatter plot to visually compare actual vs. predicted outcomes.

✅ Summary:

This script builds a binary classification model that predicts whether a patient has diabetes based on health indicators. It uses logistic regression, evaluates its accuracy, and visualizes how well the model performs.
OUTPUT:
![Image](https://github.com/user-attachments/assets/da56518a-1392-41b9-af10-41380118e832)

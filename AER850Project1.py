
"""
AER850 - Project 1

@author: Tanusha Lingam

Student Number: 501130352

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import StackingClassifier
from joblib import dump, load


# Step 1 Data Processing

# Load the dataset
data = pd.read_csv("Project 1 Data.csv")

# Displays first few coloumns
print(data.head())


# Step 2 [ Visualizing the Data ]

# 1. Histograms for each feature
data[['X','Y','Z']].hist()
plt.suptitle("Feature Distributions")
plt.show()

# 2. 3D Scatter plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d") # creates 3D canvas
sc = ax.scatter(data['X'], data['Y'], data['Z'], 
                c=data['Step'], cmap='tab20', s=50) # draws coordinates with color
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Scatter of Coordinates by Step")
plt.legend(*sc.legend_elements(), title="Step")
plt.show()

# Step 3 Correlation Analysis

# Compute correlation matrix for numeric features
corr_matrix = data[['X', 'Y', 'Z', 'Step']].corr(method='pearson')

print("Correlation among X, Y, Z:")
print(corr_matrix)

# Heatmap for visualization
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# Step 4 Classification Model Development

# Split data 
X = data[['X', 'Y', 'Z']].values
y = data['Step'].values

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Use Random Forest + GridSearchCV 
rf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [50, 100, 150]
}

grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Logistic Regression
logit_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
logit_param_grid = {'clf__C': [0.1, 1, 10]}

grid_logit = GridSearchCV(logit_pipe, logit_param_grid, cv=5, scoring='accuracy')
grid_logit.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt_param_grid = {'max_depth': [3, 5, 10]}

grid_dt = GridSearchCV(dt, dt_param_grid, cv=5, scoring='accuracy')
grid_dt.fit(X_train, y_train)

# KNN + RandomizedSearchCV
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])
knn_distros = {'clf__n_neighbors': [3, 5, 7, 9]}
rand_knn = RandomizedSearchCV(knn_pipe, knn_distros, n_iter=3, cv=5, scoring='accuracy', random_state=42)
rand_knn.fit(X_train, y_train)


# Retrieve the Best Models
best_rf = grid.best_estimator_
best_logit = grid_logit.best_estimator_
best_dt = grid_dt.best_estimator_
best_knn = rand_knn.best_estimator_

print("\nBest params:")
print("RF:", grid.best_params_)
print("Logit:", grid_logit.best_params_)
print("DT:", grid_dt.best_params_)
print("KNN (randomized):", rand_knn.best_params_)

# Step 5 Model Performance Analysis

# Store all best models for comparison
models = {
    "Random Forest": best_rf,
    "Logistic Regression": best_logit,
    "Decision Tree": best_dt,
    "KNN (Randomized)": best_knn
}

print("\n======= Model Performance Analysis =======")

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results.append((name, acc, prec, f1))
    print(f"\n{name}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

# pick best by F1
best_name, _, _, _ = max(results, key=lambda x: x[3])
best_model = models[best_name]
print(f"\nBest model by F1: {best_name}")

# confusion matrix for best model
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix — {best_name}")
plt.show()

# Step 6 Stacked Model Performance Analysis

# Combine two strong models from Step 5 (Random Forest + Logistic Regression)
estimators = [
    ('rf', best_rf),
    ('logit', best_logit)
]

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

# Train the stacked model
stack_clf.fit(X_train, y_train)

# Predict on test set
y_pred_stack = stack_clf.predict(X_test)

# Evaluate stacked model
acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test, y_pred_stack, average='weighted', zero_division=0)
f1_stack = f1_score(y_test, y_pred_stack, average='weighted', zero_division=0)

print("\n======= Stacked Model Performance =======")
print(f"Accuracy:  {acc_stack:.3f}")
print(f"Precision: {prec_stack:.3f}")
print(f"F1 Score:  {f1_stack:.3f}")

# Confusion matrix for stacked model
cm_stack = confusion_matrix(y_test, y_pred_stack)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_stack)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — Stacked Model (RF + Logistic Regression)")
plt.show()

# Step 7 Model Evaluation

dump(stack_clf, 'best_model.joblib')
loaded_model = load('best_model.joblib')

# Define new coordinates for prediction
new_coords = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# Predict the corresponding maintenance steps
predicted_steps = loaded_model.predict(new_coords)

print("\nPredicted Maintenance Steps for Given Coordinates:")
for coords, step in zip(new_coords, predicted_steps):
    print(f"Coordinates {coords} → Step {step}")


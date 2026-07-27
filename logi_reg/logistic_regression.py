# =============================================================================
#                      LOGISTIC REGRESSION CLASSIFICATION
# =============================================================================

# =============================================================================
# Step 1 : Import Libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Evaluation Metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

from matplotlib.colors import ListedColormap

# =============================================================================
# Step 2 : Load Dataset
# =============================================================================

dataset = pd.read_csv("Data/final1.csv")

# Independent Variables
X = dataset.iloc[:, [2, 3]].values

# Dependent Variable
y = dataset.iloc[:, -1].values

# =============================================================================
# Step 3 : Split Dataset
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# =============================================================================
# Step 4 : Feature Scaling
# =============================================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# Step 5 : Train Logistic Regression Model
# =============================================================================

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)

# =============================================================================
# Step 6 : Prediction
# =============================================================================

y_pred = classifier.predict(X_test)

# =============================================================================
# Step 7 : Model Evaluation
# =============================================================================

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n")

print("=" * 60)
print("Accuracy")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.4f}")

print("\n")

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(classification_report(y_test, y_pred))

print("\n")

print("=" * 60)
print("Bias & Variance")
print("=" * 60)

train_accuracy = classifier.score(X_train, y_train)
test_accuracy = classifier.score(X_test, y_test)

print(f"Training Accuracy (Bias) : {train_accuracy:.4f}")
print(f"Testing Accuracy (Variance): {test_accuracy:.4f}")

# =============================================================================
# Step 8 : ROC Curve & AUC Score
# =============================================================================

y_probability = classifier.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_probability)

print("\nAUC Score :", round(auc, 4))

fpr, tpr, threshold = roc_curve(y_test, y_probability)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr,
         color='blue',
         linewidth=2,
         label=f'ROC Curve (AUC = {auc:.3f})')

plt.plot([0,1],[0,1],
         linestyle='--',
         color='red')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.grid(True)

plt.show()

# =============================================================================
# Step 9 : Future Prediction
# =============================================================================

future_data = pd.read_csv(
    r"E:\ML\Machine learning\Logistic_Regression_Project\Data\Future prediction1.csv"
)

future_output = future_data.copy()

future_X = future_data.iloc[:, [2, 3]].values

# IMPORTANT:
# Use the SAME scaler used during training

future_X = scaler.transform(future_X)

future_output["Predicted_Class"] = classifier.predict(future_X)

future_output.to_csv(
    "Future_Prediction_Output.csv",
    index=False
)

print("\nFuture prediction saved successfully.")

# =============================================================================
# Step 10 : Visualize Training Set
# =============================================================================

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
    np.arange(X_set[:,0].min()-1,
              X_set[:,0].max()+1,
              0.01),

    np.arange(X_set[:,1].min()-1,
              X_set[:,1].max()+1,
              0.01)
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape),

    alpha=0.4,
    cmap=ListedColormap(("red","green"))
)

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==j,0],
        X_set[y_set==j,1],
        label=j
    )

plt.title("Logistic Regression (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# =============================================================================
# Step 11 : Visualize Test Set
# =============================================================================

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(X_set[:,0].min()-1,
              X_set[:,0].max()+1,
              0.01),

    np.arange(X_set[:,1].min()-1,
              X_set[:,1].max()+1,
              0.01)
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape),

    alpha=0.4,
    cmap=ListedColormap(("red","green"))
)

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==j,0],
        X_set[y_set==j,1],
        label=j
    )

plt.title("Logistic Regression (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()

plt.show()


from sklearn.navie_byes import BernoulliNB
classifier  =  BernoulliNB()
classifier.fit(X_train, y_train)  # 82.50 with scaling and bias =70 ,
   # without = 72.50, bias =  62
   # goision NB = 91.25 bias, 

from sklearn.navie_byes import GausianNB
classifier  =  GausianNB()
classifier.fit(X_train, y_train) #  with scaling  92.50 , bias = 87

# without scalingGausian NB  # without scling 92.50, bias = 87

# with scaling normalisation
from sklearn.navie_byes import NormalizationNB
classifier = NormalizationNB
classifier.fit(X_train, y_train)  # 

# without scaling  


from sklearn.navie_byes import GausianNB
classifier  =  GausianNB()
classifier.fit(X_train, y_train)

# Multinomial with scaling   score value = bias = 
# multinomial bayes doess not support -ve value then we change it into nrmalizaer,
# standerlaization not support

from sklearn.navie_byes import MultinomialNB
classifier  =  MultinomialNB()
classifier.fit(X_train, y_train)

# multinomial without scaling  , 56 very less, bias =  










# ==========================================
# AdaBoost Classification
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Lenovo\Downloads\Churn_Modelling.csv")

X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Label Encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])

# One Hot Encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    [('encoder',OneHotEncoder(),[1])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

# Train Test Split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# ====================================
# Training AdaBoost
# ====================================

from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=0
)

classifier.fit(X_train,y_train)

# Prediction

y_pred = classifier.predict(X_test)

# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)

# Accuracy

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test,y_pred)

print("Accuracy =",ac)

# Training Accuracy

bias = classifier.score(X_train,y_train)

print("Training Accuracy =",bias)

# Testing Accuracy

variance = classifier.score(X_test,y_test)

print("Testing Accuracy =",variance)

# ===================================
# Cross Validation
# ===================================

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

accuracies = cross_val_score(
    classifier,
    X_train,
    y_train,
    cv=skf,
    scoring="accuracy"
)

print(accuracies)

print("Mean Accuracy = {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation = {:.2f}%".format(accuracies.std()*100))

# ===================================
# Graph 1
# ===================================

plt.figure(figsize=(6,5))

plt.bar(
    ["Training","Testing"],
    [bias,variance]
)

plt.ylabel("Accuracy")
plt.title("Bias vs Variance (AdaBoost)")

for i,v in enumerate([bias,variance]):
    plt.text(i,v+0.01,f"{v:.3f}",ha='center')

plt.show()

# ===================================
# Graph 2
# ===================================

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    classifier,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.show()

# ===================================
# Graph 3
# ===================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    accuracies,
    marker='o',
    linewidth=2
)

plt.xlabel("Fold")

plt.ylabel("Accuracy")

plt.title("10 Fold Cross Validation (AdaBoost)")

plt.grid(True)

plt.show()

# ===================================
# Graph 4
# ===================================

plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:08:52 2026

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
dataset = pd.read_csv(
   r"E:\ML\Machine learning\ML_Classification_Algorithms\Naive_Bayes_Project\data\logistic_classification.csv")


print(dataset.head())

# Independent and Dependent variables
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# Feature Scaling (Optional for Decision Tree)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

criteria = ['gini','entropy']

estimators = [10,20,30,50,100,200]

depths = [2,3,4,5,6,8,10,None]

results = []

for criterion in criteria:

    for n in estimators:

        for depth in depths:

            classifier = RandomForestClassifier(
                criterion=criterion,
                n_estimators=n,
                max_depth=depth,
                random_state=0
            )

            # Train
            classifier.fit(X_train, y_train)

            # Prediction
            y_pred = classifier.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Train Accuracy (Bias)
            train_acc = classifier.score(X_train, y_train)

            # Test Accuracy (Variance)
            test_acc = classifier.score(X_test, y_test)

            # Gap
            gap = train_acc - test_acc

            # Save Result
            results.append([
                criterion,
                n,
                depth,
                accuracy,
                train_acc,
                test_acc,
                gap
            ])

            print("="*70)
            print(f"Criterion      : {criterion}")
            print(f"n_estimators   : {n}")
            print(f"max_depth      : {depth}")

            print(f"Accuracy       : {accuracy:.4f}")
            print(f"Train Accuracy : {train_acc:.4f}")
            print(f"Test Accuracy  : {test_acc:.4f}")
            print(f"Gap            : {gap:.4f}")

            if gap < 0.05:
                print("Model Status   : Good Fit")

            elif gap < 0.15:
                print("Model Status   : Slight Overfitting")

            else:
                print("Model Status   : High Overfitting")

            print("Confusion Matrix")
            print(confusion_matrix(y_test, y_pred))
            
            
            
            
 # Show the Best Combination
           
result_df = pd.DataFrame(results, columns=[
    "Criterion",
    "n_estimators",
    "max_depth",
    "Accuracy",
    "Train Accuracy",
    "Test Accuracy",
    "Gap"
])

print("\n================ ALL RESULTS ================\n")
print(result_df)

best = result_df.sort_values(
    by=["Test Accuracy", "Gap"],
    ascending=[False, True]
)

print("\n=========== BEST MODEL ===========\n")
print(best.head(1))




# Graph 1: Training Accuracy vs Testing Accuracy

plt.figure(figsize=(14,6))

x = range(len(result_df))

plt.plot(x,
         result_df["Train Accuracy"],
         marker='o',
         linewidth=2,
         label="Training Accuracy")

plt.plot(x,
         result_df["Test Accuracy"],
         marker='s',
         linewidth=2,
         label="Testing Accuracy")

plt.xlabel("Different Random Forest Models")
plt.ylabel("Accuracy")
plt.title("Random Forest: Training vs Testing Accuracy")
plt.legend()
plt.grid(True)

plt.show()

# Graph 2: Accuracy vs n_estimators
plt.figure(figsize=(8,6))

for criterion in ['gini', 'entropy']:

    temp = result_df[
        (result_df["Criterion"] == criterion) &
        (result_df["max_depth"] == 6)
    ].sort_values("n_estimators")

    plt.plot(
        temp["n_estimators"],
        temp["Test Accuracy"],
        marker='o',
        linewidth=2,
        label=criterion
    )

plt.xlabel("Number of Trees")
plt.ylabel("Testing Accuracy")
plt.title("Random Forest Accuracy vs Number of Trees")
plt.legend()
plt.grid(True)

plt.show()

# Graph 3: Accuracy vs max_depth

plt.figure(figsize=(8,6))

for criterion in ['gini','entropy']:

    temp = result_df[
        (result_df["Criterion"] == criterion) &
        (result_df["n_estimators"] == 100)
    ].copy()

    # Replace None/NaN
    temp["max_depth"] = temp["max_depth"].fillna(-1)

    temp = temp.sort_values("max_depth")

    labels = [
        "None" if x == -1 else str(int(x))
        for x in temp["max_depth"]
    ]

    plt.plot(
        labels,
        temp["Test Accuracy"],
        marker='o',
        linewidth=2,
        label=criterion
    )

plt.xlabel("Maximum Depth")
plt.ylabel("Testing Accuracy")
plt.title("Accuracy vs Max Depth")
plt.legend()
plt.grid(True)

plt.show()

# Graph 4: Heatmap graph

pivot = result_df.pivot_table(
    values='Test Accuracy',
    index='max_depth',
    columns='n_estimators'
)

fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(pivot.values, aspect='auto')

plt.colorbar(im)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)

ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

# Show accuracy values inside each cell
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(
            j,
            i,
            f"{pivot.iloc[i,j]:.2f}",
            ha="center",
            va="center",
            color="white"
        )

plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.title("Random Forest Accuracy Heatmap")

plt.show()





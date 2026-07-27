# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
dataset = pd.read_csv(
    r"E:\ML\Machine learning\ML_Classification_Algorithms\Naive_Bayes_Project\data\logistic_classification.csv"
)

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

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

# Compare both criteria
for criterion in ["gini", "entropy"]:

    print("\n" + "="*60)
    print("Decision Tree using :", criterion.upper())
    print("="*60)

    classifier = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=0
    )

    # Train
    classifier.fit(X_train, y_train)

    # Prediction
    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix")
    print(cm)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy :", round(accuracy,4))

    # Training Accuracy (Bias)
    train_acc = classifier.score(X_train, y_train)

    print("Training Accuracy (Bias) :", round(train_acc,4))

    # Testing Accuracy (Variance)
    test_acc = classifier.score(X_test, y_test)

    print("Testing Accuracy (Variance) :", round(test_acc,4))

    gap = train_acc - test_acc

    print("Difference :", round(gap,4))

    # Model Status
    if gap < 0.05:
        print("Model Status : Good Fit")
    elif gap < 0.15:
        print("Model Status : Slight Overfitting")
    else:
        print("Model Status : High Overfitting")


        
# 1. Plot the Decision Tree
#  This shows how the tree makes decisions.       
from sklearn.tree import plot_tree


plt.figure(figsize=(18,10))

plot_tree(
    classifier,
    feature_names=['Age','Estimated Salary'],
    class_names=['No','Yes'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree")
plt.show()



# 2. Decision Boundary Plot
# This visualizes how the Decision Tree separates the two classes.

from matplotlib.colors import ListedColormap


X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
    np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01),
    np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01)
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.35,
    cmap=ListedColormap(('red','green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==j,0],
        X_set[y_set==j,1],
        c=ListedColormap(('red','green'))(i),
        label=j
    )

plt.title("Decision Tree (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
    

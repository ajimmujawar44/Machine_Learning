

# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Lenovo\Downloads\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)
# One Hot Encoding the "Geography" column

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', 
                                      OneHotEncoder(), [1])], 
                                      remainder='passthrough')
X = np.array(ct.fit_transform(X)) 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier 
classifier = XGBClassifier() 
classifier.fit(X_train, y_train)  

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train,y_train)
bias


# logistic 
# svm
# dt
# rf 

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# stratifieed cross validation 

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(
    estimator=classifier,
    X=X_train,
    y=y_train,
    cv=10,
    scoring='accuracy'
)

print("Cross Validation Accuracy of each Fold")
print(accuracies)

print("Mean Accuracy = {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation = {:.2f}%".format(accuracies.std()*100))

# Stratified K-Fold Cross Validation

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

accuracies = cross_val_score(
    estimator=classifier,
    X=X_train,
    y=y_train,
    cv=skf,
    scoring='accuracy'
)

print("Accuracy of each Fold")
print(accuracies)

print("Mean Accuracy = {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation = {:.2f}%".format(accuracies.std()*100))

# Training Accuracy (Bias)
bias = classifier.score(X_train, y_train)
print("Training Accuracy =", bias)

# Testing Accuracy (Variance)
variance = classifier.score(X_test, y_test)
print("Testing Accuracy =", variance)



# Graph 1 : Bias and Variance

plt.figure(figsize=(6,5))

plt.bar(
    ["Training","Testing"],
    [bias, variance]
)

plt.ylabel("Accuracy")
plt.title("Bias vs Variance (XGBoost)")


for i,v in enumerate([bias,variance]):
    
    
    plt.text(i,v+0.01,f"{v:.3f}",ha='center')

plt.show()


# Graph 2 : Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    classifier,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.show()

# Graph 3 : Cross Validation Accuracy

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    accuracies,
    marker='o',
    linewidth=2
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("10-Fold Cross Validation")

plt.grid(True)

plt.show()

# Graph 4 : Prdicted vs Actual

plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.show()




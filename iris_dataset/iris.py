# libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mnso
# plot of decision tree.
from sklearn.tree import export_graphviz
# Convert to png.
from subprocess import call
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Model
from sklearn.ensemble import RandomForestClassifier
# Confusion Matrix and classification report.
from sklearn.metrics import confusion_matrix, classification_report

# data
iris = load_iris()

# Properties of the dataset
properties = dir(iris)
print("properties: ", properties)

# Column names
columns = iris.feature_names
#print(columns)

# Data
data = iris.data
#print(data[0])

# DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Shape
print("shape of the dataframe: ", df.shape)

# Target names
print("Target names", iris.target_names)

# Target
df["target"] = iris.target

# Dataframe
print(df.head())

# Data info
print(df.info())

# Missing values
print("Missing values")
print(df.isnull().sum())
print(df.isna().sum())
#mnso.matrix(df)
#print(plt.show())
#mnso.bar(df)
#print(plt.show())

# Data description
print("data description")
print(df.describe())

print("Modeliing part, using RandomForestClassifier")
# Training
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size= 0.8, test_size=0.2)


# Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Accuracy
accuracy = model.score(x_test, y_test)
print("Accuracy: ",accuracy)

# Prediction
y_predicted = model.predict(x_test)
print(" Prediction: ", y_predicted)

# Confusion Matrix
print("confusion_matrix")
cm = confusion_matrix(y_test, y_predicted)# fisrt the truth, second the predicted.
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("predicted")
plt.ylabel("Truth")
print(plt.show())


# Sensitivity an Specificity
print("Sensitivity an Specificity")
total1=sum(sum(cm))
sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

# Classification Report
print("Classification Report")
print(classification_report(y_test, y_predicted))


# Error Rate
print("Error Rate")
error_rate = 1 - accuracy
print(error_rate)

# Calculating errors
error = []
for i in range(1,50):
  rf = RandomForestClassifier(n_estimators=i)
  rf.fit(x_train, y_train)
  pred = rf.predict(x_test)
  error.append(np.mean(pred != y_test))

# Ploting error rate
plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate')
plt.xlabel('Tree')
plt.ylabel('Mean Error')
print(plt.show())
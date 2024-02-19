import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# read data set
df = pd.read_csv("Datasets/Iris.csv")

# Separating data into independent variables (X) and dependent variable (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3) 
knn_model.fit(X_train, y_train)

# Evaluate the model using the test set
y_pred = knn_model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
class_report = classification_report(y_test, y_pred)
print('Class report:\n', class_report)
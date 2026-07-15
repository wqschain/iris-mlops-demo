from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()

print(iris.keys())
print(iris.data[:5])
print(iris.target[:5])
print(iris.feature_names)
print(iris.target_names)

X_train , X_test, y_train, y_test = train_test_split( iris.data, iris.target)
print(len(X_train))
print(len(X_test))

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

joblib.dump(model, "model.pkl")
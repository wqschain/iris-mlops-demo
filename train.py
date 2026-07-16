from sklearn.datasets import load_iris #Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


iris = load_iris()

# im fixing the random state so every run of this code will result in the same split 
X_train , X_test, y_train, y_test = train_test_split( iris.data, iris.target, random_state=42)
print(len(X_train))
print(len(X_test))

# im also fixing the random state here too so the training of the model is the exact same each run 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy : {accuracy}")

# This helps me "save" the model into a file so main.py can run without retraining each time 
joblib.dump(model, "model.pkl")
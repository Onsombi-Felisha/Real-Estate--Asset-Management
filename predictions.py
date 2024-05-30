import pickle
import joblib
import sklearn

# Load logistic regression model
with open('/mnt/data/logisticregression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)
    print(f"Logistic Regression model loaded with sklearn version: {logistic_model.__module__}")

# Load decision tree classifier model
with open('/mnt/data/DecisionTreeClassifier_pickle', 'rb') as file:
    decision_tree_model = pickle.load(file)
    print(f"Decision Tree Classifier model loaded with sklearn version: {decision_tree_model.__module__}")

# Load random forest classifier model
with open('/mnt/data/RandomForestClassifier_pickle', 'rb') as file:
    random_forest_model = pickle.load(file)
    print(f"Random Forest Classifier model loaded with sklearn version: {random_forest_model.__module__}")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split data into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the model
dt = DecisionTreeClassifier(random_state=42)

# Fit the decision tree model on the training set
dt.fit(X_train, y_train)

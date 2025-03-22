from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Load the Iris dataset
data = load_iris()
features = data.data
labels = data.target

# Split data into training (60%), validation (20%), and test (20%) sets
# First Split: 60% for training, 40% for further splitting into validation and test
train_features, remaining_features, train_labels, remaining_labels = train_test_split(features, labels, test_size=0.4, random_state=42)

# Second Split: 50% of the remaining data for validation and 50% for testing
validation_features, test_features, validation_labels, test_labels = train_test_split(remaining_features, remaining_labels, test_size=0.5, random_state=42)

# Initialize the model
dt = DecisionTreeClassifier(random_state=42)

# Fit the decision tree model on the training set
dt.fit(train_features, train_labels)

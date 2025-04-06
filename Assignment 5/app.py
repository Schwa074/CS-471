import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Read Data
training_data = pd.read_csv("training.csv", header=None, usecols=[19,23], names=['Time','Current'])
test_data = pd.read_csv("test.csv", header=None, usecols=[0, 4], names=['Time','Current'])

# Print first 5 and last 5 rows of training data
print(training_data.columns)
print(training_data.head())  # First 5 rows
print(training_data.tail())  # Last 5 rows

# Print first 5 and last 5 rows of testing data
print(test_data.columns)
print(test_data.head())  # First 5 rows
print(test_data.tail())  # Last 5 rows

# Trim data based on time where fault occurs
training_data = training_data[training_data['Time'] <= 5.4]
test_data = test_data[test_data['Time'] <= 2.4]

# @@@ Training Data Plot @@@
df = training_data
fault_start = 5.1
fault_end = 5.4
# Separate the data points
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(40, 6))

# Plotting for column D
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation (D)', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (D)', alpha=0.5)


plt.title('Scatter Plot of Current Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

# @@@ Test Data plot @@@
df = test_data
fault_start = 2.1
fault_end = 2.4
# Separate the data points
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(40, 6))

# Plotting for column D
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation (D)', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (D)', alpha=0.5)


plt.title('Scatter Plot of Current Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

# Define segmenting and labeling function
def segment_labeling(data, window, overlap, time1, time2):

  # Define the number of data points per segment = window size

  #index determines the start of a window
  #in each step of segmenting loop
  index = 0

  #windolap incorporates overlaping percentage
  windolap = math.floor (window * overlap)

  # Create an empty DataFrame for storing the labels
  labels_df = pd.DataFrame(columns=['label'])

  time_series = []

  while (index + window) < len(data):
      # Extract a segment of data
      segment = data.iloc[index : (index+window)]

      # Labeling based on a given time (the oscillation time is given)
      if any((time1 <= t <= time2) for t in segment['Time']):
        label = 'oscillation'
      else:
        label = 'normal'

      time_series.append(segment['Current'])

      # Append the label to the labels DataFrame
      labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)

      #Shifting the index forward by stride = window - windolap
      index += window - windolap

  # return lables_df as a DataFrame
  return time_series, labels_df

window = 200
overlap = 0.75

train_X, train_y = segment_labeling(training_data, window, overlap, 5.1, 5.4)
test_X, test_y = segment_labeling(test_data, window, overlap, 2.1, 2.4)

train_y.value_counts()
test_y.value_counts()

X_train = np.array(train_X)
X_test = np.array(test_X)

print(X_train.shape)
print(X_test.shape)

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(train_y['label'])
y_test = le.transform(test_y['label'])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [10, 20, 30, 40],
    'max_depth': np.arange(1, 11)
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_rf.fit(X_train_scaled, y_train)

# Best model
best_rf = grid_rf.best_estimator_
best_score_rf = grid_rf.best_score_
print("Best Parameters:", grid_rf.best_params_)
print(f"Best recall from GridSearchCV: {best_score_rf * 100:.2f}%")

# Plot accuracy vs. number of trees for visualizations
mean_scores = []
for n in param_grid['n_estimators']:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train_scaled, y_train)
    score = clf.score(X_train_scaled, y_train)
    mean_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(param_grid['n_estimators'], mean_scores, marker='o')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Training Accuracy')
plt.title('Effect of n_estimators on Training Accuracy')
plt.grid(True)
plt.show()

# Evaluate on test set
y_pred = best_rf.predict(X_test_scaled)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
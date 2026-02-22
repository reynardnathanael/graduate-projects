import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

df_data = pd.read_csv("data/2025_energy_efficiency_data.csv")
df_data.head()

"""# DEALING WITH DATA"""

# Reference: https://medium.com/@ebimsv/ml-series-day-47-scaling-and-normalization-073e6a10fa7b
class MinMaxScaler:
  def __init__(self):
    self.min = None
    self.max = None

  def fit(self, X):
    self.min = np.min(X, axis=0)
    self.max = np.max(X, axis=0)

  def transform(self, X):
    return (X - self.min) / (self.max - self.min)

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)

# Split the data to training and testing
def split_data(X, Y, ratio):
  # Shuffle the dataset
  shuffle = np.random.permutation(X.shape[0])

  X_shuffled = X.iloc[shuffle].reset_index(drop=True)
  Y_shuffled = Y.iloc[shuffle].reset_index(drop=True)

  # Split percentage (75:25)
  split_index = int(ratio * len(X))

  # Split data into training and test set
  X_train = X_shuffled.iloc[:split_index]
  X_test = X_shuffled.iloc[split_index:]

  Y_train = Y_shuffled.iloc[:split_index]
  Y_test = Y_shuffled.iloc[split_index:]

  return X_train, X_test, Y_train, Y_test

# One hot encoding
# Reference: https://www.kaggle.com/code/ravaghi/neural-networks-from-scratch-using-only-numpy
def one_hot_encode(data, categorical_columns):
  # Drop categorical columns
  numerical_columns = data.drop(columns=categorical_columns).values

  list_encoded = []
  # One hot encoding process
  for column in categorical_columns:
    # Convert to int
    values = data[column].astype(int).values
    # Find unique values
    unique_values, inverse = np.unique(values, return_inverse=True)
    # One hot encode
    one_hot_encoding = np.eye(len(unique_values))[inverse]
    # Append to the list
    list_encoded.append(one_hot_encoding)

  # Return the combined numerical + one hot encoded categorical features
  return np.concatenate([numerical_columns] + list_encoded, axis=1)

def dealing_with_data(data):
  # Separate features and target
  X = data.drop(columns=["Heating Load"])
  Y = data["Heating Load"]

  # Shuffle & split
  X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.75)

  # One hot encoding
  X_train_encoded = one_hot_encode(X_train, ["Orientation", "Glazing Area Distribution"])
  X_test_encoded = one_hot_encode(X_test, ["Orientation", "Glazing Area Distribution"])

  # MinMax scaling
  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train_encoded)
  X_test_scaled = scaler.transform(X_test_encoded)

  return X_train_scaled, X_test_scaled, Y_train, Y_test

X_train, X_test, Y_train, Y_test = dealing_with_data(df_data)

print("X_train Shape:", X_train.shape)
print("Y_train Shape:", Y_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_test Shape:", Y_test.shape)

"""# MODEL ARCHITECTURE"""

# Activation Functions
# Reference: https://nthu-datalab.github.io/ml/labs/09_TensorFlow101/09_NN-from-Scratch.html
def relu(x, derivative=False):
  if derivative:
    return (x > 0).astype(float)
  else:
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
  # Clipping to prevent overflow
  # Reference: https://mangohost.net/blog/sigmoid-activation-function-in-python-explained/
  x = np.clip(x, -500, 500)
  if derivative:
    s = 1/(1 + np.exp(-x))
    return s * (1 - s)
  else:
    return 1/(1 + np.exp(-x))

# Initialize Weights (parameters)
# Reference: https://medium.com/%40keonyonglee/bread-and-butter-from-deep-learning-by-andrew-ng-course-1-neural-networks-and-deep-learning-41563b8fc5d8
def initialize_parameters(n_units):
  parameters = {}
  for i in range(1, len(n_units)):
    parameters["W" + str(i)] = np.random.randn(n_units[i-1], n_units[i]) * 0.01
    parameters["b" + str(i)] = np.zeros((1, n_units[i]))

  return parameters

# Forward Propagation
# Reference: https://www.geeksforgeeks.org/machine-learning/deep-neural-network-with-l-layers/
def forward_propagation(X, parameters, activation='relu', return_caches=True):
  caches = {}
  A = X
  L = len(parameters) // 2

  for i in range(1, L):
    A_prev = A
    Z = np.dot(A_prev, parameters["W" + str(i)]) + parameters["b" + str(i)]

    # Apply activation function
    if activation == 'relu':
      A = relu(Z)
    else:
      A = sigmoid(Z)

    caches["A" + str(i-1)] = A_prev
    caches["Z" + str(i)] = Z
    caches["A" + str(i)] = A

  # Output layer (linear activation for regression)
  A_prev = A
  ZL = np.dot(A_prev, parameters["W" + str(L)]) + parameters["b" + str(L)]
  AL = ZL

  caches["A" + str(L-1)] = A_prev
  caches["Z" + str(L)] = ZL
  caches["A" + str(L)] = AL

  # Return caches if needed
  if return_caches:
    return AL, caches
  else:
    return AL

# Back Propagation
# Reference: https://towardsdatascience.com/building-a-deep-neural-network-from-scratch-using-numpy-4f28a1df157a/
def backward_propagation(Y_pred, Y_true, parameters, caches, activation='relu'):
  L = len(parameters) // 2
  m = Y_true.shape[0]

  gradients = {}

  # Output layer gradient
  dZL = (Y_pred - Y_true) / m
  gradients["dW" + str(L)] = np.dot(caches["A" + str(L-1)].T, dZL)
  gradients["db" + str(L)] = np.sum(dZL, axis=0, keepdims=True)

  # Backpropagate through hidden layers
  dA = np.dot(dZL, parameters["W" + str(L)].T)

  for i in reversed(range(1, L)):
    # Compute dZ based on activation function
    if activation == "relu":
      dZ = dA * relu(caches["Z" + str(i)], derivative=True)
    else:
      dZ = dA * sigmoid(caches["Z" + str(i)], derivative=True)

    # Compute gradients
    gradients["dW" + str(i)] = np.dot(caches["A" + str(i-1)].T, dZ)
    gradients["db" + str(i)] = np.sum(dZ, axis=0, keepdims=True)

    # Propagate to previous layer
    if i > 1:
      dA = np.dot(dZ, parameters["W" + str(i)].T)

  return gradients

# Update Parameters
# Reference: https://towardsdatascience.com/building-a-deep-neural-network-from-scratch-using-numpy-4f28a1df157a/
def update_parameters(parameters, gradients, learning_rate):
  L = len(parameters) // 2

  for i in range(1, L + 1):
    parameters["W" + str(i)] -= learning_rate * gradients["dW" + str(i)]
    parameters["b" + str(i)] -= learning_rate * gradients["db" + str(i)]

  return parameters

# Calculate the Mean Squared Error (MSE)
def compute_mean_squared_error(Y_pred, Y_true):
  m = Y_true.shape[0]
  loss = np.sum((Y_pred - Y_true) ** 2) / (2 * m)
  return loss

# Calculate RMSE for predictions
# Reference: https://medium.com/@amit25173/calculating-rmse-using-numpy-step-by-step-guide-a6006c9e30d9
def compute_rmse(loss):
  rmse = np.sqrt(2 * loss)
  return rmse

# Training Mini-Batch SGD
# Reference 1: https://www.geeksforgeeks.org/machine-learning/ml-mini-batch-gradient-descent-with-python/
# Reference 2: https://medium.com/@enozeren/building-a-neural-network-from-scratch-with-python-905e20553b53
def train_model(X_train, Y_train, X_test, Y_test, n_units,
                learning_rate, epochs, batch_size, activation_function):
  # Reshape Y from 1D vector to 2D column vector
  if len(Y_train.shape) == 1:
    Y_train = Y_train.reshape(-1, 1)
  if len(Y_test.shape) == 1:
    Y_test = Y_test.reshape(-1, 1)

  # Initialize weights (parameters)
  parameters = initialize_parameters(n_units)

  list_train_loss = []
  list_test_loss = []
  list_train_rmse = []
  list_test_rmse = []
  n_minibatches = X_train.shape[0]

  # Iterating for each epochs
  for epoch in range(epochs):
    # Shuffle training data per epoch to prevent overfitting
    indices = np.random.permutation(n_minibatches)
    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]

    # Mini-batch training
    for i in range(0, n_minibatches, batch_size):
      # Data sampling
      X_batch = X_shuffled[i:i+batch_size]
      Y_batch = Y_shuffled[i:i+batch_size]

      # Forward propagation
      Y_pred, caches = forward_propagation(X_batch, parameters, activation_function)

      # Backward propagation
      gradients = backward_propagation(Y_pred, Y_batch, parameters, caches, activation_function)

      # Update parameters
      parameters = update_parameters(parameters, gradients, learning_rate)

    # Compute losses for monitoring
    Y_train_pred = forward_propagation(X_train, parameters, activation_function, return_caches=False)
    Y_test_pred = forward_propagation(X_test, parameters, activation_function, return_caches=False)

    # Calculate Mean Squared Error (MSE)
    train_loss = compute_mean_squared_error(Y_train_pred, Y_train)
    test_loss = compute_mean_squared_error(Y_test_pred, Y_test)

    # Calculate Root Mean Squared Error (RMSE)
    train_rmse = compute_rmse(train_loss)
    test_rmse = compute_rmse(test_loss)

    # Store MSE and RMSE to the list
    list_train_loss.append(train_loss)
    list_test_loss.append(test_loss)
    list_train_rmse.append(train_rmse)
    list_test_rmse.append(test_rmse)

    # Print progress every 500 epochs
    if (epoch + 1) % 500 == 0:
      print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f} "
            f"- Train RMSE: {train_rmse:.5f}, Test RMSE: {test_rmse:.5f}")

  return parameters, list_train_loss, list_test_loss, list_train_rmse, list_test_rmse

"""# EXPERIMENT 1: USE ALL THE FEATURES"""
"""# PLEASE COMMENT OUT THIS SECTION IF YOU WANT TO RUN THE EXPERIMENT 2"""
"""# OTHERWISE, THE RESULT WILL BE DIFFERENT!"""

# Check if Y_train is a Pandas object, if yes then convert to NumPy array
if hasattr(Y_train, "values"):
  Y_train_np = Y_train.values
else:
  Y_train_np = Y_train

# Check if Y_test is a Pandas object, if yes then convert to NumPy array
if hasattr(Y_test, "values"):
  Y_test_np = Y_test.values
else:
  Y_test_np = Y_test

# Set the hyperparameter
input_size = X_train.shape[1]
n_units = [input_size, 32, 16, 1]
learning_rate = 0.001
epochs = 20000
batch_size = 32
activation_function = "relu"

# Train the model
parameters, train_loss, test_loss, train_rmse, test_rmse = train_model(
  X_train, Y_train_np, X_test, Y_test_np,
  n_units, learning_rate, epochs,
  batch_size, activation_function
)

# Make predictions
Y_train_pred = forward_propagation(X_train, parameters, activation_function, return_caches=False)
Y_test_pred = forward_propagation(X_test, parameters, activation_function, return_caches=False)

print()
print("Final Results:")
print(f"Training RMSE: {train_rmse[-1]:.5f}")
print(f"Test RMSE: {test_rmse[-1]:.5f}")

"""# GRAPHS VISUALIZATION"""

# Plot training (learning) & prediction curve
# Reference: https://www.dataquest.io/blog/learning-curves-machine-learning
def plot_curve(train_loss, train_rmse, Y_train, Y_train_pred, Y_test, Y_test_pred):

  # Visualize the learning curve (MSE)
  plt.figure(figsize=(18, 6))
  plt.plot(train_loss, label = "Training Loss", color="blue")
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.title("Learning Curve over Epochs (MSE)")
  plt.legend()
  plt.grid(True)
  plt.show()

  # Visualize the learning curve (RMSE)
  plt.figure(figsize=(18, 6))
  plt.plot(train_rmse, label = "Training Loss", color="red")
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.title("Learning Curve over Epochs (RMSE)")
  plt.legend()
  plt.grid(True)
  plt.show()

  # Visualize the prediction curve for training data
  plt.figure(figsize=(18, 6))
  plt.plot(np.array(Y_train).flatten(), label="Label", color="blue")
  plt.plot(np.array(Y_train_pred).flatten(), label="Predict", color="#CC0000")
  plt.ylabel("Heating Load")
  plt.xlabel("#th Case")
  plt.title("Prediction for Training Data")
  plt.legend(loc="upper left")
  plt.grid(True)
  plt.show()

  # Visualize the prediction curve for test data
  plt.figure(figsize=(18, 6))
  plt.plot(np.array(Y_test).flatten(), label="Label", color="blue")
  plt.plot(np.array(Y_test_pred).flatten(), label="Predict", color="#CC0000")
  plt.ylabel("Heating Load")
  plt.xlabel("#th Case")
  plt.title("Prediction for Test Data")
  plt.legend(loc="upper left")
  plt.grid(True)
  plt.show()

# Comment out this plotting code if you want to run the experiment 2
plot_curve(train_loss, train_rmse, Y_train, Y_train_pred, Y_test, Y_test_pred)

"""# FEATURES SELECTION WITH PEARSON AND SPEARMAN"""

# Reference: https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-corr/
def pearson_spearman_correlation(X, Y, features):
  # Convert numpy arrays to dataframe
  df = pd.DataFrame(X, columns=features)
  if Y.ndim > 1:
    df['Target'] = Y.flatten()
  else:
    df['Target'] = Y

  # Find the correlation with pearson and spearman
  pearson_corr = df.corr(method='pearson')['Target'].drop('Target')
  spearman_corr = df.corr(method='spearman')['Target'].drop('Target')

  # Combine the two correlation for visualizing the heatmap later
  df_combined_correlations = pd.DataFrame({
    'Pearson': pearson_corr,
    'Spearman': spearman_corr
  }).sort_values('Pearson', ascending=False)

  # Convert to list, just take the top 10 features
  pearson_top10 = pearson_corr.sort_values(ascending=False, key=abs).index.tolist()[:10]
  spearman_top10 = spearman_corr.sort_values(ascending=False, key=abs).index.tolist()[:10]

  selected_features = []

  # If both pearson and spearman have the same values, then pick top 10 features from pearson
  if set(pearson_top10) == set(spearman_top10):
    for feature, value in pearson_corr.items():
      if feature in pearson_top10:
        selected_features.append(feature)
  # Else, average each values from both of them, then pick the top 10 features
  else:
    average = []
    for feature in features:
      pearson = pearson_corr[feature]
      spearman = spearman_corr[feature]
      average.append({
          'Feature': feature,
          'Value': (pearson + spearman) / 2
      })

    sorted_average = sorted(average, key=lambda x: abs(x['Value']), reverse=True)
    sorted_list = []

    for item in sorted_average:
      sorted_list.append(item['Feature'])

    for feature, value in pearson_corr.items():
      if feature in sorted_list[:10]:
        selected_features.append(feature)

  return selected_features, df_combined_correlations

"""# EXPERIMENT 2: ONLY USE SELECTED FEATURES"""
"""# COMMENT OUT THE EXPERIMENT 1 & THE EXPERIMENT 1's PLOTTING CODE AND EXECUTE THE CODE FROM BEGINNING"""
"""# THIS APPROACH WILL GIVE THE SAME RESULT AS IN THE REPORT, BECAUSE OF THE np.random.seed(42)"""
"""# WITHOUT COMMENTING THE EXPERIMENT 1, THE RESULT WILL BE DIFFERENT AND MUCH WORST"""

features = [
    'Cooling Load',
    'Relative Compactness',
    'Surface Area',
    'Wall Area',
    'Roof Area',
    'Overall Height',
    'Glazing Area',
    'Orientation_2', 'Orientation_3', 'Orientation_4', 'Orientation_5',
    'Glazing_Dist_0', 'Glazing_Dist_1', 'Glazing_Dist_2',
    'Glazing_Dist_3', 'Glazing_Dist_4', 'Glazing_Dist_5'
]

selected_features, combined = pearson_spearman_correlation(X_train, Y_train, features)

# Visualize the heatmap
plt.figure(figsize=(18,6))
sns.heatmap(combined.T, annot=True, cmap="coolwarm", center=0)
plt.title("Pearson and Spearman Correlations (Heatmap)")
plt.show()

selected_features_indices = []
for feature in selected_features:
  selected_features_indices.append(features.index(feature))

X_train_selected = X_train[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

# Check if Y_train is a Pandas object, if yes then convert to NumPy array
if hasattr(Y_train, "values"):
  Y_train_np = Y_train.values
else:
  Y_train_np = Y_train

# Check if Y_test is a Pandas object, if yes then convert to NumPy array
if hasattr(Y_test, "values"):
  Y_test_np = Y_test.values
else:
  Y_test_np = Y_test

# Set the hyperparameter
input_size = X_train_selected.shape[1]
n_units = [input_size, 32, 16, 1]
learning_rate = 0.001
epochs = 20000
batch_size = 32
activation_function = "relu"

# Train the model
parameters, train_loss, test_loss, train_rmse, test_rmse = train_model(
  X_train_selected, Y_train_np, X_test_selected, Y_test_np,
  n_units, learning_rate, epochs,
  batch_size, activation_function
)

# Make predictions
Y_train_pred = forward_propagation(X_train_selected, parameters, activation_function, return_caches=False)
Y_test_pred = forward_propagation(X_test_selected, parameters, activation_function, return_caches=False)

print()
print("Final Results:")
print(f"Training RMSE: {train_rmse[-1]:.5f}")
print(f"Test RMSE: {test_rmse[-1]:.5f}")

plot_curve(train_loss, train_rmse, Y_train, Y_train_pred, Y_test, Y_test_pred)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

df_data = pd.read_csv("data/2025_ionosphere_data.csv")
df_data.head()

"""# DEALING WITH DATA"""

# StandardScaler function from scratch
# Reference: https://www.kaggle.com/code/mohammadfawzy/features-scaling-from-scratch
class StandardScaler:
  def __init__(self):
    self.mean = None
    self.std = None

  def fit(self, X):
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    return self

  def transform(self, X):
    X_transformed = (X - self.mean) / (self.std + 1e-8)
    return X_transformed

  def fit_transform(self, X):
    self.fit(X)
    X_trans = self.transform(X)
    return X_trans

def split_data(X, Y, ratio):
  # Shuffle the dataset
  shuffle = np.random.permutation(X.shape[0])

  X_shuffled = X[shuffle]
  Y_shuffled = Y[shuffle]

  # Split percentage (80:20)
  split_index = int(ratio * len(X))

  # Split data into training and test set
  X_train = X_shuffled[:split_index]
  X_test = X_shuffled[split_index:]

  Y_train = Y_shuffled[:split_index]
  Y_test = Y_shuffled[split_index:]

  return X_train, X_test, Y_train, Y_test

def dealing_with_data(data):
  # Separate features and target + label encoding
  Y = data.iloc[:, -1].map({
      "g": 1,
      "b": 0
  }).values
  X = data.iloc[:, :-1].values.astype(float)

  # Shuffle & split
  X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.8)

  # Standard scaling
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

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

# Reference: https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
def tanh(x, derivative=False):
  if derivative:
    return 1 - np.tanh(x)**2
  else:
    return np.tanh(x)

# Initialize Weights (parameters)
# Reference: https://medium.com/@piyushkashyap045/mastering-weight-initialization-in-neural-networks-a-beginners-guide-6066403140e9
def initialize_parameters(n_units):
  parameters = {}
  for i in range(1, len(n_units)):
    scale = np.sqrt(2.0 / n_units[i-1])
    parameters["W" + str(i)] = np.random.randn(n_units[i], n_units[i-1]) * scale
    parameters["b" + str(i)] = np.zeros((n_units[i], 1))

  return parameters

# Forward Propagation
# Reference: https://www.geeksforgeeks.org/machine-learning/deep-neural-network-with-l-layers/
def forward_propagation(X, parameters, activation, return_caches=True, return_latent=False):
  caches = {}
  A = X
  L = len(parameters) // 2

  for i in range(1, L):
    A_prev = A
    Z = np.dot(parameters["W" + str(i)], A_prev) + parameters["b" + str(i)]

    # Apply activation function
    if activation[i - 1] == 'relu':
      A = relu(Z)
    elif activation[i - 1] == 'tanh':
      A = tanh(Z)
    else:
      A = sigmoid(Z)

    caches["A" + str(i-1)] = A_prev
    caches["Z" + str(i)] = Z
    caches["A" + str(i)] = A

  if return_latent:
    latent = A.copy()
    return latent
  else:
    # Output layer (linear activation for regression)
    A_prev = A
    ZL = np.dot(parameters["W" + str(L)], A_prev) + parameters["b" + str(L)]
    AL = sigmoid(ZL)

    caches["A" + str(L-1)] = A_prev
    caches["Z" + str(L)] = ZL
    caches["A" + str(L)] = AL

    # Return caches if needed
    if return_caches:
      return AL, caches
    else:
      return AL

# Calculate the Cross-Entropy
# Reference: https://last9.io/blog/understanding-log-loss-and-cross-entropy/
def compute_cross_entropy(Y_pred, Y_true):
  Y_pred = np.clip(Y_pred, 1e-7, 1 - 1e-7)
  loss = -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
  return loss

# Calculate Error Rate
# Reference 1: https://www.geeksforgeeks.org/machine-learning/fixing-accuracy-score-valueerror-cant-handle-mix-of-binary-and-continuous-target/
# Reference 2: https://stackoverflow.com/questions/54486014/cant-understand-this-line-of-code-np-meanpred-i-y-test
def compute_error_rate(Y_true, Y_pred):
  Y_pred_binary = (Y_pred >= 0.5).astype(int)
  error_rate = np.mean(Y_pred_binary != Y_true.flatten())
  return error_rate

# Reference: https://stackoverflow.com/questions/52865390/definition-of-error-rate-in-classification-and-why-some-researchers-use-error-ra
def compute_accuracy(error_rate):
  accuracy = (1 - error_rate) * 100
  return accuracy

# Back Propagation
# Reference: https://towardsdatascience.com/building-a-deep-neural-network-from-scratch-using-numpy-4f28a1df157a/
def backward_propagation(Y_pred, Y_true, parameters, caches, activation):
  L = len(parameters) // 2
  m = Y_true.shape[1]

  gradients = {}

  # Output layer gradient
  dZL = (Y_pred - Y_true) / m
  gradients["dW" + str(L)] = np.dot(dZL, caches["A" + str(L-1)].T)
  gradients["db" + str(L)] = np.sum(dZL, axis=1, keepdims=True)

  # Backpropagate through hidden layers
  dA = np.dot(parameters["W" + str(L)].T, dZL)

  for i in reversed(range(1, L)):
    # Compute dZ based on activation function
    if activation[i - 1] == "relu":
      dZ = dA * relu(caches["Z" + str(i)], derivative=True)
    elif activation[i - 1] == "tanh":
      dZ = dA * tanh(caches["Z" + str(i)], derivative=True)
    else:
      dZ = dA * sigmoid(caches["Z" + str(i)], derivative=True)

    # Compute gradients
    gradients["dW" + str(i)] = np.dot(dZ, caches["A" + str(i-1)].T)
    gradients["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True)

    # Propagate to previous layer
    if i > 1:
      dA = np.dot(parameters["W" + str(i)].T, dZ)

  return gradients

# Update Parameters
# Reference: https://towardsdatascience.com/building-a-deep-neural-network-from-scratch-using-numpy-4f28a1df157a/
def update_parameters(parameters, gradients, learning_rate):
  L = len(parameters) // 2

  for i in range(1, L + 1):
    parameters["W" + str(i)] -= learning_rate * gradients["dW" + str(i)]
    parameters["b" + str(i)] -= learning_rate * gradients["db" + str(i)]

  return parameters

# Training Mini-Batch SGD
# Reference 1: https://www.geeksforgeeks.org/machine-learning/ml-mini-batch-gradient-descent-with-python/
# Reference 2: https://medium.com/@enozeren/building-a-neural-network-from-scratch-with-python-905e20553b53
def train_model(X_train, Y_train, X_test, Y_test, n_units, learning_rate,
                epochs, batch_size, activation_function, visualize_epochs):
  # Transpose the data to (features, samples)
  X_train = X_train.T
  X_test = X_test.T

  # Reshape Y from 1D vector to 2D column vector
  Y_train = Y_train.reshape(1, -1)
  Y_test = Y_test.reshape(1, -1)

  # Initialize weights (parameters)
  parameters = initialize_parameters(n_units)

  # Lists for the results
  list_train_loss = []
  list_test_loss = []
  list_train_error = []
  list_test_error = []
  list_train_acc = []
  list_test_acc = []

  # Set for plotting
  data_visualization = {}

  # Minibatch
  n_minibatches = X_train.shape[1]

  # Iterating for each epochs
  for epoch in range(epochs):
    # Shuffle training data per epoch to prevent overfitting
    indices = np.random.permutation(n_minibatches)
    X_shuffled = X_train[:, indices]
    Y_shuffled = Y_train[:, indices]

    # Mini-batch training
    for i in range(0, n_minibatches, batch_size):
      # Data sampling
      X_batch = X_shuffled[:, i:i+batch_size]
      Y_batch = Y_shuffled[:, i:i+batch_size]

      # Forward propagation
      Y_pred, caches = forward_propagation(X_batch, parameters, activation_function)

      # Backward propagation
      gradients = backward_propagation(Y_pred, Y_batch, parameters, caches, activation_function)

      # Update parameters
      parameters = update_parameters(parameters, gradients, learning_rate)

    # This logic is for visualization
    if (epoch + 1) in visualize_epochs:
      visualize_parameters = {}
      for key, value in parameters.items():
        visualize_parameters[key] = value.copy()

      data_visualization[epoch + 1] = {
          "parameters": visualize_parameters,
      }

    # Compute losses for monitoring
    Y_train_pred = forward_propagation(X_train, parameters, activation_function, return_caches=False)
    Y_test_pred = forward_propagation(X_test, parameters, activation_function, return_caches=False)

    # Calculate the cross-entropy
    train_loss = compute_cross_entropy(Y_train_pred, Y_train)
    test_loss = compute_cross_entropy(Y_test_pred, Y_test)

    # Calculate the error rate
    train_error = compute_error_rate(Y_train, Y_train_pred)
    test_error = compute_error_rate(Y_test, Y_test_pred)

    # Calculate accuracy
    train_acc = compute_accuracy(train_error)
    test_acc = compute_accuracy(test_error)

    # Store cross-entropy, error rate, and accuracy to the list
    list_train_loss.append(train_loss)
    list_test_loss.append(test_loss)
    list_train_error.append(train_error)
    list_test_error.append(test_error)
    list_train_acc.append(train_acc)
    list_test_acc.append(test_acc)

    # Print progress every 500 epochs
    if (epoch + 1) % 500 == 0:
      print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f} - Train Error Rate: {train_error:.5f}, Test Error Rate: {test_error:.5f} - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

  return parameters, list_train_loss, list_test_loss, list_train_error, list_test_error, list_train_acc, list_test_acc, data_visualization

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
# n_units = [input_size, 64, 32, 1]
# n_units = [input_size, 16, 8, 1]
learning_rate = 0.01
epochs = 4000
batch_size = 64
activation_function = ["relu", "relu", "sigmoid"]
visualize_epochs = [10, 390, 500, 1000, 2000, 3000, 4000]

# Train the model
parameters, train_loss, test_loss, train_error, test_error, train_acc, test_acc, data_visualization = train_model(
  X_train, Y_train_np, X_test, Y_test_np,
  n_units, learning_rate, epochs,
  batch_size, activation_function, visualize_epochs
)

# Print the final result
print()
print("Final Results:")
print(f"Training Error Rate: {train_error[-1]:.5f}")
print(f"Test Error Rate: {test_error[-1]:.5f}")
print(f"Training Accuracy: {train_acc[-1]:.2f}%")
print(f"Test Accuracy: {test_acc[-1]:.2f}%")

"""# GRAPHS VISUALIZATION"""

# def plot_curve(train_loss, train_error):
#   # Visualize the learning curve (Cross Entropy)
#   plt.figure(figsize=(18, 6))
#   plt.plot(train_loss, label = "Training Loss", color="blue")
#   plt.ylabel("Loss")
#   plt.xlabel("Epoch")
#   plt.title("Learning Curve over Epochs (Cross Entropy)")
#   plt.legend()
#   plt.grid(True)
#   plt.show()

#   # Visualize the learning curve (Error Rate)
#   plt.figure(figsize=(18, 6))
#   plt.plot(train_error, label = "Training Loss", color="red")
#   plt.ylabel("Loss")
#   plt.xlabel("Epoch")
#   plt.title("Learning Curve over Epochs (Error Rate)")
#   plt.legend()
#   plt.grid(True)
#   plt.show()

# plot_curve(train_loss, train_error)

# PCA from scratch
# Reference: https://www.kaggle.com/code/fareselmenshawii/pca-from-scratch
def pca_fit_transform(X, n_components):
  # Mean-center the data
  mean = np.mean(X, axis=0)
  X = X - mean

  # Compute covariance matrix
  cov = np.cov(X.T)

  # Eigen decomposition
  eigenvalues, eigenvectors = np.linalg.eigh(cov)
  eigenvalues = eigenvalues[::-1]
  eigenvectors = eigenvectors[:,::-1]

  # Select top components
  components = eigenvectors[:,:n_components]

  # Project data
  transformed_data = np.dot(X, components)

  return transformed_data

# Plot the distribution of latent features
def plot_distribution_latent_features(X, Y, parameters, activations, epoch):
  # Create the latent by forward propagation without input & output layer
  latent = forward_propagation(X, parameters, activations, return_caches=False, return_latent=True)

  # PCA dimension reduction
  latent_pca = pca_fit_transform(latent.T, 2)

  # Plot the figure
  plt.figure(figsize=(10, 6))
  
  # Plot for both class 1 & class 2
  Y_flat = Y.flatten()
  plt.scatter(latent_pca[Y_flat == 0, 0], -latent_pca[Y_flat == 0, 1], color='blue', label='Class 1')
  plt.scatter(latent_pca[Y_flat == 1, 0], -latent_pca[Y_flat == 1, 1], color='red', label='Class 2')

  # Configuration
  plt.title("2D feature " + str(epoch) + " epoch")
  plt.grid(True)
  plt.legend()
  plt.show()

for epoch in visualize_epochs:
  plot_distribution_latent_features(X_train.T, Y_train.T, data_visualization[epoch]["parameters"], activation_function, epoch)
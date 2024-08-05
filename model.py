import numpy as np
# Sigmoid function


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Prediction function


def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)
# Loss function


def compute_loss(y, y_pred):
    m = len(y)
    loss = - (1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss
# Gradient Descent


def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        y_pred = predict(X, weights)
        gradients = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradients    
        if i % 1000 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Iteration {i}: Loss = {loss}")
    return weights
# Example usage with synthetic data
if __name__ == "__main__":
    # Create a synthetic dataset
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple linear boundary
    # Add intercept term to X
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the bias term
    # Initialize weights
    weights = np.zeros(X.shape[1])
    # Set hyperparameters
    learning_rate = 0.01
    iterations = 10000
    # Train the model
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    # Make predictions
    y_pred = predict(X, weights)
    y_pred_classes = (y_pred >= 0.5).astype(int)
    # Accuracy
    accuracy = np.mean(y_pred_classes == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
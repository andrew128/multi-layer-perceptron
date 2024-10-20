import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        batch_size = X.shape[0]
        
        # Gradient of the loss with respect to the output
        self.output_delta = output - y  # For cross-entropy loss and softmax output
        
        # Gradient for the hidden-to-output weights
        self.dW2 = (1 / batch_size) * np.dot(self.a1.T, self.output_delta)
        self.db2 = (1 / batch_size) * np.sum(self.output_delta, axis=0, keepdims=True)
        
        # Gradient for the input-to-hidden weights
        self.z1_delta = np.dot(self.output_delta, self.W2.T) * self.relu_derivative(self.a1)
        self.dW1 = (1 / batch_size) * np.dot(X.T, self.z1_delta)
        self.db1 = (1 / batch_size) * np.sum(self.z1_delta, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
    
    def cross_entropy_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
    
    def train(self, X, y, epochs, learning_rate):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.cross_entropy_loss(y, output)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
if __name__ == "__main__":
    # Create a simple multi-class dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])  # One-hot encoded

    # Create and train the MLP
    mlp = MLP(input_size=2, hidden_size=4, output_size=3)
    mlp.train(X, y, epochs=1000, learning_rate=0.1)

    # Test the trained MLP
    predictions = mlp.forward(X)
    print("Predictions:")
    print(predictions)
    print("Predicted classes:")
    print(np.argmax(predictions, axis=1))
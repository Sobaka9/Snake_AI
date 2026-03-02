from abc import ABC, abstractmethod


class NeuralNet(ABC):
    
    @abstractmethod
    def forward(self, x):
        """ Perform a forward pass through the network.
        Args:
            x (np.ndarray): (batch_size, state_size)
        """
        pass

    @abstractmethod
    def backward(self, grad_out, lr):
        """ Perform a backward pass and update weights.
        Args:
            grad_out (np.ndarray): (batch_size, action_size) Gradient of loss w.r.t. output
            lr (float): Learning rate
        """
        pass

    @abstractmethod
    def copy_weights_from(self, other):
        """Copy weights from another network."""
        pass

    @abstractmethod
    def save(self, path):
        """Save model weights to a file."""
        pass

    @abstractmethod
    def load(self, path):
        """Load model weights from a file."""
        pass

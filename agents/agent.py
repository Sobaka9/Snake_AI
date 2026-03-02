from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = float(kwargs.get("epsilon", 1.0))

    @abstractmethod
    def select_action(self, state, training=True):
        """Given a state, select an action.

        Args:
            state (np.ndarray): Current state of the environment.
            training (bool): Whether the agent is in training mode (for exploration).
        
        Returns:
            action (int): The action to take.
        """
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay.

        Args:
            state (np.ndarray): The state before taking the action.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The state after taking the action.
            done (bool): Whether the episode ended after this action.
        """
        pass

    @abstractmethod
    def train_step(self):
        """Perform a training step."""
        pass

    @abstractmethod
    def end_episode(self):
        """Called at the end of each episode to perform any necessary updates."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the agent's model to a file."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the agent's model from a file."""
        pass

from abc import ABC, abstractmethod

class Game(ABC):

    @abstractmethod
    def reset(self):
        """Reset the game to its initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Apply the given action to the game and return the new state, reward, and done flag."""
        raise NotImplementedError
    
    @abstractmethod
    def get_render_data(self):
        """Return data necessary for rendering the game."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_size(self):
        """Return the size of the game state."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_size(self):
        """Return the size of the action space."""
        raise NotImplementedError
 
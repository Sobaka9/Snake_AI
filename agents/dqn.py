import numpy as np
import random
from collections import deque
import pickle
import os

from agents.neural_net import NeuralNet
from agents.agent import Agent

class DQNet(NeuralNet):
    def __init__(self, state_size, hidden_size, action_size):
        self.layers = []
        sizes = [state_size] + list(hidden_size) + [action_size]
        for i in range(len(sizes) - 1):
            neurons_in = sizes[i]
            W = np.random.randn(neurons_in, sizes[i + 1]) * np.sqrt(2. / neurons_in)
            b = np.zeros((1, sizes[i + 1]))
            self.layers.append({'W': W, 'b': b, "cache": None})

    def forward(self, x):
        """ Perform a forward pass through the network.
        Args:
            x (np.ndarray): (batch_size, state_size)
        """
        out = x
        for i, layer in enumerate(self.layers):
            z = out @ layer['W'] + layer['b']
            layer["cache"] = (out, z) # Store input and pre-activation for backprop
            if i < len(self.layers) - 1:
                out = np.maximum(0, z)  # ReLU activation
            else:
                out = z  # Output layer (no activation)
        return out

    def backward(self, grad_out, lr):
        """ Perform a backward pass and update weights.
        Args:
            grad_out (np.ndarray): (batch_size, action_size) Gradient of loss w.r.t. output
            lr (float): Learning rate
        """
        g = grad_out
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            inp, z = layer["cache"]
            if i < len(self.layers) - 1:
                g = g * (z > 0)  # ReLU backward
            dW = inp.T @ g
            db = g.sum(axis=0, keepdims=True)
            g = g @ layer["W"].T
            layer["W"] -= lr * dW
            layer["b"] -= lr * db

    
    def copy_weights_from(self, other):
        for sl, tl in zip(self.layers, other.layers):
            sl["W"] = tl["W"].copy()
            sl["b"] = tl["b"].copy()

    def save(self, path):
        data = [{"W": l["W"], "b": l["b"]} for l in self.layers]
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        for layer, d in zip(self.layers, data):
            layer["W"] = d["W"]
            layer["b"] = d["b"]


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(Agent):
    """
    Hyperparameters exposed for the GUI:
        lr            Learning rate
        gamma         Discount factor
        epsilon       Initial exploration rate
        epsilon_min   Min exploration rate
        epsilon_decay Multiplicative decay per episode
        batch_size    Mini-batch size for training
        memory_size   Replay buffer capacity
        hidden_size   Hidden layer sizes (list of ints)
        target_update Target network sync every N episodes
    """

    HYPERPARAMS = {
        "lr":            ("float", 0.001,  1e-5, 0.1),
        "gamma":         ("float", 0.95,   0.5,  0.9999),
        "epsilon":       ("float", 1.0,    0.0,  1.0),
        "epsilon_min":   ("float", 0.01,   0.0,  0.5),
        "epsilon_decay": ("float", 0.995,  0.9,  0.9999),
        "batch_size":    ("int",   64,     8,    512),
        "memory_size":   ("int",   100000, 1000, 1000000),
        "hidden_size":   ("str",   "256,256", None, None),
        "target_update": ("int",   10,     1,    100),
    }

    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = float(kwargs.get("lr", 0.001))
        self.gamma = float(kwargs.get("gamma", 0.95))
        self.epsilon = float(kwargs.get("epsilon", 1.0))
        self.epsilon_min = float(kwargs.get("epsilon_min", 0.01))
        self.epsilon_decay = float(kwargs.get("epsilon_decay", 0.995))
        self.batch_size = int(kwargs.get("batch_size", 64))
        self.memory_size = int(kwargs.get("memory_size", 100000))
        self.target_update = int(kwargs.get("target_update", 10))

        hidden_raw = kwargs.get("hidden_size", "256,256")
        self.hidden_size = [int(x) for x in str(hidden_raw).split(",")]

        # Networks
        self.qnet = DQNet(state_size, self.hidden_size, action_size)
        self.target_net = DQNet(state_size, self.hidden_size, action_size)
        self.target_net.copy_weights_from(self.qnet)

        self.memory = ReplayBuffer(self.memory_size)
        self.episode_count = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon: # Exploration
            return random.randrange(self.action_size)

        q = self.qnet.forward(state.reshape(1, -1)) # Exploitation
        return int(np.argmax(q[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample a mini-batch and do one gradient step. Returns loss."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Current Q values
        q_values = self.qnet.forward(states)

        # Target Q values (Double-DQN style)
        q_next = self.target_net.forward(next_states)
        targets = q_values.copy()

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        # MSE loss gradient: dL/dQ = 2*(Q - target)/N
        loss_grad = 2 * (q_values - targets) / self.batch_size
        loss = float(np.mean((q_values - targets) ** 2))

        self.qnet.backward(loss_grad, self.lr)
        return loss

    def end_episode(self):
        self.episode_count += 1
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        # Sync target network
        if self.episode_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.qnet)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.qnet.save(path)

    def load(self, path):
        self.qnet.load(path)
        self.target_net.copy_weights_from(self.qnet)

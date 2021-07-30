import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from bcnetwork import DeepQNetwork
from bcmemory import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=1412, nb_hidden=128, learning_rate=5e-4, memory_size=int(1e5),
                 prioritized_memory=False, batch_size=64, gamma=0.99, small_eps=1e-5, update_frequency=4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            nb_hidden (int): intermediate layer size 
            learning_rate (float): learning rate of networks
            memory_size (int): max memory size of replay/priority buffer
            batch_size (int): batch size to do learning
            gamma (float): discount rate
            update_frequency (int): n episodes after which to update the target network from local network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.small_eps = small_eps
        self.update_frequency = update_frequency
        self.losses = []

        # Q-Network
        self.DeepQNetwork_w = DeepQNetwork(self.state_size, self.action_size,  seed=seed).to(device)
        self.DeepQNetwork_w_target = DeepQNetwork(self.state_size, self.action_size, seed=seed).to(device)       
        self.optimizer = optim.Adam(self.DeepQNetwork_w.parameters(), lr=self.learning_rate)

        # Define memory
        self.memory = ReplayMemory(self.memory_size, self.batch_size)
            
        # Initialize time step (for updating every update_frequency steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, i):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_frequency time steps.
        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:            
                experiences = self.memory.sample()                    
                self.learn(experiences)             

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps : epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.DeepQNetwork_w.eval()
        with torch.no_grad():
            action_values = self.DeepQNetwork_w(state)
        self.DeepQNetwork_w.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size)).astype(int)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            small_e (float): 
        """
        
        states, actions, rewards, next_states, dones = experiences
            
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.DeepQNetwork_w_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.DeepQNetwork_w(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        self.losses.append(loss)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for target_param, local_param in zip(self.DeepQNetwork_w_target.parameters(), self.DeepQNetwork_w.parameters()):
            target_param.data.copy_(local_param.data)               

    def save_model(self, path):
        torch.save(self.DeepQNetwork_w.state_dict(), path)

    def load_model(self, path):
        self.DeepQNetwork_w.load_state_dict(torch.load(path))
    
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class DeepQNetwork(nn.Module):
    """Policy network for the agent. It has input nodes as states and out nodes as action values.
       In essence it captures action value function for the policy."""

    def __init__(self, state_size, action_size, seed=199):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            layers (tuple or list): Size of each input sample for each layer
            seed (int): Random seed
        """
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, action_size, bias=False)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
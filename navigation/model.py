import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.hidden_sizes = [128, 64]
        self.fc1 = nn.Linear(state_size, self.hidden_sizes[0])
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        #self.fc3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.hidden_layers = [self.fc1, self.fc2]
        self.output = nn.Linear(self.hidden_sizes[1], action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = state
        for hl in self.hidden_layers:
            x = F.relu(hl(x))
        x = self.output(x)

        return x

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.hidden_sizes = [128, 32, 32]
        self.fc1 = nn.Linear(state_size, self.hidden_sizes[0])
        self.fc2v = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc2a = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[2])
        self.outputv = nn.Linear(self.hidden_sizes[1], 1)
        self.outputa = nn.Linear(self.hidden_sizes[2], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = state
        x = F.relu(self.fc1(x))
        v = F.relu(self.fc2v(x))
        a = F.relu(self.fc2a(x))
        v = self.outputv(v)
        a = self.outputa(a)
        output = v + a - a.max()

        return output
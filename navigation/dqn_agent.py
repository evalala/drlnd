import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self._sample_experiences()
                self.learn(experiences, GAMMA)

    def _sample_experiences(self):
        experiences = self.memory.sample()
        return experiences

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _compute_next_action_values(self, next_states):
        """Compute action values for next_states using the target network."""
        action_values = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        return action_values

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        self.optimizer.zero_grad()

        # Get max predicted Q values (for next states) from target model
        action_values = self._compute_next_action_values(next_states)

        #Compute Q targets for current states
        targets = rewards + (gamma * action_values * (1 - dones))

        # Get expected Q values from local model
        outputs = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class DoubleDQNAgent(Agent):

    def _compute_next_action_values(self, next_states):
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        action_values = self.qnetwork_target(next_states).gather(1, next_actions)
        return action_values

class PrioritizedExperienceReplayAgent(Agent):

    def __init__(self, state_size, action_size, seed, use_is = True):
        """Initialize an Agent object sampling experiences using prioritized experience replay.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            use_is: flag indicating whether to use importance sampling when computing the sampling probabilities
        """
        super().__init__(state_size, action_size, seed)
        # Replay memory
        self.memory = PrioritizedExperienceReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.use_is = use_is

    def step(self, state, action, reward, next_state, done):
        #compute priority of the experience
        next_state_value = self.qnetwork_target(torch.from_numpy(next_state).float()).detach().max().numpy()
        action_values = self.qnetwork_local(torch.from_numpy(state).float()).detach().numpy()
        action_value = action_values[action]
        priority = reward + GAMMA*next_state_value - action_value
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, priority)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.learn(GAMMA)


    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            gamma (float): discount factor
        """
        exp_indices, experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        self.optimizer.zero_grad()

        # Get max predicted Q values (for next states) from target model
        action_values = self._compute_next_action_values(next_states)

        # # Compute Q targets for current states
        targets = rewards + (gamma * action_values * (1 - dones))

        # Get expected Q values from local model
        outputs = self.qnetwork_local(states).gather(1, actions)

        if self.use_is:
            #rescale loss using importance sampling
            probs = self.memory.compute_sample_probs()[exp_indices]
            is_weights = torch.from_numpy((1/(len(self.memory)*probs)**self.memory.EXP_B).astype('float32')).unsqueeze(1)
            loss = 1/is_weights.max()*torch.sum(is_weights*((targets-outputs)**2))
        else:
            loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        #update priorities of experiences
        self.memory.update_prios(exp_indices, (targets-outputs).detach().numpy().squeeze())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

class DuelingDQNAgent(Agent):
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

class DuelingDDQNAgent(DuelingDQNAgent, DoubleDQNAgent):

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

class DuelingPERDDQNAgent(DuelingDQNAgent, PrioritizedExperienceReplayAgent, DoubleDQNAgent):
    def __init__(self, state_size, action_size, seed, use_is = True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            use_is: flag indicating whether to use importance sampling when computing the sampling probabilities
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedExperienceReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.use_is = use_is


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedExperienceReplayBuffer(ReplayBuffer):

    MIN_SAMPLE_PROB = 1/(1e5)
    EXP_A = 0.6
    EXP_B = 0.4


    def __init__(self, *args):
        super().__init__(*args)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = np.zeros(0)

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities = np.append(self.priorities, [priority])
        if len(self.priorities) > len(self.memory):
            self.priorities = self.priorities[1:]

    def compute_sample_probs(self):
        prios = (np.abs(self.priorities) + self.MIN_SAMPLE_PROB)**self.EXP_A
        probs = (prios/prios.sum()).flatten()
        return probs

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = self.compute_sample_probs()
        experience_inds = np.random.choice(range(len(self.memory)), size=self.batch_size, replace = False, p = probs).astype(np.int32)
        experiences = [self.memory[i] for i in experience_inds]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return experience_inds, (states, actions, rewards, next_states, dones)

    def update_prios(self, experience_inds, prios):
        self.priorities[experience_inds] = prios


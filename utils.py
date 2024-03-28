import collections
import gymnasium as gym
import itertools
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Tuple, Deque, Optional, Callable

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm

from GolfEnv import GolfEnv

class QNetwork(torch.nn.Module):
    """
    A Q-Network implemented with PyTorch.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        First fully connected layer.
    layer2 : torch.nn.Linear
        Second fully connected layer.
    layer3 : torch.nn.Linear
        Third fully connected layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Define the forward pass of the QNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        """
        Initialize a new instance of QNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        nn_l1 : int
            The number of neurons on the first layer.
        nn_l2 : int
            The number of neurons on the second layer.
        """
        super(QNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(n_observations, nn_l1)
        self.linear2 = torch.nn.Linear(nn_l1, nn_l2)
        self.linear3 = torch.nn.Linear(nn_l2, n_actions)

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the QNetwork.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (state).

        Returns
        -------
        torch.Tensor
            The output tensor (Q-values).
        """

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class QNetworkSequential(torch.nn.Sequential):
    def __init__(self, n_observations: int, n_actions: int, layers: List[int]):
        super().__init__()

        self.layers = [n_observations] + layers + [n_actions]

        self.linears = list()
        for i in range(1, len(self.layers)):
            self.add_module(f"hidden_layer_{i}", torch.nn.Linear(self.layers[i - 1], self.layers[i]))
            if i + 1 != len(self.layers):
                self.add_module(f"hidden_relu_{i}", torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


def test_q_network_agent(env: gym.Env, q_network: torch.nn.Module, num_episode: int = 1, render: bool = True) -> List[int]:
    """
    Test a naive agent in the given environment using the provided Q-network.

    Parameters
    ----------
    env : gym.Env
        The environment in which to test the agent.
    q_network : torch.nn.Module
        The Q-network to use for decision making.
    num_episode : int, optional
        The number of episodes to run, by default 1.
    render : bool, optional
        Whether to render the environment, by default True.

    Returns
    -------
    List[int]
        A list of rewards per episode.
    """
    episode_reward_list = []

    for episode_id in range(num_episode):

        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render_wrapper.render()

            # Convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
              action = torch.argmax(q_network(state_tensor)).item()

            state, reward, done, truncated, _ = env.step(action)

            episode_reward += reward

        episode_reward_list.append(episode_reward)
        print(f"Episode reward: {episode_reward}")

    return episode_reward_list


def test_q_network_agent(env: gym.Env, q_network: torch.nn.Module, num_episode: int = 1, render: bool = True) -> List[int]:
    """
    Test a naive agent in the given environment using the provided Q-network.

    Parameters
    ----------
    env : gym.Env
        The environment in which to test the agent.
    q_network : torch.nn.Module
        The Q-network to use for decision making.
    num_episode : int, optional
        The number of episodes to run, by default 1.
    render : bool, optional
        Whether to render the environment, by default True.

    Returns
    -------
    List[int]
        A list of rewards per episode.
    """
    episode_reward_list = []

    for episode_id in range(num_episode):

        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render_wrapper.render()

            # Convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
              action = torch.argmax(q_network(state_tensor)).item()

            state, reward, done, truncated, _ = env.step(action)

            episode_reward += reward

        episode_reward_list.append(episode_reward)
        print(f"Episode reward: {episode_reward}")

    return episode_reward_list


class EpsilonGreedy:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay:float,
                 env: gym.Env,
                 q_network: torch.nn.Module):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        np.int64
            The chosen action.
        """

        if random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(np.arange(8))
#             print(f"random {action}")
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(self.q_network(state_tensor)).item()
#             print(f"argmax {action}")

        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The multiplicative factor of learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


def train_naive_agent(env: gym.Env,
                      q_network: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: Callable,
                      epsilon_greedy: EpsilonGreedy,
                      device: torch.device,
                      lr_scheduler: _LRScheduler,
                      num_episodes: int,
                      gamma: float) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        for t in itertools.count():

            # TODO..
          state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
          action_dist = q_network(state_tensor).squeeze()
          action = epsilon_greedy(state)

          state, reward, done, truncated, _ = env.step(action)
          if done:
            break
          with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_estimation = torch.max(q_network(state_tensor)).item()
          optimizer.zero_grad()

          loss_fn(action_dist[action], torch.Tensor([gamma * q_estimation + reward]).to(device)[0]).backward()

          optimizer.step()


          episode_reward += reward

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list

class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]:
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        # Here, `random.sample(self.buffer, batch_size)`
        # returns a list of tuples `(state, action, reward, next_state, done)`
        # where:
        # - `state`  and `next_state` are numpy arrays
        # - `action` and `reward` are floats
        # - `done` is a boolean
        #
        # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
        # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return len(self.buffer)

def train_dqn1_agent(env: gym.Env,
                     q_network: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: Callable,
                     epsilon_greedy: EpsilonGreedy,
                     device: torch.device,
                     lr_scheduler: _LRScheduler,
                     num_episodes: int,
                     gamma: float,
                     batch_size: int,
                     replay_buffer: ReplayBuffer) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        for t in itertools.count():

            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # TODO...
            replay_buffer.add(state, action, reward, next_state, done)
            s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(min(len(replay_buffer), batch_size))
            s_dones = np.array(s_dones)
            s_ys = np.zeros_like(s_rewards) * 1. + s_rewards
            s_next_states_tensor = torch.tensor(s_next_states, dtype=torch.float32, device=device).unsqueeze(1)

            s_ys += gamma * (torch.max(q_network(s_next_states_tensor), dim=2).values.squeeze(-1)).detach().cpu().numpy() * (1 - s_dones)

            s_states_tensor = torch.tensor(s_states, dtype=torch.float32, device=device).unsqueeze(1)

            network_outputs = q_network(s_states_tensor).squeeze(1)[np.arange(len(s_ys)), s_actions]

            loss = loss_fn(torch.Tensor(s_ys).to(device), network_outputs)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            if done:
                break

            state = next_state
            episode_reward += reward

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    print(episode_reward_list)

    return episode_reward_list

def train_dqn2_agent(env: gym.Env,
                     q_network: torch.nn.Module,
                     target_q_network: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: Callable,
                     epsilon_greedy: EpsilonGreedy,
                     device: torch.device,
                     lr_scheduler: _LRScheduler,
                     num_episodes: int,
                     gamma: float,
                     batch_size: int,
                     replay_buffer: ReplayBuffer,
                     target_q_network_sync_period: int) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    target_q_network : torch.nn.Module
        The target Q-network to use for estimating the target Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.
    target_q_network_sync_period : int
        The number of episodes after which the target Q-network should be updated with the weights of the Q-network.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    iteration = 0
    episode_reward_list = []

    print_every = 50

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        for t in itertools.count():

            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            episode_reward += reward

            # Update the q_network weights with a batch of experiences from the buffer

            if len(replay_buffer) > batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

                # Convert to PyTorch tensors
                batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.int32, device=device)

                # TODO...
                batch_ys = torch.zeros_like(batch_rewards_tensor) + batch_rewards_tensor
                # s_next_states_tensor = torch.tensor(s_next_states, dtype=torch.float32, device=device).unsqueeze(1)
                # print(batch_dones_tensor.shape)
                # print(target_q_network(batch_next_states_tensor).shape)

                batch_ys += gamma * (torch.max(target_q_network(batch_next_states_tensor), dim=1).values.squeeze(-1)).detach() * (1 - batch_dones_tensor)

                network_outputs = q_network(batch_states_tensor)[np.arange(len(batch_ys)), batch_actions_tensor]
                loss = loss_fn(torch.Tensor(batch_ys), network_outputs)
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()

            # Update the target q-network

            # Every few training steps (e.g., every 100 steps), the weights of the target network are updated with the weights of the Q-network

            # TODO...
            if iteration % target_q_network_sync_period == 0:
                target_q_network.load_state_dict(q_network.state_dict())


            iteration += 1


            if done:
                break

            state = next_state

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

        if (episode_index + 1) % print_every == 0:
            print(np.mean(episode_reward_list[-print_every:]))

    return episode_reward_list

def get_episode_len(env, q_network, epsilon_greedy, device='cpu'):
    state, info = env.reset()
    episode_len = 0

    for t in itertools.count():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_dist = q_network(state_tensor).squeeze()
        action = epsilon_greedy(state)

        state, reward, done, truncated, _ = env.step(action)
        if done:
            break
        episode_len += 1

    return episode_len

class PolicyNetwork(torch.nn.Module):
    """
    A neural network used as a policy for the REINFORCE algorithm.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        A fully connected layer.

    Methods
    -------
    forward(state: torch.Tensor) -> torch.Tensor
        Define the forward pass of the PolicyNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize a new instance of PolicyNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        """
        super(PolicyNetwork, self).__init__()

        self.hidden_dim = 128

        self.linear1 = nn.Linear(n_observations, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, n_actions)

        self.linear = nn.Linear(n_observations, n_actions)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of each action for the given state.

        Parameters
        ----------
        state_tensor : torch.Tensor
            The input tensor (state).
            The shape of the tensor should be (N, dim),
            where N is the number of states vectors in the batch
            and dim is the dimension of state vectors.

        Returns
        -------
        torch.Tensor
            The output tensor (the probability of each action for the given state).
        """

        out = self.linear1(state_tensor)
        out = F.relu(out, inplace=False)
        out = self.linear2(out)
        out = F.relu(out, inplace=False)

        out = self.linear3(out)
        out = F.log_softmax(out)

        return out

def sample_discrete_action(policy_nn: PolicyNetwork,
                           state: NDArray[np.float64]) -> Tuple[int, torch.Tensor]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : NDArray[np.float64]
        The state based on which an action needs to be sampled.

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.

    """

    log_probs = policy_nn(torch.tensor(state, dtype=torch.float32))
    log_probs_np = log_probs.cpu().detach().numpy()
    log_probs_np -= log_probs_np.max()
    probs = np.exp(log_probs_np)
    probs /= probs.sum()
    sampled_action = np.random.choice(np.arange(len(log_probs)), p=probs)
    sampled_action_log_probability = log_probs[0]

    # Return the sampled action and its log probability.
    return sampled_action, sampled_action_log_probability

def sample_one_episode(env: gym.Env,
                       policy_nn: PolicyNetwork,
                       max_episode_duration: int,
                       render: bool = False) -> Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]:
    """
    Execute one episode within the `env` environment utilizing the policy defined by the `policy_nn` parameter.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log probability of action for each time step in the episode.
    """
    state, info = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_states.append(state)

    for t in range(max_episode_duration):

        if render:
            env.render_wrapper.render()

        action, action_log_probability = sample_discrete_action(policy_nn, torch.tensor(state))

        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        # replay_buffer.add(state, action, reward, next_state, done)

        episode_states.append(state)
        episode_actions.append(action)
        episode_log_prob_actions.append(action_log_probability)
        episode_rewards.append(reward)

        if done:
            break

    return episode_states[:-1], episode_actions, episode_rewards, episode_log_prob_actions

def avg_return_on_multiple_episodes(env: gym.Env,
                                    policy_nn: PolicyNetwork,
                                    num_test_episode: int,
                                    max_episode_duration: int,
                                    render: bool = False) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    float
        The average return.
    """

    # TODO...
    total_return = 0
    for i in range(num_test_episode):
        episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(env, policy_nn, max_episode_duration, render)
        total_return += sum(episode_rewards)

    average_return = total_return / num_test_episode

    return average_return

def train_reinforce_discrete(env: gym.Env,
                             num_train_episodes: int,
                             num_test_per_episode: int,
                             max_episode_duration: int,
                             learning_rate: float) -> Tuple[PolicyNetwork, List[float]]:
    """
    Train a policy using the REINFORCE algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode, by default EPISODE_DURATION.
    learning_rate : float
        The initial step size.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_nn = PolicyNetwork(state_size, action_size)
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)

    print_every = 50
    sum_avg = 0

    for episode_index in tqdm(range(num_train_episodes)):

        # TODO...

        episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(env, policy_nn, max_episode_duration, render=False)

        loss = 0

        for i in range(len(episode_states)):
            G = torch.sum(torch.tensor(episode_rewards[i:])).detach()
            loss += -G * episode_log_prob_actions[i]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes(env=env,
                                                          policy_nn=policy_nn,
                                                          num_test_episode=num_test_per_episode,
                                                          max_episode_duration=max_episode_duration,
                                                          render=False)

        # Monitoring
        episode_avg_return_list.append(test_avg_return)
        sum_avg += test_avg_return

    return policy_nn, episode_avg_return_list

class A2CNetwork(torch.nn.Module):
    """
    A neural network used as a policy for the REINFORCE algorithm.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        A fully connected layer.

    Methods
    -------
    forward(state: torch.Tensor) -> torch.Tensor
        Define the forward pass of the PolicyNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize a new instance of PolicyNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        """
        super(A2CNetwork, self).__init__()

        self.linear = torch.nn.Linear(n_observations, 128, bias=True)
        self.linear2 = torch.nn.Linear(128, n_actions, bias=True)
        self.linear3 = torch.nn.Linear(128, 1, bias=True)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.functional.softmax
        self.T = 1


    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of each action for the given state.

        Parameters
        ----------
        state_tensor : torch.Tensor
            The input tensor (state).
            The shape of the tensor should be (N, dim),
            where N is the number of states vectors in the batch
            and dim is the dimension of state vectors.

        Returns
        -------
        torch.Tensor
            The output tensor (the probability of each action for the given state).
        """

        out = self.linear(state_tensor)
        out = self.relu(out)
        action = self.linear2(out)
        action = self.softmax(action, dim=1)
        value = self.linear3(out)

        return action, value

def sample_discrete_action_value(policy_nn: A2CNetwork,
                           state: NDArray[np.float64]) -> Tuple[int, torch.Tensor]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : NDArray[np.float64]
        The state based on which an action needs to be sampled.

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.

    """

    # TODO...
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    output, value = policy_nn(state_tensor)

    cat_dist = torch.distributions.Categorical(output)
    sampled_action = cat_dist.sample()
    sampled_action_log_probability = cat_dist.log_prob(sampled_action)

    # Return the sampled action and its log probability.
    return sampled_action.item(), sampled_action_log_probability, value, cat_dist.entropy()

def sample_one_episode_value(env: gym.Env,
                       policy_nn: A2CNetwork,
                       max_episode_duration: int,
                       render: bool = False) -> Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]:
    """
    Execute one episode within the `env` environment utilizing the policy defined by the `policy_nn` parameter.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log probability of action for each time step in the episode.
    """
    state_t, info = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_values = []
    episode_entropies = []

    episode_states.append(state_t)

    for t in range(max_episode_duration):

        if render:
            env.render_wrapper.render()

        action, action_log_probability, value, entropy = sample_discrete_action_value(policy_nn, state_t)

        state_t, reward, done, truncated, _ = env.step(action)

        episode_actions.append(action)
        episode_log_prob_actions.append(action_log_probability)
        episode_values.append(value)
        episode_rewards.append(reward)
        episode_states.append(state_t)
        episode_entropies.append(entropy)
        if done or truncated:
            break



    return episode_states, episode_actions, episode_values, episode_rewards, episode_log_prob_actions, episode_entropies

def avg_return_on_multiple_episodes_value(env: gym.Env,
                                    policy_nn: A2CNetwork,
                                    num_test_episode: int,
                                    max_episode_duration: int,
                                    render: bool = False) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    float
        The average return.
    """

    # TODO...
    with torch.no_grad():
      cumulated_rewards = []
      for i in range(num_test_episode):
        episode_states, episode_actions, episode_values, episode_rewards, episode_log_prob_actions, episode_entropies = sample_one_episode_value(
            env, policy_nn, max_episode_duration,
            render=False
        )

        cumulated_rewards.append(sum(episode_rewards))


      average_return = sum(cumulated_rewards) / len(cumulated_rewards)
    return average_return

def train_a2c_discrete(env: gym.Env,
                             num_train_episodes: int,
                             num_test_per_episode: int,
                             max_episode_duration: int,
                             learning_rate: float) -> Tuple[A2CNetwork, List[float]]:
    """
    Train a policy using the A2C algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode, by default EPISODE_DURATION.
    learning_rate : float
        The initial step size.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n # .item()

    policy_nn = A2CNetwork(state_size, action_size)
    optimizer = torch.optim.AdamW(policy_nn.parameters(), lr=learning_rate)

    pts = []

    for episode_index in tqdm(range(num_train_episodes)):

        # TODO...
        episode_states, episode_actions, episode_values, episode_rewards, episode_log_prob_actions, episode_entropies = sample_one_episode_value(
            env, policy_nn, max_episode_duration=max_episode_duration,
            render=False
        )

        pts = list(episode_states)

        loss = 0
        gamma = 0.95
        action, action_log_probability, value, entropy = sample_discrete_action_value(policy_nn, episode_states[-1])
        R = value
        for i in range(len(episode_values)):
            next_value = episode_values[i + 1] if i + 1 < len(episode_rewards) else 0
            R = episode_rewards[i] + gamma * R
            loss += 0.5 * F.mse_loss(R, episode_values[i])
            loss +=  -((R - episode_values[i]).detach() * episode_log_prob_actions[i]).mean()
            loss -=  0.01 * episode_entropies[i][0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes_value(env=env,
                                                          policy_nn=policy_nn,
                                                          num_test_episode=num_test_per_episode,
                                                          max_episode_duration=max_episode_duration,
                                                          render=False)

        # Monitoring
        episode_avg_return_list.append(test_avg_return)

    return policy_nn, episode_avg_return_list
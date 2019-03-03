import numpy as np
from collections import deque
import torch


def train_dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, stop_when_solved=False,
        solved_ckpt_path=None, final_score = None, final_ckpt_path=None):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        stop_when_solved: whether to stop training once the env is solved (avg score of 13 over 100 episodes)
        solved_ckpt_path: path where the checkpoint will be solved once the environment is solved
        final_score: when an avg of final_score is reached over 100 episodes, training is stopped
        final_ckpt_path: path where the final checkpoint will be solved: either when final_score is reached, or after n_episodes
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    solved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[env.brain_names[0]]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[env.brain_names[0]]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if (not solved and np.mean(scores_window) >= 13.0):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            if solved_ckpt_path:
                torch.save(agent.qnetwork_local.state_dict(), solved_ckpt_path)
            solved = True
            if stop_when_solved:
                break
        if solved and final_score:
            if np.mean(scores_window) >= final_score:
                print('\nEnvironment reached final score in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                                   np.mean(scores_window)))
                break
    if not stop_when_solved:
        if final_ckpt_path:
            torch.save(agent.qnetwork_local.state_dict(), final_ckpt_path)
    return scores
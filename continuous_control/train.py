from collections import deque
import torch
import numpy as np

def train(agent, env, n_episodes=2000, max_t=1000, solved_score = 30, stop_when_solved=False,
          solved_ckpt_path_actor='checkpoint_solved_actor.pth', solved_ckpt_path_critic='checkpoint_solved_critic.pth',
          final_ckpt_path_actor='checkpoint_final_actor.pth', final_ckpt_path_critic='checkpoint_final_critic.pth'):
    """

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        solved_score: when a mean score of solved_score is reached over 100 episodes, the environment is considered solved
        stop_when_solved: whether to stop training once the env is solved (avg score of 13 over 100 episodes)
        solved_ckpt_path_actor: path where the checkpoint of the actor will be solved once the environment is solved
        solved_ckpt_path_critic: path where the checkpoint of the critic will be solved once the environment is solved
        final_ckpt_path_actor: path where the final checkpoint of the actor will be solved: either when final_score is reached, or after n_episodes
        final_ckpt_path_critic: path where the final checkpoint of the critic will be solved: either when final_score is reached, or after n_episodes
    """
    mean_scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    solved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[env.brain_names[0]]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(agent.num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[env.brain_names[0]]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        mean_score = np.mean(scores)
        scores_window.append(mean_score)  # save most recent score
        mean_scores.append(mean_score)  # save most recent score
        print('\rEpisode {} \tScore: {:.2f}'.format(i_episode, mean_score), end="\n")
        #print('\rEpisode {} \tLength: {} \tAverage Score: {:.2f}'.format(i_episode, ep_len, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if (not solved and np.mean(scores_window) >= solved_score):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            if solved_ckpt_path_actor:
                torch.save(agent.actor_local.state_dict(), solved_ckpt_path_actor)
                torch.save(agent.critic_local.state_dict(), solved_ckpt_path_critic)
            solved = True
            if stop_when_solved:
                break
    if not stop_when_solved:
        if final_ckpt_path_actor:
            torch.save(agent.actor_local.state_dict(), final_ckpt_path_actor)
            torch.save(agent.critic_local.state_dict(), final_ckpt_path_critic)
    return mean_scores



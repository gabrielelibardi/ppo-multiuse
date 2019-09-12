import time
import torch
import numpy as np


def test_performance(env, actor_critic, device, num_arenas_test, num_processes):
    """
     Evaluate current model on the test set.
     """

    recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(1, 1).to(device)

    obs = env.reset()
    episode_steps = np.zeros(num_processes)
    episode_reward = torch.FloatTensor([[0.0] for _ in range(num_processes)])
    tested_arenas = []

    while len(tested_arenas) < num_arenas_test:

        with torch.no_grad():
            (value, action, action_log_prob, recurrent_hidden_states,
             dist_entropy) = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=False)

        # Obser reward and next obs
        obs, reward, done, infos = env.step(action)
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done]).to(device)
        episode_steps += 1
        episode_reward += reward

        # If done then clean the history of observations and
        # log episode performance.
        for process_num, process_done in enumerate(done):
            if process_done:
                episode_reward[process_num] = 0.0
                episode_steps[process_num] = 0.0
                if infos[process_num]['arena'] not in tested_arenas:
                    tested_arenas.append(infos[process_num]['arena'])

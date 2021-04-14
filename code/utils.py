from simple_pid import PID
import math

"""
Utilities
"""


def collect_episode(env, agent, behavior_buffer, trajectory_buffer, render_option, num_eps):
    """
    Collect steps from a single episode play and record
    with replay buffer

    :param agent:
    :param env: OpenAI gym environment
    :param behavior_buffer: replay buffer for behavior planner
    :param trajectory_buffer: replay buffer for trajectory planner
    :param render_option: (bool) if should render the game play
    :param num_eps:  the number of the current iteration
    :return: None
    """
    obs = env.reset()
    state = obs['birdeye']
    print("shape:", obs["birdeye"].shape)
    done = False
    re = 0
    while not done:
        if render_option == 'collect':
            env.render()
        trajectory_point, behavior_state, trajectory_state = agent.collect_policy(state, num_eps)
        action = compute_action(trajectory_point)
        next_obs, reward, done, _ = env.step(action)
        next_state = next_obs['birdeye']
        reward_behavior, reward_trajectory = agent.hybrid_reward(done, reward)
        agent.store(behavior_buffer, trajectory_buffer, behavior_state, trajectory_state,
                    reward_behavior, reward_trajectory, next_state, done)
        state = next_state
        re = re + reward_behavior + reward_trajectory
    agent.update_epsilon(re, num_eps)


def compute_avg_reward(env, policy, num_episodes):
    """
    Compute the average reward across num_episodes under policy

    :param env: OpenAI gym environment
    :param policy: DQN agent policy
    :param num_episodes: the number of episode to take average from
    :return: (int) average reward
    """
    total_return = 0.0
    for _ in range(num_episodes):
        obs = env.reset()
        state = obs['birdeye']
        done = False
        episode_return = 0.0
        while not done:
            trajectory_point = policy(state)
            action = compute_action(trajectory_point)
            next_obs, reward, done, _ = env.step(action)
            next_state = next_obs['birdeye']
            if done:
                reward = -1.0
            episode_return += reward
            state = next_state
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return


def train_buffer(agent, buffer, batch_size, eps_cnt, t_net_update_frequency):
    used_target_network_update_frequency = t_net_update_frequency
    if eps_cnt % used_target_network_update_frequency == 0:
        agent.update_target_network()

    if buffer.can_sample_batch(batch_size):
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = \
            buffer.sample_batch(batch_size=batch_size)
        loss = agent.train(state_batch=state_batch,
                           next_state_batch=next_state_batch,
                           action_batch=action_batch,
                           reward_batch=reward_batch, done_batch=done_batch,
                           batch_size=batch_size)
        return loss
    else:
        print('Not enough sample, skipping...')


def sigma(first, last, const):
    summation = 0
    for i in range(first, last + 1):
        summation += const * i
    return summation


def compute_action(point):
    pid_acc = PID(1, 0.1, 0.05, setpoint=1)
    pid_steer = PID(1, 0.1, 0.05, setpoint=1)
    dx = point[0]
    dy = point[1]
    if dx == 0:
        if dy < 0:
            angle = -1.571
        else:
            angle = 1.571
    else:
        angle = math.atan(dy / dx)
    distance = math.sqrt(dx * dx + dy * dy)
    acc = pid_acc(distance)
    steer = pid_steer(angle)
    action = [acc, steer]
    return action

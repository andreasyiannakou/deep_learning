import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
import datetime
import pickle

d1 = datetime.datetime.now()

# None, ER, PER
replay = 'PER'

def create_replay_buffer(buffer_type, size):
    if buffer_type == 'PER':
        replay_buffer = PrioritizedReplayBuffer(size, 0.5)
    elif buffer_type == 'ER':
        replay_buffer = ReplayBuffer(size)
    else:
        replay_buffer = None
    return replay_buffer

def model(input1, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = input1
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    # Create the environment
    env = gym.make("CartPole-v0")
    # Create all the functions necessary to train the model
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
        q_func=model,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        double_q=False
    )
    # Create the replay buffer
    replay_buffer = create_replay_buffer(replay, 50000)
    # Create the schedule for exploration starting from 1 (every action is random) down to
    exploration = LinearSchedule(schedule_timesteps=100000, initial_p=1.0, final_p=0.02)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    obs = env.reset()
    for t in itertools.count():
        # Take action and update exploration to the newest value
        action = act(obs[None], update_eps=exploration.value(t))[0]
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        if replay != 'None':
            replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0)

        is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
        if is_solved:       
            break
        else:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 1000:
                if replay != 'None':
                    if replay == 'ER':
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    if replay == 'PER':
                        obses_t, actions, rewards, obses_tp1, dones, weights, idxes = replay_buffer.sample(32, beta=0.5)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # Update target network periodically.
            if t % 1000 == 0:
                update_target()

        if done and len(episode_rewards) % 10 == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()
            if is_solved:
                break
        if len(episode_rewards) == 5000:
            break

env.close()

del episode_rewards[-1]

with open('Duel_DQN_PER_1.pkl', 'wb') as f:
    pickle.dump(episode_rewards, f)

d2 = datetime.datetime.now()
print(d2-d1)
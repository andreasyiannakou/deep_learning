import gym
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
import datetime
d1 = datetime.datetime.now()

logger.configure()
env = make_atari('PongNoFrameskip-v4')
env = bench.Monitor(env, logger.get_dir())
env = deepq.wrap_atari_dqn(env)

model = deepq.learn(
    env,
    "conv_only",
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=True,
    lr=1e-4,
    total_timesteps=50000,
    buffer_size=10000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    train_freq=4,
    learning_starts=1000,
    target_network_update_freq=1000,
    gamma=0.99,
)

model.save('pong_model_50k.pkl')
env.close()

d2 = datetime.datetime.now()
print(d2-d1)

"""
import gym
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari

env = gym.make("PongNoFrameskip-v4")
env = deepq.wrap_atari_dqn(env)
model = deepq.learn(
    env,
    "conv_only",
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=True,
    total_timesteps=0,
    load_path="pong_model_50k.pkl"
)

while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(model(obs[None])[0])
        episode_rew += rew
print("Episode reward", episode_rew)

"""
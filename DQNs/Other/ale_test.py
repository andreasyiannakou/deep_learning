
import gym
import time

env = gym.make('CartPole-v0')
env.reset()
env.render()
time.sleep(1)
env.close()

env = gym.make('Pong-v0')
env.reset()
env.render()
time.sleep(1)
env.close()
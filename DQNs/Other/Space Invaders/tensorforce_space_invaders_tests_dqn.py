import numpy as np
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import datetime
import pickle

o1 = datetime.datetime.now()

# Either 'No_Mem', 'ER' or 'PER'
memory_types = ['No_Mem', 'ER', 'PER']
# Either 'DQN' or 'DDQN'
model_type = 'DQN'

def define_memory(memory_type):
    # set the memory
    if memory_type == 'ER':
        memory=dict(
            type='replay',
            include_next_states=True,
            capacity=50000
        )
    elif memory_type == 'PER':
        memory=dict(
            type='prioritized_replay',
            include_next_states=True,
            buffer_size=25000,
            capacity=50000
        )
    else:
        memory = None
    return memory

def define_model(model_type):
    # set the model type
    if model_type == 'DQN':
        double_model = False
    else:
        double_model = True
    return double_model

def create_agent(memory, double_model, environment):
    # create the agent
    agent = DQNAgent(states=environment.states, actions=environment.actions, network=network_spec, double_q_model=double_model, memory=memory,
        update_mode=None,
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        states_preprocessing=[
            dict(type='running_standardize'),
            dict(type='sequence')
        ],
        target_sync_frequency=1000,
        # Comment in to test exploration types
        actions_exploration=dict(
            type="epsilon_decay",
            initial_epsilon=1.0,
            final_epsilon=0.1,
            timesteps=3500000
        )
    )
    return agent

# Callback function printing episode statistics
# Callback function printing episode statistics
def episode_finished(r):
    if r.episode % 100 == 0:
        print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
        print("Episode reward: {}".format(r.episode_rewards[-1]))
        print("Average of last 100 rewards: {}".format(np.mean(r.episode_rewards[-100:])))
    return True

# set the network layout
network_spec = [dict(type='dense', size=64), dict(type='dense', size=32), dict(type='dense', size=32)]

for memory_type in memory_types:
    #create filename
    fn = 'Space_Invaders_3_5_' + str(model_type) + '_' + str(memory_type) + '.pkl'
    print(fn)
    d1 = datetime.datetime.now()
    # set the breakout atari environment
    environment = OpenAIGym('SpaceInvaders-ram-v0', visualize=False)
    #define the memory and model types
    memory = define_memory(memory_type)
    double_model = define_model(model_type)
    # create the agent
    agent = create_agent(memory, double_model, environment)
    # create the runner
    runner = Runner(agent=agent, environment=environment)
    # teach the agent
    runner.run(episodes=5000, episode_finished=episode_finished)
    runner.close()
    # Print statistics
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(ep=runner.episode, ar=np.mean(runner.episode_rewards[-100:])))
    # print time taken
    d2 = datetime.datetime.now()
    print(d2-d1)
    #save results
    episode_rewards = runner.episode_rewards
    episode_times = runner.episode_times
    episode_timesteps = runner.episode_timesteps
    results = [episode_rewards, episode_times, episode_timesteps]
    with open(fn, 'wb') as f:
        pickle.dump(results, f)
    del environment, memory, double_model, agent, runner, episode_rewards, episode_times, episode_timesteps, results

o2 = datetime.datetime.now()
print(o2-o1)
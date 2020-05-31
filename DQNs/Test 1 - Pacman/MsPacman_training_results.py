import json
import matplotlib.pyplot as plt
import pandas as pd

# load log files
with open(r'C:\Users\Andreas\Documents\GitHub\DeepLearning\DQNs\Test 1 - Pacman\dqn_MsPacman-v0_log.json') as json_file:  
    dqn = json.load(json_file)

with open(r'C:\Users\Andreas\Documents\GitHub\DeepLearning\DQNs\Test 1 - Pacman\duel_MsPacman-v0_log.json') as json_file:  
    duel = json.load(json_file)
    

# takes the results from all the episodes, smooth them and plot them in a line graph
def draw_smoothed_results_graph(episodes, rewards, steps, smoothing_window):
    fig, ax1 = plt.subplots()
    # x axis
    rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    line1, = plt.plot(rewards_smoothed, color='red', label="cumulative reward")
    ax1.set_ylabel('cumulative reward (smoothed)')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    steps_smoothed = pd.Series(steps).rolling(smoothing_window, min_periods=smoothing_window).mean()
    line2, = plt.plot(steps_smoothed, color='blue', label="steps")
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('steps (smoothed)')
    ax2.tick_params('y')
    plt.title("Episode reward amd steps over time (Smoothed over window size {})".format(smoothing_window))
    fig.tight_layout()
    plt.legend(handles=[line1, line2])
    plt.show()
    return

# draw the results graphs
draw_smoothed_results_graph(dqn['episode'], dqn['episode_reward'], dqn['nb_episode_steps'], 100)
draw_smoothed_results_graph(duel['episode'], duel['episode_reward'], duel['nb_episode_steps'], 100)

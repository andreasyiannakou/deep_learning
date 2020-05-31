import matplotlib.pyplot as plt
import numpy as np
import pickle

# get the metrics for the results table
def get_results(rewards_list):
    episodes = []
    scores = []
    for test in rewards_list:
        episodes.append(len(test))
        scores.append(np.mean(test[-101:-1]))
    return np.mean(episodes), np.min(episodes), np.mean(scores), np.max(scores)

# read in the results files and generate the results
dqn_none = []
with open('DQN_None_1.pkl', 'rb') as f:
    dqn_none.append(pickle.load(f))
with open('DQN_None_2.pkl', 'rb') as f:
    dqn_none.append(pickle.load(f))
with open('DQN_None_3.pkl', 'rb') as f:
    dqn_none.append(pickle.load(f))
dqn_none_mean_episodes, dqn_none_min_episodes, dqn_none_mean_score, dqn_none_max_score = get_results(dqn_none)


dqn_er = []
with open('DQN_ER_1.pkl', 'rb') as f:
    dqn_er.append(pickle.load(f))
with open('DQN_ER_2.pkl', 'rb') as f:
    dqn_er.append(pickle.load(f))
with open('DQN_ER_3.pkl', 'rb') as f:
    dqn_er.append(pickle.load(f))
dqn_er_mean_episodes, dqn_er_min_episodes, dqn_er_mean_score, dqn_er_max_score = get_results(dqn_er)

dqn_per = []
with open('DQN_PER_1.pkl', 'rb') as f:
    dqn_per.append(pickle.load(f))
with open('DQN_PER_2.pkl', 'rb') as f:
    dqn_per.append(pickle.load(f))
with open('DQN_PER_3.pkl', 'rb') as f:
    dqn_per.append(pickle.load(f))
dqn_per_mean_episodes, dqn_per_min_episodes, dqn_per_mean_score, dqn_per_max_score = get_results(dqn_per)

ddqn_none = []
with open('Double_DQN_None_1.pkl', 'rb') as f:
    ddqn_none.append(pickle.load(f))
with open('Double_DQN_None_2.pkl', 'rb') as f:
    ddqn_none.append(pickle.load(f))
with open('Double_DQN_None_3.pkl', 'rb') as f:
    ddqn_none.append(pickle.load(f))
ddqn_none_mean_episodes, ddqn_none_min_episodes, ddqn_none_mean_score, ddqn_none_max_score = get_results(ddqn_none)

ddqn_er = []
with open('Double_DQN_ER_1.pkl', 'rb') as f:
    ddqn_er.append(pickle.load(f))
with open('Double_DQN_ER_2.pkl', 'rb') as f:
    ddqn_er.append(pickle.load(f))
with open('Double_DQN_ER_3.pkl', 'rb') as f:
    ddqn_er.append(pickle.load(f))
ddqn_er_mean_episodes, ddqn_er_min_episodes, ddqn_er_mean_score, ddqn_er_max_score = get_results(ddqn_er)

ddqn_per = []
with open('Double_DQN_PER_1.pkl', 'rb') as f:
    ddqn_per.append(pickle.load(f))
with open('Double_DQN_PER_2.pkl', 'rb') as f:
    ddqn_per.append(pickle.load(f))
with open('Double_DQN_PER_3.pkl', 'rb') as f:
    ddqn_per.append(pickle.load(f))
ddqn_per_mean_episodes, ddqn_per_min_episodes, ddqn_per_mean_score, ddqn_per_max_score = get_results(ddqn_per)



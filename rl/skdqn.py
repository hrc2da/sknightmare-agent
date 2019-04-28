import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from rl.environment import Environment
from rl.qagent import QAgent
from bayesopt.bo import SKOutcomes, PreferenceDummy, SKBayesOpt
import time
import argparse
import numpy as np
import csv
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", nargs='?', default='empty')
    parser.add_argument("--episodes", nargs='?', default='empty')
    args = parser.parse_args()
    if args.saved == 'empty':
        saved_model = None
        saved_weights = None
    else:
        base = args.saved
        print("Using save name {}".format(base))
        saved_model = base + ".yaml"
        saved_weights = base + ".h5"
    if args.episodes == 'empty':
        num_episodes = 30
    else:
        num_episodes = int(args.episodes)
    outcomes = SKOutcomes()
    preferences = [0.5,0.7,0.05,0.5,0.3,0.3,0.4]
    pd = PreferenceDummy(outcomes,preferences)
    bo = SKBayesOpt(pd)
    env = Environment(width = 65, height = 34, tables = [], equipment = [], staff = [], reward_model = bo)
    qa = QAgent(input_shape = (65,34),saved_model=saved_model, saved_weights=saved_weights)
    #qa = QAgent(input_shape = (65,34))
    episodes = num_episodes
    base_iterations_per_ep = 2 
    max_iterations_per_ep = 20
    restaurants = []
    state = env.reset(init_state = None) # implement this!!!
    stats = []
    for e in range(episodes):
        episode_stats = {"mistakes":0,"nops":0,"actions":0,"rewards":[]}
        print("Episode {}".format(e))
        state = env.reset(init_state = None)
        #restaurants.append((state,None,0))
        num_iter = min(base_iterations_per_ep * (e+1), max_iterations_per_ep)
        for i in range(num_iter): # scale up the number of episodes as we learn more hopefully
            q_vals = qa.predict_q(state.image)
            source_mask,target_mask = qa.get_mask(state.image)
            action = qa.get_action(q_vals[0],source_mask,target_mask)
            next_state, reward = env.step(action)
            if reward > 5000:
                restaurants.append((next_state,action,reward))
                next_state.png.save("/content/"+str(reward)+".png")
            qa.remember(state.image, action, reward, next_state.image)
            if reward == -1e6:
                episode_stats["mistakes"] += 1
            elif reward == -1e8:
                episode_stats["nops"] += 1
            else:
                episode_stats["actions"] += 1
            episode_stats["rewards"].append(reward)
        next_state.png.show()
        stats.append(episode_stats)
        qa.retrain(num_iter)
    qa.save_model("skdqn")
    with open("skdnlog_{}.csv".format(time.time()),"w+") as logfile:
        logwriter = csv.writer(logfile,delimiter=',')
        logwriter.writerow(["mean_reward","max_reward","mistakes","nops","actions"])
        for e in stats:
            max_reward = max(e["rewards"])
            mean_reward = np.mean(e["rewards"])
            logwriter.writerow([mean_reward,max_reward,e["mistakes"],e["nops"],e["actions"]])
    with open("allstars_{}.pkl".format(time.time()),"wb+") as picklefile:
        print("THIS MANY ALLSTARS: {}".format(len(restaurants)))
        #this needs to pickle
        pickle.dump(restaurants,picklefile)
    print("finished.")
        





    #pngs = {}
    # for r in restaurants:
    #     # if not r.png in pngs:
    #     #     pngs[r.png] = 1
    #     #     r.png.show()
    #     # else:
    #     #     pngs[r.png] += 1
    #     if r[-1] > -1e6:
    #         r[0].png.show()
    # time.sleep(30)
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

WIDTH = 15
HEIGHT = 10

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", nargs='?', default='empty')
    parser.add_argument("--episodes", nargs='?', default='empty')
    parser.add_argument("--epsdecay", nargs='?', default='empty')
    parser.add_argument("--preference", nargs='?', default='empty')
    parser.add_argument("--savepath", nargs='?', default='empty')
    parser.add_argument("--init", nargs='?', default='empty')
    parser.add_argument("--fixed_target", nargs='?', default='empty')
    parser.add_argument("--eps", nargs='?', default='empty')
    parser.add_argument("--discount", nargs='?', default='empty')
    args = parser.parse_args()
    if args.saved == 'empty':
        saved_model = None
        saved_weights = None
    else:
        base = args.saved
        print("Using save name {}".format(base))
        saved_model = base + ".yaml"
        saved_weights = base + ".h5"

    if args.savepath == 'empty':
        savepath = './'
    else:
        savepath = args.savepath

    if args.episodes == 'empty':
        num_episodes = 30
    else:
        num_episodes = int(args.episodes)

    if args.epsdecay == 'empty':
        eps_decay = 0.999
    else:
        eps_decay = float(args.epsdecay)

    if args.preference == 'empty':
        preferences = [0.5,0.7,0.05,0.5,0.3,0.3,0.4]
    elif args.preference == 'revenue':
        preferences = [1,0,0,0,0,0,0]
    elif args.preference == 'profit':
        preferences = [0,1,0,0,0,0,0]
    elif args.preference == 'avg_noise':
        preferences = [0,0,1,0,0,0,0]
    elif args.preference == 'daily_customers':
        preferences = [0,0,0,1,0,0,0]
    elif args.preference == 'service_rating':
        preferences = [0,0,0,0,1,0,0]
    elif args.preference == 'avg_check':
        preferences = [0,0,0,0,0,1,0]
    elif args.preference == 'satisfaction':
        preferences = [0,0,0,0,0,0,1]
    else:
        preferences = [0.5,0.7,0.05,0.5,0.3,0.3,0.4]

    if args.init == 'empty':
        init_state = None
    else:
        init_state = args.init

    if args.fixed_target in ('true', 'True', 't'):
        print("Using Fixed Target Network. This fixes the target network over this entire batch of episodes, run again using the generated saved h5 weights file to 'update' Q.")
        fixed_target = True
    else:
        fixed_target = False
    
    if args.discount == 'empty':
        discount_rate = 0.9
    else:
        discount_rate = float(args.discount)

    if args.eps == 'empty':
        eps = 1.0
    else:
        eps = float(args.eps)

    outcomes = SKOutcomes()
    pd = PreferenceDummy(outcomes,preferences)
    bo = SKBayesOpt(pd)
    env = Environment(width = WIDTH, height = HEIGHT, tables = [], equipment = [], staff = [], reward_model = bo)
    
    qa = QAgent(input_shape = (WIDTH,HEIGHT),saved_model=saved_model, saved_weights=saved_weights, eps = eps, eps_decay = eps_decay, discount = discount_rate, fixed_target=fixed_target)
 
    #qa = QAgent(input_shape = (65,34))
    episodes = num_episodes
    base_iterations_per_ep = 2 
    max_iterations_per_ep = 20
    print("RESETTING")
    state = env.reset(init_state = init_state) # implement this!!!
    print("STARTING")
    for e in range(episodes):
        stats = []
        simulations = []
        restaurants = []
        episode_stats = {"mistakes":0,"nops":0,"actions":0,"rewards":[]}
        print("Episode {}".format(e))
        state = env.reset(init_state = init_state)
        #restaurants.append((state,None,0))
        num_iter = max_iterations_per_ep #min(base_iterations_per_ep * (e+1), max_iterations_per_ep)
        for i in range(num_iter): # scale up the number of episodes as we learn more hopefully
            q_vals = qa.predict_q(state.image)
            source_mask,target_mask = qa.get_mask(state.image)
            action = qa.get_action(q_vals[0],source_mask,target_mask)
            next_state, reward = env.step(action)
            sim_outcomes = env.sim_outcomes
            simulations.append((next_state.image,sim_outcomes,reward))
            print("\nREWARD:{}\n".format(reward))
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
        qa.save_model(savepath+"skdqn")
        with open(savepath+"skdnlog_{}.csv".format(time.time()),"w+") as logfile:
            logwriter = csv.writer(logfile,delimiter=',')
            logwriter.writerow(["mean_reward","max_reward","mistakes","nops","actions"])
            for e in stats:
                max_reward = max(e["rewards"])
                mean_reward = np.mean(e["rewards"])
                logwriter.writerow([mean_reward,max_reward,e["mistakes"],e["nops"],e["actions"]])
        with open(savepath+"allstars_{}.pkl".format(time.time()),"wb+") as picklefile:
            print("THIS MANY ALLSTARS: {}".format(len(restaurants)))
            #this needs to pickle
            pickle.dump(restaurants,picklefile)
        with open(savepath+"simulations_{}.pkl".format(time.time()),"wb+") as picklefile:
            print("This many simulations: {}".format(len(simulations)))
            #this needs to pickle
            pickle.dump(simulations,picklefile)
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
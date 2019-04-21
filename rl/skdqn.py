import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from rl.environment import Environment
from rl.qagent import QAgent
from bayesopt.bo import SKOutcomes, PreferenceDummy, SKBayesOpt
import time

if __name__ == "__main__":
    outcomes = SKOutcomes()
    pd = PreferenceDummy(outcomes)
    bo = SKBayesOpt(pd)
    env = Environment(width = 65, height = 34, tables = [], equipment = [], staff = [], reward_model = bo)
    #qa = QAgent(input_shape = (65,34),saved_model="skdqn.yaml", saved_weights="skdqn.h5")
    qa = QAgent(input_shape = (65,34))
    episodes = 10
    base_iterations_per_ep = 2 
    max_iterations_per_ep = 100
    restaurants = []
    state = env.reset(init_state = None) # implement this!!!
    for e in range(episodes):
        print("Episode {}".format(e))
        state = env.reset(init_state = None)
        restaurants.append((state,None,0))
        num_iter = min(base_iterations_per_ep * (e+1), max_iterations_per_ep)
        for i in range(num_iter): # scale up the number of episodes as we learn more hopefully
            q_vals = qa.predict_q(state.image)
            action = qa.get_action(q_vals[0])
            next_state, reward = env.step(action)
            restaurants.append((next_state,action,reward))
            qa.remember(state.image, action, reward, next_state.image)
        next_state.png.show()
        qa.retrain(num_iter)
    qa.save_model("skdqn")




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
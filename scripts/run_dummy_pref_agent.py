import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

from sknightmare.sknightmare.restaurant import Restaurant
from bayesopt.bo import SKBayesOpt, PreferenceDummy, SKOutcomes
from util.image_processing import ImageWriter
from rl.random_search import RandomSearch


if __name__=="__main__":
    outcomes = SKOutcomes()
    pd = PreferenceDummy(outcomes)
    print("Preference Bounds: {}".format(pd.get_outcome_bounds()))
    print("Sanity Check!\n\tPreferences: {}\n\tOutcomes: {}\n\tRating: {}".format(["{:.2f}".format(p) for p in outcomes.get_preferences()], ["{:.2f}".format(o) for o in outcomes.get_outcomes()], pd.rate(outcomes.get_outcomes())))
    init_points = [(outcomes.get_outcomes(),pd.rate(outcomes.get_outcomes()))]
    bo = SKBayesOpt(pd,init_points=init_points)

    qagent = RandomSearch()
    counter = 0
    while counter < 1000:
        prev_state = qagent.get_state()
        action = qagent.get_action()
        cur_state = qagent.act(action) # updates the current state and gets a json
        if q_agent.check_legality(prev_state, cur_state) == False:
            reward = -1e8
            qagent.update(action,reward)
        else:
            r = Restaurant("Sophie's Kitchen", state["equipment"], state["tables"], state["staff"], verbose=False)
            r.simulate(days=14)
            outcomes = r.ledger.generate_final_report()
            reward = bo.get_reward(outcomes,)
            qagent.update(action,reward)
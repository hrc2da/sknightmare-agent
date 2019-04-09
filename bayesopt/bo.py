from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
import numpy as np
import time


class SKOutcomes:
    def __init__(self):
        self.outcomes = {
            'revenue': {'bounds':[-1e9,1e9], 'val': 0.0},
            'upfront_cost': {'bounds':[-1e6,1e6], 'val': 0.0},
            'rating': {'bounds':[0.0,1.0], 'val': 0.5},
            'noisiness': {'bounds': [0,1e6], 'val': 0},
            'busyness': {'bounds': [0,1e5], 'val': 0},
            'service': {'bounds': [0.0,1.0], 'val':0.5},
            'price_range': {'bounds': [0.0,1.0], 'val':0.5}
        }
        self.preferences = np.random.rand(len(self.outcomes))
    def get_bounds(self):
        return np.array([self.outcomes[o]['bounds'] for o in self.outcomes])

    def get_outcomes(self):
        return np.array([self.outcomes[o]['val'] for o in self.outcomes])
    def get_preferences(self):
        return self.preferences

class PreferenceDummy:
    '''
        in the absence of a user, supply rewards based on a preference profile
    '''
    def __init__(self,outcomes_schema):
        '''
            preferences should be an np array of preference weights
        '''
        self.preferences = outcomes_schema.get_preferences()
        self.bounds = outcomes_schema.get_bounds()
    
    def get_outcome_bounds(self):
        return self.bounds

    def rotate_preferences(self,n_shift):
        self.preferences = np.roll(self.preferences,n_shift)

    def rate(self,design):
        #TODO: normalize based on bounds
        return np.dot(self.preferences, design)   

class SKBayesOpt:
    def __init__(self,evaluator,init_points = []):
        '''
            evaluator should have a rate function. Considering building an abstract base class
            for this (e.g. PreferenceDummy would extend it)
        '''
        self.evaluator = evaluator
        self.interrupted = False
        self.optimizer = Optimizer(dimensions = self.evaluator.get_outcome_bounds(),base_estimator="GP", n_initial_points=len(init_points), acq_func="gp_hedge", random_state=int(time.time()))

        # fit GP on initial data if provided
        if(len(init_points) > 0):
            x,y = init_points[0]
            print(x,y)
            for x,y in init_points[:-1]:
                self.optimizer.tell(list(x),y,fit=False)
            # only fit once we've loaded all the data
            x,y = init_points[-1]
            self.optimizer.tell(list(x),y,fit=True)
        print(self.optimizer.models[0].get_params())
    
    def get_reward(self, outcomes):
        # check if the outcomes surpass the AF
        if acquisition_check() :
            reward = self.evaluator.rate(outcomes)
            self.optimizer.tell(outcomes, reward, fit=True)
        else:
            reward = self.optimizer.base_estimator_.predict(outcomes)
        return reward
        
    def acquisition_check(self, outcomes):
        return true
    def objective(self,design):
        # ask the user for a rating of a design
        # for the PreferenceDummy, return immediately. For a real human, this would block and return when they answer
        return self.evaluator.rate(design)


    def run_agent(self):
        
        while not self.interrupted:
            # while true, wait for a sample from a_f;
            time.sleep(5)
            # ask if we want to 
        # r_dim = Real(-1, 1)
        # i_dim = Integer(0, 1)
        # bounds = [i_dim, i_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim,
        #         r_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim, r_dim]
        # res = gp_minimize(reward, bounds, acq_func="EI", n_random_starts=10, n_calls=50, random_state=int(time.time()))
        # explored_points = res.x_iters
        # r_json = {"restaurants": [construct_restaurant_json(x) for x in explored_points]}
        # with open('result.json', 'w') as fp:
        #     json.dump(r_json, fp)
        # print(res)
        # plot_convergence(res)
        # plt.show()
from keras.models import Sequential, Model, model_from_yaml
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import keras.backend as K
from collections import namedtuple
import numpy as np 
import random

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from util.image_processing import ImageWriter
#from util.generate_restaurants import RestaurantGenerator


Action = namedtuple('Action', 'action x y')
class QAgent:
    '''
        borrows a lot from https://keon.io/deep-q-learning/
    '''
    actions = [
        'source_x',
        'source_y',
        'target_x',
        'target_y'
        #etc
    ]
    def __init__(self,input_shape,saved_model=None,saved_weights=None):
        self.eps = 1.0
        self.eps_decay = 0.999
        self.eps_min = 0.01
        self.discount_rate = 0.9
        self.width,self.height = input_shape
        self.actions = np.zeros([2,self.width,self.height])
        self.replay_buffer = []
        self.lossfn = 'mse' # for now
        self.model = self.setup_q_network(input_shape,saved_model,saved_weights)
        # self.state = RestaurantGenerator(input_shape[0], input_shape[1])
        # self.state.load_dicts(tables, equipment, staff)
 
    
    def setup_q_network(self,input_shape,saved_model,saved_weights):
        if saved_model != None:
            yaml_file = open(saved_model, 'r')
            model_yaml = yaml_file.read()
            yaml_file.close()
            model = model_from_yaml(model_yaml)
            model.compile(loss = self.lossfn, optimizer = Adam(lr = 0.001))
        else:
            input_shape = (input_shape[0],input_shape[1],1)
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(4, 4), strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape))
            model.add(Conv2D(64, (5, 5), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(1000, activation='relu'))
            model.add(Dense(2*self.width*self.height, activation='linear'))
            model.compile(loss = self.lossfn, optimizer = Adam(lr = 0.001))
        if saved_weights != None:
            model.load_weights(saved_weights)
        return model
    
    def save_model(self,filename):
        with open(filename+".yaml", 'w') as yaml_file:
            yaml_file.write(self.model.to_yaml())
        self.model.save_weights(filename+".h5")

    def get_action(self,q_vals,source_mask,target_mask):
        dim = self.width*self.height
        source_field = source_mask * q_vals[0:dim] # this is assuming q-vals stay positive (if the mask is 1 for valid and 0 for not)
        # an alternative would be to have the "0" mask values be some huge negative number that we just subtract
        target_field = target_mask * q_vals[dim:]
        source_loc = np.argmax(source_field)
        target_loc = np.argmax(target_mask) + dim
        return source_loc, target_loc

    def predict_q(self, state):
        state = state.reshape((1,self.width,self.height,-1))
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        if np.random.random() < self.eps:
            return np.random.rand(1,2*self.width*self.height)
        else:
            return self.model.predict(state)
    
    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def get_mask(self, state):
        # returns a "valid" mask for a given state
        # note that this ONLY WORKS because input state is the SAME dimension as the q-vals (when divided into src q-vals and tgt q-vals)!!!!!!!
        staging_threshold = int(np.floor(self.width * 0.8))
        source_mask = state != 0 # or state is in staging ) #
        source_mask[:,staging_threshold:] = True

        add_mask = state == 0 # or state is in staging is false ) #
        source_mask[:,staging_threshold:] = False

        move_mask = state == 0 #can move anywhere that's not blocked -- staging is remove

        return source_mask.flatten(), move_mask.flatten()

        

    def retrain(self, batch_size):
        minibatch = random.sample(self.replay_buffer,batch_size) #if you make self.replay_memory an np array, you could do np.random.choice(self.replay_buffer,batch_size)
        for state, action, reward, next_state in minibatch:
            # start by getting the max q_val from the next state
            # note that because we have src/target and have collapsed from w*h*w*h to 2*w*h
            # we assumed that src/target are independent, which may not be true (it isn't)
            # so we update both of the src/target vals for the action here using a single reward
            next_state_q_vals = self.model.predict(next_state.reshape((1,self.width,self.height,-1)))[0]
            s_mask, t_mask = self.get_mask(next_state)
            best_next_action = self.get_action(next_state_q_vals,s_mask,t_mask)
            a_src, a_tgt = best_next_action

            max_next_q_vals = np.array((next_state_q_vals[a_src],next_state_q_vals[a_tgt]))
            # get the actual reward plus discounted future q_val
            target_q_src,target_q_tgt = reward + self.discount_rate * max_next_q_vals
            # assign it to the vector predicted q_vals from this state so we only update for the given action
            target_f = self.model.predict(state.reshape((1,self.width,self.height,-1)))
            #update the target for illegal cells
            s_mask, t_mask = self.get_mask(state)
            target_f[0] *= np.concatenate([s_mask,t_mask])
            target_f[0][a_src] = target_q_src
            target_f[0][a_tgt] = target_q_tgt
            # fit the action src/target to the "real" values
            self.model.fit(state.reshape((1,self.width,self.height,-1)), target_f, epochs=1, verbose=0)
            

            
        
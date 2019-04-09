from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
import keras.backend as K
from collections import namedtuple
import numpy as np 

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from util.image_processing import ImageWriter
from 

Action = namedtuple('Action', 'action x y')
class QAgent:
    actions = [
        'source_x',
        'source_y',
        'target_x',
        'target_y'
        #etc
    ]
    def __init__(self,input_shape,tables,equipment,staff):
        self.eps = 1.0
        self.eps_decay = 0.9
        
        self.action_space_dims = len(actions) * 2
        self.model = self.setup_q_network(input_shape)
        self.state = ImageWriter(input_shape[0], input_shape[1],"util","tables.json", "items.json")
        self.state.load_dicts(tables, equipment, staff)
    
    def setup_q_net(self,input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.actions_space_dims, activation='sigmoid'))
        model.compile(loss = self.lossfn, optimizer = Adam(lr = 0.001))
        return model
    
    def predict(self, state):
        self.eps *= self.eps_decay
        if np.random.random() < self.eps:
            return np.random.rand(self.action_space_dims)
        else:
            self.model.predict(state)
    
    def lossfn(true, predicted):
        return 1.0



    def update_state(self,source_x,source_y,target_x,target_y):
        '''
            network sees 0-1 as 0-0.8 in image space
            so we scale up by 1.2 to get back into image space
        '''
        source_x *= 1.2
        target_x *= 1.2
        if source_x > 1 and target_x > 1:
            # return no move (penalize this)
            return self.state
        elif source_x > 1:
            # this is an add

        elif target_x > 1:
            # this is a remove

        else:
            # it's a straight move
            # identify the item we are moving
            
            # update the item in the state
            return 
        

    def update_state(self,action):    
        if action["type"] == "add":

        elif action["type"] == "remove":
        
        
        

import numpy as np 
from collections import namedtuple()


class Environment:
    '''
        Environment contains and maintains:
            (1) State
            (2) Action Space 
            (3) Update/Step
            (4) Different Reward Functions
    '''
    Add = namedtuple('Add', ['catalog_id', 'x', 'y']) # catalog_id is the index in the catalog x and y are the target loc
    Move = namedtuple('Move', ['layout_id', 'x', 'y'])
    Remove = namedtuple('Remove', ['layout_id'])
    
    def __init__(self,width,height,tables,equipment,staff,table_fp="util/tables.json",eq_fp="util/items.json"):
        self.width = width
        self.height = height
        self.action_dims = 4
        self.tables = tables
        self.equipment = equipment
        self.staff = staff

    def update_state(action,):

    def network_space2image_space(x, y):
        return (x * 1.25, y * 1.25)
    
    def image_space2network_space(x, y):
        return (x * 0.8, y * 0.8)

    # randomly sample the action space for a set of actions
    def sample_action_space():
        return np.random.rand(4)

import numpy as np 
from collections import namedtuple()
import json
from util.image_processing import ImageWriter

class Environment:
    '''
        Environment contains and maintains:
            (1) State
            (2) Action Space 
            (3) Update/Step
            (4) Different Reward Functions
    '''
    Add = namedtuple('Add', ['catalog_id', 'x', 'y']) # catalog_id is the name in the catalog x and y are the target loc
    Move = namedtuple('Move', ['layout_id', 'x', 'y'])
    Remove = namedtuple('Remove', ['layout_id'])
    RestaurantItem = namedtuple('RestaurantItem', ['type','item'])

    def __init__(self,width,height,tables,equipment,staff,table_fp="util/tables.json",eq_fp="util/items.json"):
        self.width = width
        self.height = height
        self.action_dims = 4
        self.image_writer = ImageWriter(width,height)
        self.initialize_state(tables,equipment,staff)
        self.populate_catalog()

    def initialize_state(self,tables,equipment,staff):
        self.restaurant_layout = []
        for table in tables:
            self.restaurant_layout.append(self.RestaurantItem("table",table))
        for eq in equipment:
            self.restaurant_layout.append(self.RestaurantItem("equipment",eq))
        for s in staff:
            self.restaurant_layout.append(self.RestaurantItem("staff",s))
        self.image_writer.load_dicts(tables,equipment,staff)
        self.restaurant_image = self.image_writer.get_nparr()

    def get_tables(self):
        return [layout_item.item for layout_item in self.restaurant_layout if layout_item.type = "table"]
    
    def get_equipment(self):
        return [layout_item.item for layout_item in self.restaurant_layout if layout_item.type = "equipment"]
    
    def get_staff(self):
        return [layout_item.item for layout_item in self.restaurant_layout if layout_item.type = "staff"]

    def populate_catalog(self,table_fp,eq_fp):
        self.catalog = {}
        self.table_names = []
        self.eq_names = []
        with open(table_fp, 'r+') as table_file:
            tables = json.load(table_file)
            for table in tables.values():
                self.table_names.append(table["name"])
                self.catalog[table["name"]] = self.RestaurantItem("table",table)
        with open(eq_fp, 'r+') as eq_file:
            equipment = json.load(eq_file)
            for eq in equipment.values():
                self.eq_names.append(eq["name"])
                self.catalog[eq["name"]] = self.RestaurantItem("equipment",equipment)
        self.catalog["staff"] = self.RestaurantItem("staff",{"x":-1, "y":-1})

    def update_state(action):
        # action is of type Add, Move, or Remove
        if type(action) == self.Add:
            # get the item to add 
        elif type(action) == self.Move:

        else:


    def network_space2image_space(x, y):
        return (x * 1.25, y * 1.25)
    
    def image_space2network_space(x, y):
        return (x * 0.8, y * 0.8)

    # randomly sample the action space for a set of actions
    def sample_action_space():
        return np.random.rand(4)
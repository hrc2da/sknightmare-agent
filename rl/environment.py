
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
    Move = namedtuple('Move', ['layout_index', 'x', 'y']) #layout_index is the index of the item in the restaurant layout
    Remove = namedtuple('Remove', ['layout_index'])
    Nop = namedtuple('Nop',['reason','x','y']) #the source x and y
    RestaurantItem = namedtuple('RestaurantItem', ['type','item'])

    def __init__(self, width, height, tables, equipment, staff, table_fp="util/tables.json", eq_fp="util/items.json"):
        self.width = width
        self.height = height
        self.action_dims = 4
        self.image_writer = ImageWriter(width,height)
        self.initialize_state(tables,equipment,staff)
        self.populate_catalog()

    def initialize_state(self, tables, equipment, staff):
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

    def populate_catalog(self, table_fp, eq_fp):
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
        self.divisions = {}
        num_items = len(self.catalog)
        self.increment = 1 / num_items
        index = 0
        for key, item in self.catalog.items():
            self.divisions[index] = key
            index += self.increment
    
    def get_item_from_source_y(self, y):
        # assume y is within [0,1]
        index = int(source_y / self.increment)
        return self.catalog[self.divisions[index]]

    def set_item_location(self, item, x, y):
        item["x"] = x
        item["y"] = y
        item["attributes"]["x"] = x
        item["attributes"]["y"] = y
        return item

    def update_state(self, action):
        # action is of type Add, Move, or Remove
        if type(action) == self.Add:
            item_to_add = self.catalog[action.catolog_id]
            item_to_add.item = self.set_item_location(item_to_add.item,action.x,action.y)
            self.restaurant_layout.append(item_to_add)    
        
        elif type(action) == self.Remove:
            self.restaurant_layout.pop(action.layout_index)

        else:
            # action is of type move
            item_to_move = self.restaurant_layout[action.layout_index]
            item_to_move.item = self.set_item_location(item_to_move.item,action.x,action.y)
    
    def update_image(self):
        self.image_writer.reset_image()
        self.image__writer.load_dicts(self.get_tables(),self.get_equipment(),self.get_staff())


    def network_space2image_space(self, x, y):
        return (x * 1.25, y * 1.25)
    
    def image_space2network_space(self, x, y):
        return (x * 0.8, y * 0.8)

    # randomly sample the action space for a set of actions
    def sample_action_space(self):
        return np.random.rand(4)

    def action2move(self, source_x, source_y, target_x, target_y):
        # source_x and target_x are already rescaled
        if source_x > 1 and target_x > 1:
            # return no move (penalize this)
            return Nop('move within staging',source_x,source_y)
        elif source_x > 1:
            # this is an add
            index,item = self.find_in_catalog(source_x, source_y)
            return Add(item.item["name"],target_x,target_y) # why don't we just pass the actual item????

        elif target_x > 1:
            # this is a remove
            index,item = self.find_in_layout(source_x, source_y)
            return Remove(index)

        else:
            # it's a straight move
            index,item = self.find_in_layout(source_x, source_y)
            return Move(index,target_x,target_y)
    
    def find_in_catalog(self,x,y):
        index = None # why do we need index
        item = self.get_item_from_source_y(y)
        return index,item
    
    def find_in_layout(self,x,y):
        index = None
        item = None
        return index,item
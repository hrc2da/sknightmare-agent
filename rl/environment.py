import numpy as np 
import math
from collections import namedtuple
from recordclass import recordclass
from copy import copy, deepcopy
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../sknightmare') # this is so dumb. why do I have to do this
from util.image_processing import ImageWriter
from sknightmare.restaurant import Restaurant

class Environment:
    '''
        Environment contains and maintains:
            (1) State
            (2) Action Space 
            (3) Update/Step
            (4) Different Reward Functions
    '''
    Add = recordclass('Add', ['catalog_id', 'x', 'y']) # catalog_id is the name in the catalog x and y are the target loc
    Move = recordclass('Move', ['layout_index', 'x', 'y']) #layout_index is the index of the item in the restaurant layout
    Remove = recordclass('Remove', ['layout_index'])
    Nop = recordclass('Nop',['reason','x','y']) #the source x and y
    RestaurantItem = recordclass('RestaurantItem', ['type','item'])
    State = recordclass('State', ['layout','image', 'png'])
    mistake_threshold = 0
    SIZE_SCALAR = 20 # this is just to adjust for the fact that our items are relatively too big for our layout size
    name_delimiter = "___"

    def __init__(self, width, height, tables, equipment, staff, reward_model, table_fp="util/tables.json", eq_fp="util/items.json"):
        self.width = width
        self.height = height
        self.action_dims = 4
        self.reward_model = reward_model
        self.image_writer = ImageWriter(width,height,"util","tables.json", "items.json")
        
        self.populate_catalog("util/tables.json", "util/items.json")
        self.initialize_state(tables,equipment,staff)
        self.sim_outcomes = None
    
    def generate_random_state(self):
        # design choice: there can be a random start with 0 tables or 0 eq. it may be important for the agent
        # to learn that if this is the case, it should add the missing table or eq. I'll try this for now but if too
        # hard, we can fix to have min 1 of each table, eq, staff
        gridw = 5
        gridh = 5
        num_tables = 1 + np.random.randint(4)
        num_eq = 1 + np.random.randint(4)
        num_staff = 1 + np.random.randint(4)
        tables = []
        eq = []
        staff = []
        assert num_tables + num_eq + num_staff <= gridw * gridh
        #generate an 10x10 grid for locations
        #for each element of the grid, generate a random number
        #then take the max N grid spots as locations and assign them (pop from end of ascending list)
        location_lottery = np.random.random((gridw,gridh))
        locations = list(map(lambda x: np.unravel_index(x,location_lottery.shape), np.argsort(location_lottery.flatten())))
        print(locations)
        for t in range(num_tables):
            name = self.table_names[np.random.randint(len(self.table_names))]
            new_table = deepcopy(self.catalog[name])
            locx, locy = locations.pop()
            print(locx,locy)
            x = (1.0/gridw)*(np.random.random() + locx)
            y = (1.0/gridh)*(np.random.random() + locy)
            new_table.item = self.set_item_location(new_table.item,x,y,downscale=False)
            #new_table.item = self.set_name(new_table.item,new_table.type)
            tables.append(new_table.item)
        for e in range(num_eq):
            name = self.eq_names[np.random.randint(len(self.eq_names))]
            new_eq = deepcopy(self.catalog[name])
            locx, locy = locations.pop()
            x = (1.0/gridw)*(np.random.random() + locx)
            y = (1.0/gridh)*(np.random.random() + locy)
            new_eq.item = self.set_item_location(new_eq.item,x,y,downscale=False)
            #new_eq.item = self.set_name(new_eq.item,new_eq.type)
            eq.append(new_eq.item)
        for s in range(num_staff):
            new_staff = deepcopy(self.catalog["staff"])
            locx, locy = locations.pop()
            x = (1.0/gridw)*(np.random.random() + locx)
            y = (1.0/gridh)*(np.random.random() + locy)
            new_staff.item = self.set_item_location(new_staff.item,x,y,downscale=False)
            #new_staff.item = self.set_name(new_staff.item,new_staff.type)
            staff.append(new_staff.item)
        return tables,eq,staff
    def reset(self, init_state = None):
        if init_state == "random":

            tables,eq,staff = self.generate_random_state()
            self.initialize_state(tables,eq,staff)
            self.update_image()
            print([(r.item['name'],r.item['x'],r.item['y']) for r in self.restaurant_layout])
            self.count_collisions(self.restaurant_layout)
            return self.get_state()

        elif not init_state:
            
            self.initialize_state(tables = [{
                                    "type": "image",
                                    "size": 30,
                                    "radius": 30,
                                    "x":0.7,
                                    "y":0.7,
                                    "seats": 4,
                                    "name": "Med Round Table",
                                    "svg_path": "svgs/4_table_round.svg",
                                    "cost": 600,
                                    "daily_upkeep": 1,
                                    "noisiness": 1,
                                    "privacy": 1,
                                    "attributes":{
                                        "size": 30,
                                        "radius": 30,
                                        "x":0.7,
                                        "y":0.7,
                                        "seats": 4,
                                        "name": "Med Round Table",
                                        "svg_path": "svgs/4_table_round.svg",
                                        "cost": 600,
                                        "daily_upkeep": 1,
                                        "noisiness": 1,
                                        "privacy": 1,
                                    },
                                    "appliances": []
                                }],
                                equipment = [{
                                        "name": "Basic Oven",
                                        "type": "image",
                                        "size": 25,
                                        "path": "svgs/oven.svg",
                                        "x": 0.2,
                                        "y": 0.2,
                                        "attributes": {
                                            "category": "cooking",
                                            "capabilities": {
                                                "pizza": {
                                                "quality_mean": 0.5,
                                                "quality_std": 0.4,
                                                "cook_time_mult": 2,
                                                "cook_time_std": 0.1
                                                }
                                            },
                                            "path": "svgs/oven.svg",
                                            "difficulty_rating": 0.2,
                                            "cost": 300,
                                            "daily_upkeep": 5,
                                            "reliability": 0.2,
                                            "noisiness": 0.1,
                                            "atmosphere": 0.1,
                                            "x": 0.2,
                                            "y": 0.2,
                                            "radius": 25
                                        }
                                }],
                                staff = [{"name": "staff","x":0.5,"y":0.5, "attributes":{"x":0.5,"y":0.5}}])
        self.update_image()
        
        return self.get_state()

    def set_name(self,obj,obj_type):
        if obj_type == "table":
            uid = self.table_uid
            self.table_uid += 1
        elif obj_type == "equipment":
            uid = self.eq_uid
            self.eq_uid += 1
        else:
            uid = self.staff_uid
            self.staff_uid += 1    
        if "name" not in obj:
            obj["name"] = obj_type
        obj["name"] += self.name_delimiter+str(uid)
        return obj

    def initialize_state(self, tables, equipment, staff):
        self.table_uid = 0
        self.eq_uid = 0
        self.staff_uid = 0
        self.restaurant_layout = []
        for table in tables:
            if "attributes" not in table:
                #table["attributes"] = {"x":table["x"], "y":table["y"]}
                table["attributes"] = {}
                for attr in table:
                    table["attributes"][attr] = table[attr]
            table = self.set_name(table,"table")
            self.restaurant_layout.append(self.RestaurantItem("table",table))
        for eq in equipment:
        #     eq["x"] = -1
        #     eq["y"] = -1
            eq = self.set_name(eq,"equipment")
            self.restaurant_layout.append(self.RestaurantItem("equipment",eq))
        for s in staff:
            s["name"] = "staff"
            s = self.set_name(s,"staff")
            self.restaurant_layout.append(self.RestaurantItem("staff",s))
        self.image_writer.load_dicts(tables,equipment,staff)
        self.restaurant_image = self.image_writer.get_nparr()

    def get_tables(self,layout = None):
        if layout is not None:
            return [layout_item.item for layout_item in layout if layout_item.type == "table"]
        else:
            return [layout_item.item for layout_item in self.restaurant_layout if layout_item.type == "table"]
    
    def get_equipment(self,layout = None):
        if layout is not None:
            return [layout_item.item for layout_item in layout if layout_item.type == "equipment"]
        else:
            return [layout_item.item for layout_item in self.restaurant_layout if layout_item.type == "equipment"]
    
    def get_staff(self, layout = None):
        if layout is not None:
            return [layout_item.item for layout_item in layout if layout_item.type == "equipment"]
        else:
            return [layout_item.item for layout_item in self.restaurant_layout if layout_item.type == "staff"]

    def populate_catalog(self, table_fp, eq_fp):
        self.catalog = {}
        self.table_names = []
        self.eq_names = []
        with open(table_fp, 'r+') as table_file:
            tables = json.load(table_file)
            for table in tables.values():
                table["x"] = -1
                table["y"] = -1
                if "attributes" not in table:
                    table["attributes"] = {}
                    for attr in table:
                        table["attributes"][attr] = table[attr]
                self.table_names.append(table["name"])
                self.catalog[table["name"]] = self.RestaurantItem("table",table)
        with open(eq_fp, 'r+') as eq_file:
            equipment = json.load(eq_file)
            for eq in equipment.values():
                eq["x"] = -1
                eq["y"] = -1
                self.eq_names.append(eq["name"])
                self.catalog[eq["name"]] = self.RestaurantItem("equipment",eq)
        self.catalog["staff"] = self.RestaurantItem("staff",{"name":"staff","x":-1, "y":-1, "attributes":{"x":-1,"y":-1}})

    def set_item_location(self, item, x, y, downscale=True):
        if downscale == True:
            x /= self.height
            y /= self.width

        item["x"] = x
        item["y"] = y
        item["attributes"]["x"] = x
        item["attributes"]["y"] = y
        return item

    def update_state(self, action):
        # action is of type Add, Move, or Remove
        if type(action) == self.Add:
            print("Adding")
            item_to_add = deepcopy(self.catalog[action.catalog_id]) #deep copy so we don't modify actual catalog
            item_to_add.item = self.set_item_location(item_to_add.item,action.x,action.y)
            item_to_add.item = self.set_name(item_to_add.item,item_to_add.type)
            print("Adding this: ",item_to_add)
            self.restaurant_layout.append(item_to_add)    
        
        elif type(action) == self.Remove:
            print("Removing")
            self.restaurant_layout.pop(action.layout_index)

        elif type(action) == self.Move:
            print("Moving")
            # action is of type move
            item_to_move = self.restaurant_layout[action.layout_index]
            item_to_move.item = self.set_item_location(item_to_move.item,action.x,action.y)
        else:
            print("Noping: {}".format(action.reason))
            pass
    
    def update_image(self):
        self.image_writer.reset_image()
        self.image_writer.load_dicts(self.get_tables(),self.get_equipment(),self.get_staff())
        self.restaurant_image = self.image_writer.get_nparr()


    def network_space2image_space(self, x, y):
        return (x * 1.25, y * 1.25)
    
    def image_space2network_space(self, x, y):
        return (x * 0.8, y * 0.8)

    # randomly sample the action space for a set of actions
    def sample_action_space(self):
        return np.random.rand(4)

    def act_vect2act_coords(self,source,target):
        #depending on if flatten goes by rows or columns, divide by width or height
        #the np flatten seems to go by rows, so I'll go by that.
        print("SOURCE IS ",source)
        # source_x = source % self.width #/ self.width #converting to 0 to 1
        # source_y = source // self.width #/ self.height # converting to 0 to 1
        # target = target-self.width*self.height
        # target_x = target % self.width #/ self.width
        # target_y = target // self.width #/ self.height

        #turns out that np arrays flip x and y since they do rows and columns
        source_y = source % self.width #/ self.width #converting to 0 to 1
        source_x = source // self.width #/ self.height # converting to 0 to 1
        target = target-self.width*self.height
        target_y = target % self.width #/ self.width
        target_x = target // self.width #/ self.height
        return source_x, source_y, target_x, target_y


    def action2move(self, source_x, source_y, target_x, target_y):
        # source_x and target_x are not rescaled
        staging_threshold = int(np.floor(0.8*self.width))
        if source_y >= staging_threshold and target_y >= staging_threshold:
            # return no move (penalize this)
            return self.Nop('move within staging',source_x,source_y)
        elif source_y >= staging_threshold:
            # this is an add
            index,item = self.find_in_catalog(source_x, source_y)
            return self.Add(item.item["name"],target_x,target_y) # why don't we just pass the actual item????

        elif target_y >= staging_threshold:
            # this is a remove
            index,item = self.find_in_layout(source_x, source_y)
            if index is None:
                print("Can't find ",source_x,source_y)
                return self.Nop('trying to remove nothing',source_x,source_y)
            return self.Remove(index)

        else:
            # it's a straight move
            index,item = self.find_in_layout(source_x, source_y)
            if index is None:
                print("Can't find ",source_x,source_y)
                return self.Nop('trying to move nothing',source_x,source_y)
            return self.Move(index,target_x,target_y)
    

    def find_in_catalog(self,x,y):
        # x /= self.width
        # backwards because of numpy
        y /= self.width
        print("indexing catalog for {}".format(y))
        index = math.floor(y * len(self.catalog))
        print("catalog index is {}".format(index))
        keys = list(self.catalog)
        keys.sort()
        print("catalog:",keys)
        key = keys[index]
        item = self.catalog[key]
        return index,item
    
    def find_in_layout(self,x,y,threshold=0.1):
        #this is backwards again b/c numpy
        x /= self.height
        y /= self.width

        closest_index = 0
        closest_dist = 10 # can never be this big
        closest_coords = []
        for i,obj in enumerate(self.restaurant_layout):
            o_x = obj.item["x"]
            o_y = obj.item["y"]
            dist = np.linalg.norm((x - o_x, y - o_y), 2)
            if dist < closest_dist:
                closest_index = i
                closest_dist = dist
                closest_coords = [o_x,o_y]
        if closest_dist < threshold:
            return closest_index, self.restaurant_layout[closest_index]
        else:
            print("dist too far: ",closest_dist, (x,y), closest_coords)
            return None,None

    def get_state(self):
        return self.State(self.restaurant_layout, self.restaurant_image, self.image_writer.get_image())

    def set_state(self, state):
        self.restaurant_layout = state.layout
        self.update_image()

    def get_reward(self, old_state, new_state, action):
        if type(action) == self.Nop:
            return 0
        # new_layout = new_state.layout
        # return len(new_layout)-self.count_collisions(new_layout) # not sure if I should account for the old state in reward or not
        num_mistakes = self.count_collisions(new_state.layout)
        if num_mistakes > self.mistake_threshold:
            print("Found {} mistakes in the restaurant: {}".format(num_mistakes,new_state.layout))
            return 0
        else:
            new_layout = new_state.layout
            try:
                r = Restaurant("Sophie's Kitchen", self.get_equipment(new_layout), self.get_tables(new_layout), self.get_staff(new_layout), verbose=False)
            except ValueError as e:
                print("Missing minumum of tables or equipment: ",e)
                return 0
            r.simulate(days=14)
            print("SIMULATING RESTAURANT!!!")
            self.sim_outcomes = r.ledger.generate_final_report()
            print(self.sim_outcomes)
            return self.reward_model.get_reward(self.sim_outcomes)



    def step(self,act_vect):
        s_x, s_y, t_x, t_y = self.act_vect2act_coords(*act_vect)
        move = self.action2move(s_x, s_y, t_x, t_y)
        old_state = deepcopy(self.get_state())
        self.update_state(move)
        self.update_image()
        new_state = self.get_state()
        reward = self.get_reward(old_state, new_state, move)
        if reward == 0: #this is odd, but let's assume 0 is a penalty
            # for any penalty, revert the state
            print("REVERTING!!!!!!!")
            self.set_state(old_state)
            new_state = old_state
        print("VALID SOURCES ",np.nonzero(new_state.image.flatten()))
        return new_state, reward



    def check_collision(self,x,y,size,blocking_item):
        size /= self.SIZE_SCALAR
        # check if item centered at x,y and size size overlaps with blocking_item
        try:
            # get x, y and size in pixels
            bi_x = blocking_item["attributes"]["x"]*self.width
            bi_y = blocking_item["attributes"]["y"]*self.height
            bi_size = blocking_item["size"] / self.SIZE_SCALAR # already in pixels
            bi_size = 0 #turning OFF collision check unless on top of each other for now
        except KeyError as e:
            # this is dumb, but I do want it to break if this happens
            raise(KeyError("{}\nTried to check collision with an item that doesn't have (x,y)".format(e)))

        bi_x_min = bi_x
        bi_x_max = bi_x + bi_size
        bi_y_min = bi_y
        bi_y_max = bi_y + bi_size

        x_min = x
        x_max = x+size
        y_min = y
        y_max = y+size

        # using Separating Axis Thm: https://gamedevelopment.tutsplus.com/tutorials/collision-detection-using-the-separating-axis-theorem--gamedev-169 (thanks Stack Overflow!)
        # if they do not overlap, one of the right sides of the bounding boxes will be to the left side of the other bounding box
        # AND one of the top sides of the bounding boxes will be under the bottom side of the other bounding box
        collision = False
        collision_x = False
        collision_y = False

        if not ( bi_x_max < x_min or x_max < bi_x_min ):
            # overlaps in x direction
            #print("OVERX")
            collision_x = True
        if not ( bi_y_max < y_min or y_max < bi_y_min ):
            # overlaps in y direction
            #print("OVERY")
            collision_y = True
        if collision_x == True and collision_y == True:
            print("*******************************************")
            print("COLLISION: ({},{}) and ({},{}), sizes = ({},{})".format(x,y,bi_x,bi_y,size,bi_size))
            print("*******************************************")
            collision = True
        return collision

    def count_collisions(self, restaurant_layout):
        #returns True if there is at least one collision, False if no collisions
        num_collisions = 0
        for i in range(len(restaurant_layout)):
            for j in range(i+1,len(restaurant_layout)):
                item = restaurant_layout[i]
                compare = restaurant_layout[j]
                if item.type == "staff" or compare.type == "staff":
                    continue
                else:
                    if self.check_collision(item.item['x']*self.width,item.item['y']*self.height,item.item['size'],compare.item) == True:
                        num_collisions += 1
                    else:
                        continue
        return num_collisions
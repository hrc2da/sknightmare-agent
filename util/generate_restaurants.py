import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')


import json
import random
import time
import numpy as np
from util.image_processing import ImageWriter
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

class RestaurantGenerator:
    size_scalar = 10
    def __init__(self,width=65,height=34,table_path="util/tables.json",equipment_path="util/items.json"):
        self.tables = None
        self.equipment = None
        self.width = width
        self.height = height
        self.restaurant_layout = [] # list of items in the restaurant
        with open (table_path, 'r') as table_file:
            self.tables = json.load(table_file)
            self.fix_tables()
        with open (equipment_path, 'r') as equipment_file:
            self.equipment = json.load(equipment_file)
        random.seed(time.time())

    def fix_tables(self):
        # adds an empty attributes field for all tables because our notation is all over the place
        for t in self.tables:
            self.tables[t]["attributes"] = {}

    def check_collision(self,x,y,size,blocking_item):
        size /= self.size_scalar
        # check if item centered at x,y and size size overlaps with blocking_item
        try:
            # get x, y and size in pixels
            bi_x = blocking_item["attributes"]["x"]*self.width
            bi_y = blocking_item["attributes"]["y"]*self.height
            bi_size = blocking_item["size"]/self.size_scalar # already in pixels
        except KeyError as e:
            # this is dumb, but I do want it to break if this happens
            raise(KeyError("{}\nTried to check collision with an item that doesn't have (x,y)".format(e)))

        bi_x_min = bi_x - size/2.0
        bi_x_max = bi_x + size/2.0
        bi_y_min = bi_y - size/2.0
        bi_y_max = bi_y + size/2.0

        x_min = x-size/2.0
        x_max = x+size/2.0
        y_min = y-size/2.0
        y_max = y+size/2.0

        # using Separating Axis Thm: https://gamedevelopment.tutsplus.com/tutorials/collision-detection-using-the-separating-axis-theorem--gamedev-169 (thanks Stack Overflow!)
        # if they do not overlap, one of the right sides of the bounding boxes will be to the left side of the other bounding box
        # AND one of the top sides of the bounding boxes will be under the bottom side of the other bounding box
        collision = False
        if not ( bi_x_max < x_min or x_max < bi_x_min ):
            # overlaps in x direction
            #print("OVERX")
            collision = True
        if not ( bi_y_max < y_min or y_max < bi_y_min ):
            # overlaps in y direction
            #print("OVERY")
            collision = True
        return collision

    def check_collisions_restaurant(self):
        #returns True if there is at least one collision, False if no collisions
        for i in range(len(self.restaurant_layout)):
            for j in range(i,len(self.restaurant_layout)):
                item = self.restaurant_layout[i]
                compare = self.restaurant_layout[j]
                if item[0] == "staff" or compare[0] == "staff":
                    continue
                else:
                    if self.check_collision(item[1]['x']*self.width,item[1]['y']*self.height,item[1]['size'],compare[1]) == True:
                        return True
                    else:
                        continue
        return False
    def place_item(self,item,item_type,blocking=True,x=None,y=None):
        placed = False
        if x==None and y==None:
            while(not placed):
                x = random.random()
                y = random.random()
                if blocking == False:
                    break
                x_pixels = x*self.width
                y_pixels = y*self.height
                placed = True    
                for curr_item in self.restaurant_layout:
                    if curr_item[0] == "staff":
                        continue
                    if self.check_collision(x_pixels,y_pixels,item["size"],curr_item[1]) == True:
                        placed = False
                        break
        if item_type=="staff":
            item["x"] = x
            item["y"] = y
        else:
            item["x"] = x
            item["y"] = y
            item["attributes"]["x"] = x
            item["attributes"]["y"] = y
        # now if this is a bar we also need to place appliances
        if item_type=="table":
            if len(item["appliances"]) > 0:
                for a in item["appliances"]:
                    self.place_item(a,"equipment",x=x,y=y)
        self.restaurant_layout.append((item_type,item))

    def generate_item(self,item_type):
        if item_type == "table":
            return random.choice(list(self.tables.values()))
        elif item_type == "equipment":
            return random.choice(list(self.equipment.values()))

    def generate_restaurant(self,max_items =10,max_waiters = 5):
        self.restaurant_layout = []
        p_table = 0.3
        p_equipment = 0.2
        # probability of nothing is the remainder
        for i in range(max_items):
            flip = random.random()
            if flip < p_table:
                self.place_item(self.generate_item("table"),"table",blocking=False)
            elif flip < (p_table + p_equipment):
                self.place_item(self.generate_item("equipment"),"equipment",blocking=False)
            # else do nothing

        p_waiter = 0.5
        for i in range(max_waiters):
            flip = random.random()
            if flip < p_waiter:
                self.place_item({},"staff",blocking=False)
        
        tables = [item[1] for item in self.restaurant_layout if item[0] == "table"]
        equipment = [item[1] for item in self.restaurant_layout if item[0] == "equipment"]
        staff = [item[1] for item in self.restaurant_layout if item[0] == "staff"]
        #return {"tables":tables, "equipment":equipment, "staff": staff}
        return tables, equipment, staff


if __name__=="__main__":
    g = RestaurantGenerator(width=65, height=34)
    data = []
    for i in range(10000):
        print(i)
        tables, equipment, staff = g.generate_restaurant()
        collisions = g.check_collisions_restaurant()
        valid = not collisions
        print("Tables: {}".format(["{}: ({},{})".format(t['name'],t['x'],t['y']) for t in tables]))
        print("Equipment: {}".format(["{}: ({},{})".format(e['name'],e['x'],e['y']) for e in equipment]))
        print("Staff: {}".format(staff))
        writer = ImageWriter(65, 34, "util","tables.json", "items.json")
        writer.load_dicts(tables, equipment, staff)
        # plt.imshow(writer.get_image())
        # plt.show()
        #writer.get_image().show()
        data.append((writer.get_nparr(),valid))

        #time.sleep(0)
    np.save("restaurants.npz",data)
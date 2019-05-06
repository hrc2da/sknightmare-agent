import svgwrite
import json
from cairosvg import svg2png
from scipy import misc
import cv2 as cv
import numpy as np
import time
import os
import glob
from PIL import Image
from io import BytesIO
import numpy as np
from skimage.draw import circle
import io
import cv2
from copy import deepcopy

class ImageWriter():
    def __init__(self, width, height, base_fp='.', tables_fn='tables.json', equipment_fn='items.json'):
        self.name_delimiter = "___"
        self.width = width
        self.height = height
        self.bp = base_fp
        self.objects = {"empty":0} # this dict defines a value for every object type that we can add (for the imgarr) 
        self.inverse_object_map = ["empty"] # this array gives the item name for an imagarr value
        self.load_maps(os.path.join(base_fp,tables_fn),os.path.join(base_fp,equipment_fn))
        print("OBJECTS!!!",self.objects)
        self.dwg = svgwrite.Drawing( size = (width, height))
        self.imgarr = np.zeros((height,width))

        #self.dwg.add(self.dwg.rect(insert = (0,0), size = (width, height), stroke = svgwrite.rgb(10, 10, 10), fill = svgwrite.rgb(255, 255, 255)))

    def reset_image(self):
        self.dwg = svgwrite.Drawing( size = (self.width, self.height))
        self.imgarr = np.zeros((self.height,self.width))

    def load_maps(self, tables_fp, equipment_fp):
        
        with open(equipment_fp, 'r+') as infile:
            data = json.load(infile)
            self.item_svg_map = {}
            keys = list(data)
            keys.sort()
            for key in keys:
                item = data[key]
                self.item_svg_map[item['name']] = item
                self.objects[item["name"]] = len(self.objects)
                self.inverse_object_map.append(item["name"])
        with open(tables_fp, 'r+') as infile:
            self.tables_svg_map = {}
            data= json.load(infile)
            keys = list(data)
            keys.sort()
            for key in keys:
                item = data[key]
                print(item["name"], len(self.objects))
                self.tables_svg_map[item['name']] = item
                self.objects[item["name"]] = len(self.objects)
                self.inverse_object_map.append(item["name"])
                #note that the bars will show up as either a table or an appliance; one will overwrite the other; it doesn't really matter I think for training
        self.objects["staff"] = len(self.objects)
        self.inverse_object_map.append("staff")
    
    

    def load_dicts(self, tables, equipment, staff):
        self.draw_staff(staff)
        self.draw_tables(tables)
        self.draw_equipment(equipment)
        self.svg_string = self.dwg.tostring()
        return
        
    def load_json(self, filepath):
        with open("test.json", "r+") as infile:
            data = json.load(infile)
        tables = data["tables"]
        equipment = data["equipment"]
        staff = data["staff"]

        self.draw_staff(staff)
        self.draw_tables(tables)
        self.draw_equipment(equipment)
        self.svg_string = self.dwg.tostring()
        return

    def get_image(self):
        if not self.svg_string:
            raise Exception("no image input")
        img = svg2png(bytes(self.svg_string, 'utf-8'))
        #nparr = np.frombuffer(img, np.uint8)
        #nparr = PIL.Image.frombytes("RGB",(self.width,self.height),img)
        # nparr = np.array(Image.open(BytesIO(img))
        pil_image = Image.open(BytesIO(img))
        # return cv.imdecode(nparr,0)
        return pil_image

    def np_arr2json(self, arr):
        layout = []
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                obj_id = int(arr[row,col])
                if obj_id >= len(self.inverse_object_map):
                    print("wtf",obj_id)
                if obj_id > 0:
                    name = self.inverse_object_map[obj_id]
                    if name in self.tables_svg_map:
                        print(self.tables_svg_map[name])
                        table = deepcopy(self.tables_svg_map[name])
                        table['x'] = 1.0*col/self.width
                        table['y'] = 1.0*row/self.height
                        table['path'] = table['svg_path']
                        layout.append(table)
                    elif name in self.item_svg_map:
                        print(self.item_svg_map[name])        
                        item = deepcopy(self.item_svg_map[name])
                        item['x'] = 1.0*col/self.width
                        item['y'] = 1.0*row/self.height
                        layout.append(item)
                    elif name == "staff":
                        print("staff")
                        x = 1.0*col/self.width
                        y = 1.0*row/self.height
                        layout.append({"name":"staff","x":x,"y":y,"path":"svgs/waiter.svg"})
        return layout

    def json_to_img(self,layout,width=640,height=480,icon_height=50):
        container = np.ones((height,width,3), dtype='uint8')*255 # why by 4? rgba?
        for item in layout:
            path = os.path.join(self.bp,item["path"])
            c_col,c_row = (int(item['x']*width), int(item['y']*height))
            if c_col > width or c_row > height:
                print("invalid position!!!!!!",item['x'],item['y'])
                continue
            img = self.svg2nparr(path,icon_height,icon_height)
            print("IMAGGGEE: ",path,img.shape)
            print(img)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            start_col = c_col - int(np.floor(icon_height/2))
            end_col = start_col + icon_height
            if start_col < 0:
                icon_start_col = - start_col
                start_col = 0
            else:
                icon_start_col = 0
            
            if end_col > width:
                print(end_col,c_col)
                icon_end_col = icon_height - (end_col-width)
                end_col = width  
            else:
                icon_end_col = icon_height
            
            start_row = c_row - int(np.floor(icon_height/2))
            end_row = start_row + icon_height
            if start_row < 0:
                icon_start_row = - start_row
                start_row = 0
            else:
                icon_start_row = 0
            
            if end_row > height:
                icon_end_row = icon_height - (end_row - height)
                end_row = height
            else:
                icon_end_row = icon_height
            
            #now overwrite the specific part of the np array
            print(start_row,end_row,start_col,end_col)
            print(icon_start_row,icon_end_row,icon_start_col,icon_end_col)

            container[start_row:end_row, start_col:end_col, :3] = img[icon_start_row:icon_end_row, icon_start_col:icon_end_col, :3]
            print("Image")
            print(img[icon_start_row:icon_end_row, icon_start_col:icon_end_col, :3])
            print("output")
            print(container[start_row:end_row, start_col:end_col, :3] )

        return container

    def svg2nparr(self, filename, width, height):
         # https://stackoverflow.com/questions/55440483/how-can-i-optimize-the-palette-image-size-with-pil/55442505#55442505
        mem = io.BytesIO()
        svg2png(url=filename, write_to=mem, parent_height=height, parent_width=width, output_height=height, output_width=width)
        img = np.array(Image.open(mem), np.uint8)
        img[img == 0] = 255
        rows,cols,depth = img.shape
        print(filename, img.shape)
        blank_icon = np.ones((height,width,depth))*255
        row_padding = int(np.floor((height-rows)/2))
        col_padding = int(np.floor((width-cols)/2))
        blank_icon[row_padding:row_padding+rows,col_padding:col_padding+cols,:] = img[:,:,:]
        img = blank_icon
        # print(img.shape)
        # temp = img [:,:,3]
        # img[:,:,3] = img[:,:,0]
        # img[:,:,0] = temp
        return img


    def play_video(self,frames):
        for i,f in enumerate(frames):
            layout = self.np_arr2json(f)
            img = self.json_to_img(layout)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Restaurant',img)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def get_nparr(self):
        return self.imgarr

    def draw_staff(self, staff):
        for waiter in staff:
            self.dwg.add(self.dwg.image(href=os.path.join(self.bp,"svgs/waiter.svg"), insert = (self.width * waiter['x'], self.height * waiter['y']), size = (8,8)))
            x = int(self.width*waiter['x'])
            y = int(self.height*waiter['y'])
            self.imgarr[y,x] = self.objects["staff"]
    def draw_tables(self, tables):
        for table in tables:
            table_name = table['name'].split(self.name_delimiter)[0]
            x = int(self.width * table['x'])
            y = int(self.height * table['y'])
            self.imgarr[y,x] = self.objects[table_name]
            # if("Round" in table["name"]):
            #     rr,cc = circle(x,y,table["size"]/2)
            #     self.imgarr[rr,cc] = self.objects[table['name']]


            if table['appliances'] == []:
                self.dwg.add(self.dwg.image(href=os.path.join(self.bp,"svgs/"+str(table['seats'])+"_table_round.svg"), insert = (self.width * table['x'], self.height * table['y']), size = (table['seats']*2,table['seats']*2)))
                
            else:
                self.dwg.add(self.dwg.image(href=os.path.join(self.bp,"svgs/"+str(table['seats'])+"_bar_rect.svg"), insert = (self.width * table['x'], self.height * table['y']), size = (table['seats']*2,table['seats']*2)))
    
    def draw_equipment(self, equipment):
        for item in equipment:
            item_name = item['name'].split(self.name_delimiter)[0]
            if item_name in ['Mini Bar', 'Full Bar']:
                continue
            x = int(self.width * item['attributes']['x'])
            y = int(self.height * item['attributes']['y'])
            self.imgarr[y,x] = self.objects[item_name]
            self.dwg.add(self.dwg.image(href=os.path.join(self.bp,self.item_svg_map[item_name]['path']), insert = (self.width * item['attributes']['x'], self.height * item['attributes']['y']), size = (8, 8)))


if __name__ == "__main__":
    
    writer = ImageWriter(10, 10,tables_fn='tables_simple.json', equipment_fn='items_simple.json')
    layout = [{
                                    "type": "image",
                                    "size": 30,
                                    "radius": 30,
                                    "x":0.7,
                                    "y":0.7,
                                    "seats": 4,
                                    "name": "Med Round Table",
                                    "path": "svgs/4_table_round.svg",
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
                                },
                                {
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
                                },
                                {"name": "staff","x":0.5,"y":0.5, "attributes":{"x":0.5,"y":0.5}, "path":"svgs/waiter.svg"}]
    
    import pickle as pkl
    frames = []
    base_path = '/Users/administrator/designAssistant/code/sknightmare-agent/data/simple_profit_low_d'
    for filename in sorted(glob.glob(os.path.join(base_path, '2019*','simulation*'))):
        with open(filename, 'rb') as picklefile:
            data = pkl.load(picklefile)
            for row in data:
                frames.append(row[0])
    writer.play_video(frames)
    # nparr = writer.json_to_img(layout)
    # nparr = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
    # cv2.imshow('img',nparr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(writer.get_image())
    # plt.show()
    # time.sleep(10)
    # writer.get_image().show()

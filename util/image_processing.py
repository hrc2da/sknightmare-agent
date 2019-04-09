import svgwrite
import json
from cairosvg import svg2png
from scipy import misc
import cv2 as cv
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import time
import os
from PIL import Image
from io import BytesIO
import numpy as np
from skimage.draw import circle

class ImageWriter():
    def __init__(self, width, height, base_fp='.', tables_fn='tables.json', equipment_fn='items.json'):
        self.width = width
        self.height = height
        self.bp = base_fp
        self.objects = {"empty":0} # this dict defines a value for every object type that we can add (for the imgarr) 
        self.load_maps(os.path.join(base_fp,tables_fn),os.path.join(base_fp,equipment_fn))
        print("OBJECTS!!!",self.objects)
        self.dwg = svgwrite.Drawing( size = (width, height))
        self.imgarr = np.zeros((width,height))

        #self.dwg.add(self.dwg.rect(insert = (0,0), size = (width, height), stroke = svgwrite.rgb(10, 10, 10), fill = svgwrite.rgb(255, 255, 255)))

    def load_maps(self, tables_fp, equipment_fp):
        with open(equipment_fp, 'r+') as infile:
            data = json.load(infile)
            self.item_svg_map = {}
            for key, item in data.items():
                self.item_svg_map[item['name']] = item
                self.objects[item["name"]] = len(self.objects)
        with open(tables_fp, 'r+') as infile:
            self.tables_svg_map = json.load(infile)
            for key, item in self.tables_svg_map.items():
                print(item["name"], len(self.objects))
                self.objects[item["name"]] = len(self.objects)
                #note that the bars will show up as either a table or an appliance; one will overwrite the other; it doesn't really matter I think for training
        self.objects["staff"] = len(self.objects)
    
    

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

    def get_nparr(self):
        return self.imgarr

    def draw_staff(self, staff):
        for waiter in staff:
            self.dwg.add(self.dwg.image(href=os.path.join(self.bp,"svgs/waiter.svg"), insert = (self.width * waiter['x'], self.height * waiter['y']), size = (4,4)))
            x = int(self.width*waiter['x'])
            y = int(self.height*waiter['y'])
            self.imgarr[x,y] = self.objects["staff"]
    def draw_tables(self, tables):
        for table in tables:
            x = int(self.width * table['x'])
            y = int(self.height * table['y'])
            self.imgarr[x,y] = self.objects[table['name']]
            # if("Round" in table["name"]):
            #     rr,cc = circle(x,y,table["size"]/2)
            #     self.imgarr[rr,cc] = self.objects[table['name']]


            if table['appliances'] == []:
                self.dwg.add(self.dwg.image(href=os.path.join(self.bp,"svgs/"+str(table['seats'])+"_table_round.svg"), insert = (self.width * table['x'], self.height * table['y']), size = (table['seats'],table['seats'])))
                
            else:
                self.dwg.add(self.dwg.image(href=os.path.join(self.bp,"svgs/"+str(table['seats'])+"_bar_rect.svg"), insert = (self.width * table['x'], self.height * table['y']), size = (table['seats'],table['seats'])))
    
    def draw_equipment(self, equipment):
        for item in equipment:
            if item['name'] in ['Mini Bar', 'Full Bar']:
                continue
            x = int(self.width * item['attributes']['x'])
            y = int(self.height * item['attributes']['y'])
            self.imgarr[x,y] = self.objects[item['name']]
            self.dwg.add(self.dwg.image(href=os.path.join(self.bp,self.item_svg_map[item['name']]['path']), insert = (self.width * item['attributes']['x'], self.height * item['attributes']['y']), size = (4, 4)))


if __name__ == "__main__":
    filepath = "test.json"
    writer = ImageWriter(65, 34)
    writer.load_json(filepath)
    # plt.imshow(writer.get_image())
    # plt.show()
    # time.sleep(10)
    writer.get_image().show()

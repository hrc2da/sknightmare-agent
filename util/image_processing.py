import svgwrite
import json
from cairosvg import svg2png
from scipy import misc

class ImageWriter():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.load_maps('items.json', 'tables.json')
        self.dwg = svgwrite.Drawing('test.svg', size = (width, height))
        self.dwg.add(self.dwg.rect(insert = (0,0), size = (width, height), stroke = svgwrite.rgb(10, 10, 10), fill = svgwrite.rgb(255, 255, 255)))

    def load_maps(self, items_fp, tables_fp):
        with open(items_fp, 'r+') as infile:
            data = json.load(infile)
            self.item_svg_map = {}
            for key, item in data.items():
                self.item_svg_map[item['name']] = item
        with open(tables_fp, 'r+') as infile:
            self.tables_svg_map = json.load(infile)
        

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
        svg2png(bytes(self.svg_string, 'utf-8'), write_to=open("temp.png", 'wb'))
        return misc.imread('temp.png')

    def draw_staff(self, staff):
        for waiter in staff:
            self.dwg.add(self.dwg.image(href="svgs/waiter.svg", insert = (self.width * waiter['x'], self.height * waiter['y']), size = (4,4)))
    
    def draw_tables(self, tables):
        for table in tables:
            if table['attributes']['appliances'] == []:
                self.dwg.add(self.dwg.image(href="svgs/"+str(table['seats'])+"_table_round.svg", insert = (self.width * table['x'], self.height * table['y']), size = (table['seats'],table['seats'])))
    
    def draw_equipment(self, equipment):
        for item in equipment:
            self.dwg.add(self.dwg.image(href=self.item_svg_map[item['name']]['path'], insert = (self.width * item['attributes']['x'], self.height * item['attributes']['y']), size = (4, 4)))


if __name__ == "__main__":
    filepath = "test.json"
    writer = ImageWriter(65, 34)
    writer.load_json(filepath)
    writer.get_image()
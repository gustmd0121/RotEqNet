import json
import os
import numpy as np
from scipy import misc
from PIL import Image
from ast import literal_eval

class Parser:
    def __init__(self, folder):
        self.folder = folder
        self.img_size = (1080, 1920)

    def parse(self, file):
        beetle_props = open(self.folder + file + "/" + file + '_beetle_props.txt', 'w')
        ball_props = open(self.folder + file + "/" + file + '_ball_props.txt', 'w')

        with open(self.folder + file + "/" + file + ".grndr") as json_data:
            data = json.load(json_data)

            data_beetle = []
            data_ball = []

            json_color = ''
            json_ref = 0
            json_x = 0
            json_y = 0
            json_width = 0
            json_heigt = 0
            json_has_direction = False
            json_direction = 0

            json_length = 0

            #get json length
            for images in data['ImageReferences']:
                json_length += 1

            #get json information
            for label in data['Labels']:
                for pool in label['Label']['ImageBuildPool']:
                    for ref in pool['Item']['ImageBuilds']:
                        json_ref = ref['ImageBuild']['ImageReference']
                        for layer in ref['ImageBuild']['Layers']:
                            for draftitem in layer['Layer']['DraftItems']:
                                for prop in draftitem['DraftItem']['Properties']:
                                    if(prop['Property']['ID'] == 'PrimaryColor'):
                                        json_color = prop['Property']['Value']
                                    if(prop['Property']['ID'] == 'Position'):
                                        json_x = int(float(prop['Property']['Value'].split(";")[0]))
                                        json_y = int(float(prop['Property']['Value'].split(";")[1]))
                                    if(prop['Property']['ID'] == 'HasDirection'):
                                        if(prop['Property']['Value']=='true'):
                                            json_has_direction =  True
                                        else:
                                            json_has_direction =  False
                                    if(prop['Property']['ID'] == 'Direction'):
                                        json_direction = int(float(prop['Property']['Value']))
                                    if(prop['Property']['ID'] == 'Size'):
                                        json_width = int(float(prop['Property']['Value'].split(";")[0]))
                                        json_height = int(float(prop['Property']['Value'].split(";")[1]))
                                #color labeled method
                                #beetle
                                if(json_color == '#ff0000'):
                                    data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                                #ball
                                if(json_color == '#00ff00'):
                                    data_ball.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                                #two label method
                                if(json_color == '#ffffff'):
                                    if(label['Label']['Name'] == 'Beetle'):
                                        data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                                    if(label['Label']['Name'] == 'Ball'):
                                        data_ball.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction))
                                #reset
                                json_color = ''

            #sort data
            data_beetle.sort()
            #add missing
            for i in range(0,json_length):
                if not(i in [x[0] for x in data_beetle]):
                    data_beetle.insert(i, (i, (0, 0), (0, 0), False, 0))
            #write data
            for x in data_beetle:
                #print(x)
                beetle_props.write(str(x) + "\n")

            #sort data
            data_ball.sort()
            #add missing
            for i in range(0,json_length):
                if not(i in [x[0] for x in data_ball]):
                    data_ball.insert(i, (i, (0, 0), (0, 0), False, 0))
            #write data
            for x in data_ball:
                #print(x)
                ball_props.write(str(x) + "\n")

    def create_numpy_arrays(self, file, scale_factor):
        if os.path.exists(self.folder + file +"/" + file + "_masks.npz"):
            print("Numpy array file already exists.")
            return

        print("Generating ground truth numpy arrays ...")

        if scale_factor != 1:
            self.set_img_size(self.img_size, scale_factor)

        img_folder = self.folder + file + "/" + file + "_imgs/"

        beetle_props_file = self.folder + file + "/" + file + "_beetle_props.txt"
        ball_props_file = self.folder + file + "/" + file + "_ball_props.txt"

        with open(beetle_props_file) as f:
            beetle_props = [literal_eval(line) for line in f.readlines()]

        with open(ball_props_file) as f:
            ball_props = [literal_eval(line) for line in f.readlines()]

        beetle_masks = self.__create_mask(beetle_props, scale_factor)
        ball_masks = self.__create_mask(ball_props, scale_factor)

        np.savez_compressed(self.folder + file +"/" + file + "_masks", beetle=beetle_masks, ball=ball_masks)


    def __create_mask(self, props, scale_factor):
        mask = np.zeros([self.img_size[0], self.img_size[1], len(props)])
        for i in range(0, len(props)):
            index, pos, size, hasdir, dir = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            for x in range(int(pos_x * scale_factor), int((pos_x + size_w) * scale_factor)):
                for y in range(int(pos_y * scale_factor), int((pos_y + size_h) * scale_factor)):
                    mask[y][x][i] = 1
        return mask


    def generate_input(self, file, scale_factor):
        if os.path.exists(self.folder + file +"/" + file + "_input.npz"):
            print("Numpy array file already exists.")
            return

        print("Generating input numpy array ...")
        img_count = len(os.listdir(self.folder + file + "/" + file + "_imgs/"))
        data = np.zeros([self.img_size[0], self.img_size[1], img_count])
        for i in range(0, img_count):
            for x in range(0, img_count, int(1/scale_factor)):
                for y in range(0, img_count, int(1/scale_factor)):
                    data[y][x][i] = 1
        np.savez_compressed(self.folder + file +"/" + file + "_input", data=data)


    def set_img_size(self, img_size, scale_factor):
        self.img_size = (int(img_size[0] * scale_factor), int(img_size[1] * scale_factor))

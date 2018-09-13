import json
import os
import numpy as np
import cv2
from scipy import misc
from PIL import Image
from ast import literal_eval

class Parser:
    def __init__(self, folder, img_size):
        self.folder = folder
        self.img_size = img_size
        self.new_img_size = img_size
        self.offset = []


    def parse(self, file):
        self.offset = []
        print("Parsing grinder file ...")

        grndr_name = self.folder + file + "/" + file + ".grndr"

        if not os.path.isfile(grndr_name):
            grndr_name = self.folder + file + "/" + file + "_db.grndr"
            if not os.path.isfile(grndr_name):
                print("No grinder file.")
                return

        beetle_props = open(self.folder + file + "/" + file + '_beetle_props.txt', 'w')
        ball_props = open(self.folder + file + "/" + file + '_ball_props.txt', 'w')

        with open(grndr_name) as json_data:
            data = json.load(json_data)

            data_beetle = []
            data_ball = []
            data_ref = []

            json_color = ''
            json_ref = 0
            json_file = ''
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
                data_ref.append((images['ImageReference']['Index'], images['ImageReference']['File']))

            #get json information
            for label in data['Labels']:
                for pool in label['Label']['ImageBuildPool']:
                    for ref in pool['Item']['ImageBuilds']:
                        json_ref = ref['ImageBuild']['ImageReference']
                        json_file = [data_ref for data_ref in data_ref if data_ref[0] == int(json_ref)][0][1]
                        json_file = json_file.split('/')[len(json_file.split('/'))-1]
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

                                if not(os.path.isfile(self.folder + file + "/" + file + "_imgs/" + json_file)):
                                    print("Error: Cannot find", "'" + self.folder + file + "/" + file + "_imgs/" + json_file)
                                else:
                                    #color labeled method
                                    #beetle
                                    if(json_color == '#ff0000'):
                                        data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction, json_file))
                                    #ball
                                    if(json_color == '#00ff00'):
                                        data_ball.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction, json_file))
                                    #two label method
                                    if(json_color == '#ffffff'):
                                        if(label['Label']['Name'] == 'Beetle'):
                                            data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction, json_file))
                                        if(label['Label']['Name'] == 'Ball'):
                                            data_ball.append((json_ref, (json_x, json_y), (json_width, json_height), json_has_direction, json_direction, json_file))
                                    #reset
                                json_color = ''

            #sort data
            data_beetle.sort()
            #add missing
            # for i in range(0,json_length):
            #     if not(i in [x[0] for x in data_beetle]):
            #         data_beetle.insert(i, (i, (0, 0), (0, 0), False, 0, ''))
            for i in range(0,len(data_ball)):
                if not(data_ball[i][0] in [x[0] for x in data_beetle]):
                    data_beetle.insert(i, (i, (0, 0), (0, 0), False, 0, data_ball[i][5]))
            #write data
            for x in data_beetle:
                #print(x)
                beetle_props.write(str(x) + "\n")

            #sort data
            data_ball.sort()
            #add missing
            # for i in range(0,json_length):
            #     if not(i in [x[0] for x in data_ball]):
            #         data_ball.insert(i, (i, (0, 0), (0, 0), False, 0, ''))
            for i in range(0,len(data_beetle)):
                if not(data_beetle[i][0] in [x[0] for x in data_ball]):
                    data_ball.insert(i, (i, (0, 0), (0, 0), False, 0, data_beetle[i][5]))

            #check wheter files are 'equal'
            if(len(data_ball)!=len(data_beetle)):
                print("Error: The number of ball labels is different from the number of beetle labels.")

            for i in range(0,len(data_ball)):
                self.offset.append([0, 0])
                if(data_ball[i][5] != data_beetle[i][5]):
                    print("Error: Label number", i, "is different. Ball image:", data_ball[i][5], ", Beetle image:", data_beetle[i][5])

            #write data
            for x in data_ball:
                #print(x)
                ball_props.write(str(x) + "\n")


    def calc_offset(self, file, new_img_size, target = 'beetle'):
        file_name = ''
        other_file_name = ''
        if(target == 'beetle'):
            file_name = self.folder + file + "/" + file + "_beetle_props.txt"
            other_file_name = self.folder + file + "/" + file + "_ball_props.txt"
        elif(target == 'ball'):
            file_name = self.folder + file + "/" + file + "_ball_props.txt"
            other_file_name = self.folder + file + "/" + file + "_beetle_props.txt"
        else:
            print(target, "is no option.")
            return

        with open(file_name) as f:
            props = [literal_eval(line) for line in f.readlines()]

        with open(other_file_name) as f:
            other_props = [literal_eval(line) for line in f.readlines()]

        if(len(props) != len(self.offset)):
            print("You have to call parse before calc_offset().")
            return

        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            index_o, pos_o, size_o, hasdir_o, dir_o, file_o = other_props[i]
            pos_x_o, pos_y_o = pos_o
            size_w_o, size_h_o = size_o

            #TODO check if other target is outside the crop
            #center target
            self.offset[i][0] = int(self.img_size[1] - (self.img_size[1] - pos_x - size_w/2 + new_img_size[1]/2))
            if self.offset[i][0] < 0:
                self.offset[i][0] = 0
            if(self.offset[i][0] + new_img_size[1] >= self.img_size[1]):
                self.offset[i][0] = self.img_size[1] - new_img_size[1] - 1

            self.offset[i][1] = int(self.img_size[0] - (self.img_size[0] - pos_y - size_h/2 + new_img_size[0]/2))
            if self.offset[i][1] < 0:
                self.offset[i][1] = 0
            if(self.offset[i][1] + new_img_size[0] >= self.img_size[0]):
                self.offset[i][1] = self.img_size[0] - new_img_size[0] - 1

            # if(pos_x_o + size_w_o/2 > new_img_size[0] + self.offset[i][0]):
            #     self.offset[i][0] = int(self.offset[i][0] - (pos_x_o + size_w_o/2) - (new_img_size[0] + self.offset[i][0]))
            # elif(pos_x_o + size_w_o/2 < new_img_size[0] + self.offset[i][0]):
            #     self.offset[i][0] = int(self.offset[i][0] - (new_img_size[0] + self.offset[i][0]) - (pos_x_o + size_w_o/2))
            #
            # if(pos_y_o + size_h_o/2 > new_img_size[1] + self.offset[i][1]):
            #     offset[i][0] = int(self.offset[i][1] - (pos_x_o + size_h_o/2) - (new_img_size[1] + self.offset[i][0]))
            # elif(pos_x_o + size_w_o/2 < new_img_size[1] + self.offset[i][0]):
            #     offset[i][0] = int(self.offset[i][1] - (new_img_size[1] + self.offset[i][0]) - (pos_x_o + size_h_o/2))
        self.new_img_size = new_img_size


    def create_numpy_arrays(self, file, override = False):
        if not(override) and os.path.exists(self.folder + file +"/" + file + "_masks.npz"):
            print("Ground truth numpy array file already exists.")
            return

        print("Generating ground truth numpy array ...")

        beetle_props_file = self.folder + file + "/" + file + "_beetle_props.txt"
        ball_props_file = self.folder + file + "/" + file + "_ball_props.txt"

        if (not os.path.isfile(beetle_props_file) or not os.path.isfile(ball_props_file)):
            print("No props files.")
            return

        with open(beetle_props_file) as f:
            beetle_props = [literal_eval(line) for line in f.readlines()]

        with open(ball_props_file) as f:
            ball_props = [literal_eval(line) for line in f.readlines()]

        beetle_masks = self.__create_mask(beetle_props)
        ball_masks = self.__create_mask(ball_props)

        print("Compressing and saving ground truth numpy array ...")
        np.savez_compressed(self.folder + file +"/" + file + "_masks", beetle=beetle_masks, ball=ball_masks)


    def __create_mask(self, props):
        mask = np.zeros([self.new_img_size[0] , self.new_img_size[1], len(props)], dtype='float32')
        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            try:
                for x in range((pos_x - self.offset[i][0]), (pos_x + size_w - self.offset[i][0])):
                    for y in range((pos_y - self.offset[i][1]), (pos_y + size_h - self.offset[i][1])):
                        mask[y][x][i] = 1
            except (IndexError):
                print(i)
                print(self.offset[i])
                print(pos)
                print(size)
                raise
        return mask


    def generate_input(self, file, override = False):
        if not(override) and os.path.exists(self.folder + file +"/" + file + "_input.npz"):
            print("Input numpy array file already exists.")
            return

        print("Generating input numpy array ...")

        ball_props_file = self.folder + file + "/" + file + "_ball_props.txt"
        beetle_props_file = self.folder + file + "/" + file + "_beetle_props.txt"

        if (not os.path.isfile(beetle_props_file) or not os.path.isfile(ball_props_file)):
            print("No props files.")
            return

        with open(ball_props_file) as f:
            ball_props = [literal_eval(line) for line in f.readlines()]
        with open(beetle_props_file) as f:
            beetle_props = [literal_eval(line) for line in f.readlines()]

        if(len(ball_props)!=len(beetle_props)):
            print("Error: The number of ball labels is diffent from the number of beetle labels.")

        data = np.zeros([len(ball_props), self.new_img_size[0], self.new_img_size[1]], dtype='float32')
        for i in range(0, len(ball_props)):
            img_name = ball_props[i][5]
            if(img_name != beetle_props[i][5]):
                print("Error: Different image files.")
            scaled_img = cv2.imread(self.folder + file + "/" + file + "_imgs/" + img_name)
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
            scaled_img = scaled_img[self.offset[i][1]:self.offset[i][1]+self.new_img_size[0], self.offset[i][0]:self.offset[i][0]+self.new_img_size[1]]
            try:
                data[i, :, :] = scaled_img
            except (ValueError):
                print(self.offset[i][0])
                print(self.offset[i][0]+self.new_img_size[0])
                raise

        print("Compressing and saving input numpy array ...")
        np.savez_compressed(self.folder + file +"/" + file + "_input", data=data)


    def set_img_size(self, img_size):
        self.img_size = img_size


    def load_image(self, file, i):
        if (not os.path.isfile(self.folder + file +"/" + file + "_masks.npz") or not os.path.isfile(self.folder + file +"/" + file + "_input.npz")):
            print("No numpy files.")
            return

        print("Loading ground truth numpy array ...")
        ground = np.load(self.folder + file +"/" + file + "_masks.npz")
        print("Loading input numpy array ...")
        data = np.load(self.folder + file +"/" + file + "_input.npz")

        beetle = ground['beetle']
        ball = ground['ball']
        full = data['data']

        if(len(ball[0][0]) < i or len(beetle[0][0]) < i or len(full) < i):
            print("Error: There are less than", i, "images.")
            return;

        np_ball = ball[:, :, i] * 255
        np_beetle = beetle[:, :, i] * 255
        np_img = np.asarray(full[i, :, :], dtype="uint8")

        img_ball = Image.fromarray(np_ball)
        img_beetle = Image.fromarray(np_beetle)
        img = Image.fromarray(np_img)

        #Sometimes show() doesnt work without print ...
        print(img_beetle.show())
        print(img_ball.show())
        print(img.show())

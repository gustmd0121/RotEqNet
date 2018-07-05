import json
import os
import numpy as np
import cv2
from scipy import misc
from PIL import Image
from ast import literal_eval
from random import randint

class Parser:
    def __init__(self, folder):
        self.folder = folder
        self.img_size = (1080, 1920)
        self.new_img_size = self.img_size
        self.offset = []


    def create_scaled_files(self, file, scale_factor, override = True):
        print("")
        print("Parsing", file, "...")
        self.__generate_prop(file)
        self.__generate_ground_truth(file, override, scale_factor)
        self.__generate_images(file, override, scale_factor)
        print("Finished", file + ".")


    def create_cropped_files(self, file, new_size, center = "beetle", override = True):
        print("")
        print("Parsing", file, "...")
        self.__generate_prop(file)
        self.__calc_offset(file, new_size, center)
        self.__generate_ground_truth(file, override)
        self.__generate_images(file, override)
        print("Finished", file + ".")


    def __generate_prop(self, file):
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
            for i in range(0,len(data_ball)):
                if not(data_ball[i][0] in [x[0] for x in data_beetle]):
                    data_beetle.insert(i, (i, (0, 0), (0, 0), False, 0, data_ball[i][5]))

            #sort data
            data_ball.sort()
            #add missing
            for i in range(0,len(data_beetle)):
                if not(data_beetle[i][0] in [x[0] for x in data_ball]):
                    data_ball.insert(i, (i, (0, 0), (0, 0), False, 0, data_beetle[i][5]))

            #write data
            for x in data_beetle:
                beetle_props.write(str(x) + "\n")

            #check wheter files are 'equal'
            if(len(data_ball)!=len(data_beetle)):
                print("Error: The number of ball labels is different from the number of beetle labels.")

            self.offset = []
            for i in range(0,len(data_ball)):
                self.offset.append([0, 0])
                if(data_ball[i][5] != data_beetle[i][5]):
                    print("Error: Label number", i, "is different. Ball image:", data_ball[i][5], ", Beetle image:", data_beetle[i][5])

            #write data
            for x in data_ball:
                ball_props.write(str(x) + "\n")


    def __calc_offset(self, file, new_img_size, target = 'beetle'):
        file_name = ''
        if(target == 'beetle'):
            file_name = self.folder + file + "/" + file + "_beetle_props.txt"
        elif(target == 'ball'):
            file_name = self.folder + file + "/" + file + "_ball_props.txt"
        else:
            print(target, "is no option.")
            return

        with open(file_name) as f:
            props = [literal_eval(line) for line in f.readlines()]

        if(len(props) != len(self.offset)):
            print("You have to call parse before calc_offset().")
            return

        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            #center target
            self.offset[i][0] = int(self.img_size[1] - (self.img_size[1] - pos_x - size_w/2 + new_img_size[1]/2))
            self.offset[i][0] = self.offset[i][0] + randint(-int((new_img_size[1] - size_w) / 4), int((new_img_size[1] - size_w) / 4))
            if self.offset[i][0] < 0:
                self.offset[i][0] = 0
            if(self.offset[i][0] + new_img_size[1] >= self.img_size[1]):
                self.offset[i][0] = self.img_size[1] - new_img_size[1] - 1

            self.offset[i][1] = int(self.img_size[0] - (self.img_size[0] - pos_y - size_h/2 + new_img_size[0]/2))
            self.offset[i][1] = self.offset[i][1] + randint(-int((new_img_size[0] - size_h) / 4), int((new_img_size[0] - size_h) / 4))
            if self.offset[i][1] < 0:
                self.offset[i][1] = 0
            if(self.offset[i][1] + new_img_size[0] >= self.img_size[0]):
                self.offset[i][1] = self.img_size[0] - new_img_size[0] - 1
        self.new_img_size = new_img_size


    def __generate_ground_truth(self, file, override = False, scale_factor = -1):
        if not(override) and os.path.exists(self.folder + file +"/" + file + "_masks.npz"):
            print("Ground truth numpy array file already exists.")
            return

        img_folder = self.folder + file + "/" + file + "_imgs/"
        beetle_props_file = self.folder + file + "/" + file + "_beetle_props.txt"
        ball_props_file = self.folder + file + "/" + file + "_ball_props.txt"

        if (not os.path.isfile(beetle_props_file) or not os.path.isfile(ball_props_file)):
            print("No props files.")
            return

        print("Generating ground truth numpy array ...")

        with open(beetle_props_file) as f:
            beetle_props = [literal_eval(line) for line in f.readlines()]

        with open(ball_props_file) as f:
            ball_props = [literal_eval(line) for line in f.readlines()]

        if(scale_factor != -1):
            beetle_masks = self.__create_scaled_mask(beetle_props, scale_factor)
            ball_masks = self.__create_scaled_mask(ball_props, scale_factor)
        else:
            beetle_masks = self.__create_cropped_mask(beetle_props)
            ball_masks = self.__create_cropped_mask(ball_props)

        print("Compressing and saving ground truth numpy array ...")
        np.savez_compressed(self.folder + file +"/" + file + "_masks", beetle=beetle_masks, ball=ball_masks)



    def __create_scaled_mask(self, props, scale_factor):
        mask = np.zeros([len(props), 2, int(self.img_size[0] * scale_factor), int(self.img_size[1] * scale_factor)], dtype='float32')
        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            for x in range(int(pos_x * scale_factor), int((pos_x + size_w) * scale_factor)):
                for y in range(int(pos_y * scale_factor), int((pos_y + size_h) * scale_factor)):
                    mask[i][0][y][x] = 1
                    if hasdir:
                        mask[i][1][y][x] = dir
        return mask


    def __create_cropped_mask(self, props):
        mask = np.zeros([len(props), 2, self.new_img_size[0] , self.new_img_size[1]], dtype='float32')
        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            for x in range(abs(pos_x - self.offset[i][0]), abs(pos_x + size_w - self.offset[i][0])):
                for y in range(abs(pos_y - self.offset[i][1]), abs(pos_y + size_h - self.offset[i][1])):
                    try:
                        mask[i][0][y][x] = 1
                        if hasdir:
                            mask[i][1][y][x] = dir
                    except (IndexError):
                        pass
        return mask


    def __generate_images(self, file, override = False, scale_factor = -1):
        if not(override) and os.path.exists(self.folder + file +"/" + file + "_input.npz"):
            print("Input numpy array file already exists.")
            return

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
            return

        print("Generating input numpy array ...")

        if(scale_factor != -1):
            data = np.zeros([len(ball_props), 1, int(self.img_size[0]*scale_factor), int(self.img_size[1]*scale_factor)], dtype='float32')
        else:
            data = np.zeros([len(ball_props), 1, self.new_img_size[0], self.new_img_size[1]], dtype='float32')
        for i in range(0, len(ball_props)):
            img_name = ball_props[i][5]
            if(img_name != beetle_props[i][5]):
                print("Error: Different image files.")
            scaled_img = cv2.imread(self.folder + file + "/" + file + "_imgs/" + img_name)
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
            if(scale_factor != -1):
                scaled_img = cv2.resize(scaled_img, (int(self.img_size[1]*scale_factor), int(self.img_size[0]*scale_factor)), interpolation=cv2.INTER_CUBIC)
            else:
                scaled_img = scaled_img[self.offset[i][1]:self.offset[i][1]+self.new_img_size[0], self.offset[i][0]:self.offset[i][0]+self.new_img_size[1]]
            data[i, 0, :, :] = scaled_img

            # if(scale_factor != -1):
            #     data[i, 1, :, :] = np.zeros([int(self.img_size[0]*scale_factor), int(self.img_size[1]*scale_factor)], dtype='float32')
            # else:
            #     data[i, 1, :, :] = np.zeros([self.new_img_size[0], self.new_img_size[1]], dtype='float32')

        print("Compressing and saving input numpy array ...")
        np.savez_compressed(self.folder + file +"/" + file + "_input", data=data)



    def set_img_size(self, img_size):
        self.img_size = (img_size[0], img_size[1])


    #Doesnt work with current parser
    def load_numpy(self, file, i):
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

        np_ball = ball[:, :, i]
        np_beetle = beetle[:, :, i]
        np_img = full[i, :, :]

        np_ball = np_ball * 255
        np_beetle = np_beetle * 255
        img_ball = Image.fromarray(np_ball)
        img_beetle = Image.fromarray(np_beetle)
        img = Image.fromarray(np_img)
        # color image must contain integer values
        # np_img = np.asarray(full[i, :, :], dtype="uint8")

        #Sometimes show() doesnt work without print ...
        print(img_beetle.show())
        print(img_ball.show())
        print(img.show())

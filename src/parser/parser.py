# Global imports
import json
import os
import numpy as np
import cv2
from ast import literal_eval
import math
from random import randint

# Local imports
from framework.utils.utils import *


class Parser:
    def __init__(self, folder, combined_folder):
        self.folder = folder
        self.img_size = (1080, 1920)
        self.new_img_size = self.img_size
        self.offset = []
        self.combined_folder = combined_folder

    def combine_numpy_arrays(self):
        """ Combines all available datasets into one single dataset. """
        all_data = []
        all_masks_beetle = []
        all_masks_ball = []
        print("")
        print("Combining", "...")
        if not os.path.isdir(self.folder + self.combined_folder):
            os.mkdir(self.folder + self.combined_folder)
        for sub_folder in os.listdir(self.folder):
            sub_path = os.path.join(self.folder, sub_folder)
            if os.path.isdir(sub_path) and sub_folder != self.combined_folder:
                print("Loading ground truth", sub_folder, "...")
                ground = np.load(self.folder + sub_folder + "/" + sub_folder + "_masks.npz")
                print("Loading input", sub_folder, "...")
                data = np.load(self.folder + sub_folder + "/" + sub_folder + "_input.npz")['data']
                if len(all_data) == 0:
                    all_data = data
                    all_masks_beetle = ground['beetle']
                    all_masks_ball = ground['ball']
                else:
                    all_data = np.append(all_data, data, axis=0)
                    all_masks_beetle = np.append(all_masks_beetle, ground['beetle'], axis=0)
                    all_masks_ball = np.append(all_masks_ball, ground['ball'], axis=0)
        print("Compressing and saving ground truth numpy array ...")
        np.savez_compressed(self.folder + self.combined_folder + "/" + self.combined_folder + "_masks",
                            beetle=all_masks_beetle, ball=all_masks_ball)
        print("Compressing and saving input numpy array ...")
        np.savez_compressed(self.folder + self.combined_folder + "/" + self.combined_folder + "_input", data=all_data)
        print("Finished combining", len(all_data), "images")

    def create_scaled_files(self, file, scale_factor, overwrite=True):
        """
        Scales images in dataset.
        :param file: dataset name
        :param scale_factor: scale factor
        :param overwrite: whether to overwrite existing files
        """
        if file != self.combined_folder:
            print("")
            print("Parsing", file, "...")
            self._generate_prop(file)
            self._generate_ground_truth(file, overwrite, scale_factor)
            self._generate_images(file, overwrite, scale_factor)
            print("Finished", file + ".")

    def create_cropped_files(self, file, new_size, center="beetle", overwrite=True):
        """
        Crops images around beetle/ball.
        :param file: dataset name
        :param new_size: tuple (height, width) of cropped image
        :param center: "beetle" or "ball"
        :param overwrite: whether to overwrite existing files
        """
        if file != self.combined_folder:
            print("")
            print("Parsing", file, "...")
            self._generate_prop(file)
            self._calc_offset(file, new_size, center)
            self._generate_ground_truth(file, overwrite)
            self._generate_images(file, overwrite)
            print("Finished", file + ".")

    def _generate_prop(self, file):
        """
        Create temporary prop file, which contain all infos of .grndr file in tuple format.
        :param file: dataset name
        """
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
            json_x = 0
            json_y = 0
            json_width = 0
            json_height = 0
            json_has_direction = False
            json_direction = 0
            json_length = 0

            # get json length
            for images in data['ImageReferences']:
                json_length += 1
                data_ref.append((images['ImageReference']['Index'], images['ImageReference']['File']))

            # get json information
            for label in data['Labels']:
                for pool in label['Label']['ImageBuildPool']:
                    for ref in pool['Item']['ImageBuilds']:
                        json_ref = ref['ImageBuild']['ImageReference']
                        json_file = [data_ref for data_ref in data_ref if data_ref[0] == int(json_ref)][0][1]
                        json_file = json_file.split('/')[len(json_file.split('/')) - 1]
                        for layer in ref['ImageBuild']['Layers']:
                            for draftitem in layer['Layer']['DraftItems']:
                                for prop in draftitem['DraftItem']['Properties']:
                                    if (prop['Property']['ID'] == 'PrimaryColor'):
                                        json_color = prop['Property']['Value']
                                    if (prop['Property']['ID'] == 'Position'):
                                        json_x = int(float(prop['Property']['Value'].split(";")[0]))
                                        json_y = int(float(prop['Property']['Value'].split(";")[1]))
                                    if (prop['Property']['ID'] == 'HasDirection'):
                                        if (prop['Property']['Value'] == 'true'):
                                            json_has_direction = True
                                        else:
                                            json_has_direction = False
                                    if (prop['Property']['ID'] == 'Direction'):
                                        # change 0 degrees from east orientation in .grndr file to north orientation
                                        # change rotation direction from counterclockwise to clockwise
                                        json_direction = ((360 - int(float(prop['Property']['Value']))) + 90) % 360
                                    if (prop['Property']['ID'] == 'Size'):
                                        json_width = int(float(prop['Property']['Value'].split(";")[0]))
                                        json_height = int(float(prop['Property']['Value'].split(";")[1]))

                                if not (os.path.isfile(self.folder + file + "/" + file + "_imgs/" + json_file)):
                                    print("Error: Cannot find",
                                          "'" + self.folder + file + "/" + file + "_imgs/" + json_file)
                                else:
                                    # color labeled method
                                    # beetle
                                    if (json_color == '#ff0000'):
                                        data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height),
                                                            json_has_direction, json_direction, json_file))
                                    # ball
                                    if (json_color == '#00ff00'):
                                        data_ball.append((json_ref, (json_x, json_y), (json_width, json_height),
                                                          json_has_direction, json_direction, json_file))
                                    # two label method
                                    if (json_color == '#ffffff'):
                                        if (label['Label']['Name'] == 'Beetle'):
                                            data_beetle.append((json_ref, (json_x, json_y), (json_width, json_height),
                                                                json_has_direction, json_direction, json_file))
                                        if (label['Label']['Name'] == 'Ball'):
                                            data_ball.append((json_ref, (json_x, json_y), (json_width, json_height),
                                                              json_has_direction, json_direction, json_file))
                                    # reset
                                json_color = ''

            # sort data
            data_beetle.sort()
            # add missing
            for i in range(0, len(data_ball)):
                if not (data_ball[i][0] in [x[0] for x in data_beetle]):
                    data_beetle.insert(data_ball[i][0], (data_ball[i][0], (0, 0), (0, 0), False, 0, data_ball[i][5]))
            data_beetle.sort()

            # sort data
            data_ball.sort()
            # add missing
            for i in range(0, len(data_beetle)):
                if not (data_beetle[i][0] in [x[0] for x in data_ball]):
                    data_ball.insert(data_beetle[i][0],
                                     (data_beetle[i][0], (0, 0), (0, 0), False, 0, data_beetle[i][5]))
            data_ball.sort()

            # write data
            for x in data_beetle:
                beetle_props.write(str(x) + "\n")

            # check wheter files are 'equal'
            if (len(data_ball) != len(data_beetle)):
                print("Error: The number of ball labels is different from the number of beetle labels.")

            self.offset = []
            for i in range(0, len(data_ball)):
                self.offset.append([0, 0])
                if (data_ball[i][5] != data_beetle[i][5]):
                    print("Error: Label number", i, "is different. Ball image:", data_ball[i][5], ", Beetle image:",
                          data_beetle[i][5])

            # write data
            for x in data_ball:
                ball_props.write(str(x) + "\n")

    def _calc_offset(self, file, new_img_size, target='beetle'):
        """
        Calcs random offset, so that beetle and ball are both in the frame but not centered.
        :param file: dataset name
        :param new_img_size: tuple (height, width) of cropped image
        :param target: "beetle" or "ball"
        """
        if (target == 'beetle'):
            file_name = self.folder + file + "/" + file + "_beetle_props.txt"
        elif (target == 'ball'):
            file_name = self.folder + file + "/" + file + "_ball_props.txt"
        else:
            print(target, "is no option.")
            return

        with open(file_name) as f:
            props = [literal_eval(line) for line in f.readlines()]

        if len(self.offset) == 0:
            self.offset = []
            for i in range(0, len(props)):
                self.offset.append([0, 0])

        if (len(props) != len(self.offset)):
            print("You have to call parse before calc_offset().")
            return

        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            # center target
            self.offset[i][0] = int(self.img_size[1] - (self.img_size[1] - pos_x - size_w / 2 + new_img_size[1] / 2))
            self.offset[i][0] = self.offset[i][0] + randint(-int((new_img_size[1] - size_w) / 4),
                                                            int((new_img_size[1] - size_w) / 4))
            if self.offset[i][0] < 0:
                self.offset[i][0] = 0
            if (self.offset[i][0] + new_img_size[1] >= self.img_size[1]):
                self.offset[i][0] = self.img_size[1] - new_img_size[1] - 1

            self.offset[i][1] = int(self.img_size[0] - (self.img_size[0] - pos_y - size_h / 2 + new_img_size[0] / 2))
            self.offset[i][1] = self.offset[i][1] + randint(-int((new_img_size[0] - size_h) / 4),
                                                            int((new_img_size[0] - size_h) / 4))
            if self.offset[i][1] < 0:
                self.offset[i][1] = 0
            if (self.offset[i][1] + new_img_size[0] >= self.img_size[0]):
                self.offset[i][1] = self.img_size[0] - new_img_size[0] - 1
        self.new_img_size = new_img_size

    def _generate_ground_truth(self, file, overwrite=False, scale_factor=-1):
        """
        Creates ground truth data from masks in compressed .npz file format.
        :param file: dataset name
        :param overwrite: whether to overwrite existing files
        :param scale_factor: scale factor
        """
        if not (overwrite) and os.path.exists(self.folder + file + "/" + file + "_masks.npz"):
            print("Ground truth numpy array file already exists.")
            return

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

        if (scale_factor != -1):
            beetle_masks = self._create_scaled_mask(beetle_props, scale_factor)
            ball_masks = self._create_scaled_mask(ball_props, scale_factor)
        else:
            beetle_masks = self._create_cropped_mask(beetle_props)
            ball_masks = self._create_cropped_mask(ball_props)

        print("Compressing and saving ground truth numpy array ...")
        np.savez_compressed(self.folder + file + "/" + file + "_masks", beetle=beetle_masks, ball=ball_masks)

    def _create_scaled_mask(self, props, scale_factor):
        """
        Creates scaled mask images.
        :param props: name of props file
        :param scale_factor: scale factor
        :return: scaled mask
        """
        mask = np.zeros([len(props), 2, int(self.img_size[0] * scale_factor), int(self.img_size[1] * scale_factor)],
                        dtype='float32')
        for i in range(0, len(props)):
            index, pos, size, hasdir, dir, file = props[i]
            pos_x, pos_y = pos
            size_w, size_h = size

            for x in range(int(pos_x * scale_factor), int((pos_x + size_w) * scale_factor)):
                for y in range(int(pos_y * scale_factor), int((pos_y + size_h) * scale_factor)):
                    mask[i][0][y][x] = 1
                    if hasdir:
                        mask[i][1][y][x] = dir
                    else:
                        mask[i][1][y][x] = math.nan
        return mask

    def _create_cropped_mask(self, props):
        """
        Creates cropped mask images.
        :param props: name of props file
        :return: cropped mask
        """
        mask = np.zeros([len(props), 2, self.new_img_size[0], self.new_img_size[1]], dtype='float32')
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
                        else:
                            mask[i][1][y][x] = math.nan
                    except (IndexError):
                        pass
        return mask

    def _generate_images(self, file, overwrite=False, scale_factor=-1):
        """
        Creates input data from original images in compressed .npz file format.
        :param file: dataset name
        :param overwrite: whether to overwrite existing files
        :param scale_factor: scale factor
        """
        if not (overwrite) and os.path.exists(self.folder + file + "/" + file + "_input.npz"):
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

        if (len(ball_props) != len(beetle_props)):
            print("Error: The number of ball labels is diffent from the number of beetle labels.")
            return

        print("Generating input numpy array ...")

        if (scale_factor != -1):
            data = np.zeros(
                [len(ball_props), 1, int(self.img_size[0] * scale_factor), int(self.img_size[1] * scale_factor)],
                dtype='float32')
        else:
            data = np.zeros([len(ball_props), 1, self.new_img_size[0], self.new_img_size[1]], dtype='float32')
        for i in range(0, len(ball_props)):
            img_name = ball_props[i][5]
            if (img_name != beetle_props[i][5]):
                print("Error: Different image files.")
            scaled_img = cv2.imread(self.folder + file + "/" + file + "_imgs/" + img_name)
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
            if (scale_factor != -1):
                scaled_img = cv2.resize(scaled_img,
                                        (int(self.img_size[1] * scale_factor), int(self.img_size[0] * scale_factor)),
                                        interpolation=cv2.INTER_CUBIC)
            else:
                scaled_img = scaled_img[self.offset[i][1]:self.offset[i][1] + self.new_img_size[0],
                             self.offset[i][0]:self.offset[i][0] + self.new_img_size[1]]
            data[i, 0, :, :] = scaled_img

        print("Compressing and saving input numpy array ...")
        np.savez_compressed(self.folder + file + "/" + file + "_input", data=data)

    def set_img_size(self, img_size):
        """
        Sets class attribute img_size to passed tuple value.
        :param img_size: tuple (height, width)
        """
        self.img_size = (img_size[0], img_size[1])

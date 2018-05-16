#!/usr/bin/python
import numpy as np
import json
from jsonpath_ng.ext import parse
from pprint import pprint

Filename = "Allogymnopleuri_#01"

# Load Grinder file as JSON String
grndr_file = open("../../Dung_Beetle_Database/" + Filename + "/" +  Filename +"_db.grndr")
grndr_dict = json.load(grndr_file)

beetle_props = open(Filename + '_beetle_props.txt', 'a')
ball_props = open(Filename + '_ball_props.txt', 'a')

# Parse Beetle and Ball separately
beetle_parse = parse("$..Labels[?(@.Label.Name='Beetle')]")
ball_parse = parse("$..Labels[?(@.Label.Name='Ball')]")
beetle_labels = [match.value for match in beetle_parse.find(grndr_dict)]
ball_labels = [match.value for match in ball_parse.find(grndr_dict)]

labels=[("beetle", beetle_labels), ("ball", ball_labels)]

# Calculate total number of images in given video
beetle_len = parse("$..ImageReferences[*]")
num_imgs= len(beetle_len.find(grndr_dict))

for label in labels:
    for i in range(0, num_imgs):
        position_parse = parse("$..ImageBuilds[?(@..ImageReference=" + str(i) + ")] where $..Properties[?(@.Property.ID='Position')].Property.Value")
        size_parse = parse("$..ImageBuilds[?(@..ImageReference=" + str(i) + ")] where $..Properties[?(@.Property.ID='Size')].Property.Value")
        has_direction_parse = parse("$..ImageBuilds[?(@..ImageReference=" + str(i) + ")] where $..Properties[?(@.Property.ID='HasDirection')].Property.Value")
        direction_parse = parse("$..ImageBuilds[?(@..ImageReference=" + str(i) + ")] where $..Properties[?(@.Property.ID='Direction')].Property.Value")

        # Default values:
        position = (0, 0)
        size = (0, 0)
        has_direction = 'false'
        direction = 0

        if hasattr((position_parse.find(label[1]) or [None])[0], 'value'):
            position_raw = position_parse.find(label[1])[0].value
            x_value = ""
            y_value = ""
            for j in range(0, len(position_raw)//2):
                x_value += str(position_raw[j])
            for j in range(len(position_raw)//2+1, len(position_raw)):
                y_value += str(position_raw[j])
            position = (int(x_value), int(y_value))

        if hasattr((size_parse.find(label[1]) or [None])[0], 'value'):
            size_raw = size_parse.find(label[1])[0].value
            w_value = ""
            h_value = ""
            for j in range(0, len(size_raw) // 2):
                w_value += str(size_raw[j])
            for j in range(len(size_raw) // 2 + 1, len(size_raw)):
                h_value += str(size_raw[j])
            size = (int(w_value), int(h_value))

        if hasattr((has_direction_parse.find(label[1]) or [None])[0], 'value'):
            has_direction = has_direction_parse.find(label[1])[0].value

        if hasattr((direction_parse.find(label[1]) or [None])[0], 'value'):
            direction = int(direction_parse.find(label[1])[0].value)

        box_attrs = (i, position, size, has_direction, direction)
        if(label[0] == "beetle"):
            beetle_props.write(str(box_attrs) + "\n")
        else:
            ball_props.write(str(box_attrs) + "\n")




# for label in grndr_dict['Labels']:
#     if(label['Label']['Name'] == 'Ball'):
#         image_builds = label['Label']['ImageBuildPool'][0]['Item']['ImageBuilds']
#         for build in image_builds:
#             pprint(build['ImageBuild']['ImageReference'])
#             if build['ImageBuild']['Layers'] and build['ImageBuild']['Layers'][0]['Layer']['DraftItems']:
#                 props = build['ImageBuild']['Layers'][0]['Layer']['DraftItems'][0]['DraftItem']['Properties']
#                 for prop in props:
#                     if(prop['Property']['ID'] == 'Position'):
#                         pprint("Position: " + prop['Property']['Value'])
#                     elif(prop['Property']['ID'] == 'Size'):
#                         pprint("Size: " + prop['Property']['Value'])
#                     elif (prop['Property']['ID'] == 'HasDirection'):
#                         pprint("HasDirection: " + prop['Property']['Value'])
#                     elif (prop['Property']['ID'] == 'Direction'):
#                         pprint("Size: " + prop['Property']['Value'])

# TODO:
# identify every pixel inside the bounding box for every image
# create 2 binary images: (Beetle: white, bg: black) (Ball: white, bg: black)
# save images as numpy array
from PIL import Image
import numpy as np
import os
import random

def random_no_repeat(numbers):
    number_list = list(numbers)
    random.shuffle(number_list)
    return number_list

# Params
dataset_folder = "../data/artificial"
img_folder = dataset_folder + "/artificial_imgs"

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

num_images = 360

pos_x = 940
pos_y = 505

box_height_full = 85
box_width_full = 80

box_height = 35
box_width = 40

# -------------------------

# Create images
empty_img = np.zeros((1080, 1920))
img = np.zeros((box_height_full, box_width_full))

# Create rectangle with front and back
box_white_t1 = np.array([255] * box_height * box_width_full).reshape(box_height, box_width_full)
box_white_t2 = np.array([255] * 50 * 26).reshape(50, 26)

# Replace original image with boxes
img[0:0 + box_height, 0:0 + box_width_full] = box_white_t1
img[box_height:box_height + 50, 27:27 + 26] = box_white_t2

beetle_props = open(dataset_folder + "/" + "artificial_beetle_props.txt", 'w')

data_beetle = []

random.seed(0)
angles = random_no_repeat(range(360))

for i in range(0, num_images):
    file_name = "img_" +str(i).zfill(5) + ".png"

    #random_angle = np.random.randint(0, 359)
    random
    box = Image.fromarray(img)
    rot = box.rotate(360-angles[i], expand=1)
    out = Image.fromarray(empty_img)
    out.paste(rot, (pos_x, pos_y))
    if out.mode != 'RGB':
        out = out.convert('RGB')
    out.save(img_folder + "/" + file_name, "PNG")

    data_beetle.append((i, (pos_x, pos_y), rot.size, True, (angles[i]+180) % 360, file_name))

# write data
for x in data_beetle:
    beetle_props.write(str(x) + "\n")







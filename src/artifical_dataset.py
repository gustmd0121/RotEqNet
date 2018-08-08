from PIL import Image
import numpy as np
import os

# Params
dataset_folder = "../data/artificial"
img_folder = dataset_folder + "/artifical_imgs"

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

num_images = 5

pos_x = 940
pos_y = 505

box_height_full = 70

box_height = 35
box_width = 40

# -------------------------

# Create images
empty_img = np.zeros((1080, 1920))
img = np.zeros((70, 40))

# Create rectangle with front and back
box_white = np.array([255] * box_height * box_width).reshape(box_height, box_width)
box_grey = np.array([128] * box_height * box_width).reshape(box_height, box_width)

# Replace original image with boxes
img[0:0 + box_height, 0:0 + box_width] = box_white
img[0+box_height:0+box_height + box_height, 0:0 + box_width] = box_grey

beetle_props = open(dataset_folder + "/" + "artificial_beetle_props.txt", 'w')

data_beetle = []

for i in range(0, num_images):
    file_name = "artifical_" + str(i) + ".png"

    random_angle = np.random.randint(0, 359)
    box = Image.fromarray(img)
    rot = box.rotate(random_angle, expand=1)
    out = Image.fromarray(empty_img)
    out.paste(rot, (pos_x, pos_y))
    if out.mode != 'RGB':
        out = out.convert('RGB')
    out.save(img_folder + "/" + file_name, "PNG")

    data_beetle.append((i, (pos_x, pos_y), rot.size, True, 360-random_angle, file_name))

# write data
for x in data_beetle:
    beetle_props.write(str(x) + "\n")




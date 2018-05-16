import os
import numpy as np
from PIL import Image
from ast import literal_eval

img_size = (1080, 1920)
Filename = "Allogymnopleuri_#05"
base_folder = "../../Dung_Beetle_Database/" + Filename + "/"

img_folder = base_folder + Filename + "_imgs/"
#mask_folder = base_folder + Filename + "_masks/"

beetle_props_file = Filename + "_beetle_props.txt"
ball_props_file = Filename + "_ball_props.txt"

def create_mask(props):
    mask = np.zeros([1080, 1920, len(props)])
    for i in range(0, len(props)):
        index, pos, size, hasdir, dir = props[i]
        pos_x, pos_y = pos
        size_w, size_h = size

        for x in range(pos_x, pos_x + size_w):
            for y in range(pos_y, pos_y + size_h):
                mask[y][x][i] = 1
    return mask

with open(beetle_props_file) as f:
    beetle_props = [literal_eval(line) for line in f.readlines()]

with open(ball_props_file) as f:
    ball_props = [literal_eval(line) for line in f.readlines()]

beetle_masks = create_mask(beetle_props)
ball_masks = create_mask(ball_props)

#if not os.path.exists(mask_folder):
#    os.mkdir(mask_folder)

np.savez_compressed(base_folder + Filename + "_masks", beetle=beetle_masks, ball=ball_masks)

loaded = np.load(base_folder + Filename + "_masks.npz")

# img = Image.fromarray(loaded['beetle'][:, :, 20])
# img2 = Image.fromarray(loaded['ball'][:, :, 20])
# img.show()
# img2.show()
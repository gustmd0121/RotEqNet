from src.parser import Parser
import os

data_folder = "./data/"

p = Parser(data_folder)
for sub_folder in os.listdir(data_folder):
    sub_path = os.path.join(data_folder, sub_folder)
    if os.path.isdir(sub_path):
        p.create_cropped_files(sub_folder, (300, 400))
        # p.create_scaled_files(sub_folder, 0.5)
        # try to open input
        # p.load_numpy(sub_folder, 100)
print("Finished.")

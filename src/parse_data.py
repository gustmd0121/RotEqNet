from parser import Parser
import os

data_folder = "./data/"
if not os.path.isdir(data_folder):
    data_folder = "." + data_folder

p = Parser(data_folder, (1080, 1920))
for sub_folder in os.listdir(data_folder):
    sub_path = os.path.join(data_folder, sub_folder)
    if os.path.isdir(sub_path):
        print("Parsing", sub_folder, "...")
        p.parse(sub_folder)
        p.calc_offset(sub_folder, (300, 400), "ball")
        p.create_numpy_arrays(sub_folder, True)
        p.generate_input(sub_folder, True)
        print("Finished", sub_folder + ".")
        print("")
print("Finished.")

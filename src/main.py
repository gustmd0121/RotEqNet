from parser2 import Parser
import os

# file = "Allogymnopleuri_#05"
data_folder = "../data/"

# p = Parser(data_folder)
# p.parse(file)
# p.create_numpy_arrays(file, scale_factor)
# p.generate_input(file, scale_factor)
# print("Finished.")
# p.load_image(file, 100)

p = Parser(data_folder, (1080, 1920))
for sub_folder in os.listdir(data_folder):
    sub_path = os.path.join(data_folder, sub_folder)
    if os.path.isdir(sub_path):
        # print("Parsing", sub_folder, "...")
        p.parse(sub_folder)
        p.calc_offset(sub_folder, (540, 960), "ball")
        p.create_numpy_arrays(sub_folder, override=True)
        p.generate_input(sub_folder, override=True)
        # print("Finished", sub_folder + ".")
        # print("")
        p.load_image(sub_folder, 80)
print("Finished.")

from parser import Parser
import os

# file = "Allogymnopleuri_#09"
data_folder = "../data/"
scale_factor = 0.5

# p = Parser(data_folder)
# p.parse(file)
# p.create_numpy_arrays(file, scale_factor)
# p.generate_input(file, scale_factor)
# print("Finished.")
# p.load_image(file, 100)

p = Parser(data_folder)
for sub_folder in os.listdir(data_folder):
    sub_path = os.path.join(data_folder, sub_folder)
    if os.path.isdir(sub_path):
        # print("Parsing", sub_folder, "...")
        # p.parse(sub_folder)
        # p.create_numpy_arrays(sub_folder, scale_factor, True, 1)
        # p.generate_input(sub_folder, scale_factor, True)
        # print("Finished", sub_folder + ".")
        # print("")
        p.load_image(sub_folder, 100)
print("Finished.")

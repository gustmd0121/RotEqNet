from parser import Parser
import os

data_folder = "./data/"
combined_folder = "combined"
if not os.path.isdir(data_folder):
    data_folder = "." + data_folder

p = Parser(data_folder, combined_folder)
# p.create_cropped_files("artificial", (300, 400), override=True)
for sub_folder in os.listdir(data_folder):
    sub_path = os.path.join(data_folder, sub_folder)
    if os.path.isdir(sub_path):
        p.create_cropped_files(sub_folder, (256, 256), override=True)
        # p.create_scaled_files(sub_folder, 0.5)
        # try to open input
        # p.load_numpy(sub_folder, 100)

# p.combine_numpy_arrays()
print("Finished.")

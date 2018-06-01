from parser import Parser

file = "Allogymnopleuri_#09"
data_folder = "../data/"
scale_factor = 0.5

p = Parser(data_folder)
p.parse(file)
p.create_numpy_arrays(file, scale_factor)
p.generate_input(file, scale_factor)
print("Finished.")

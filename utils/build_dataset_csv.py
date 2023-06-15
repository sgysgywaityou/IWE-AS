import numpy as np
import os


data_path = "./data/text/"
save_data_csv = "./data/data.csv"
file_list = os.listdir(data_path)
file_list = np.array([data_path + file for file in file_list]).reshape(-1, 1)
print(file_list)
np.savetxt(save_data_csv, file_list, fmt="%s")
import os
from glob import glob

cwd = os.getcwd()
data_dirs = ["raw_files","interim_files","processed_files"]

for data_dir in data_dirs:
    files = glob(data_dir + r"/*.csv")
    for filePath in files:
        # print(f"dvc add {filePath}")
        os.path.join(cwd,"data")
        os.system(f"dvc add {filePath}")

print("\n #### all files added to dvc ####")
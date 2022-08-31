import os
from glob import glob

import argparse
import pandas as pd

cwd = os.getcwd()
data_dirs = ["raw_files/clientname_data","interim_files","processed_files"]

for data_dir in data_dirs:
    files = glob(data_dir + r"/*.csv")
    for filePath in files:
        # print(f"dvc add {filePath}")
        os.path.join(cwd,"data")
        os.system(f"dvc add {filePath}")

print("\n #### all files added to dvc ####")

import os
import numpy as np
from scipy.io import loadmat

from tqdm import *

def get_mat_numpy(path):
    mat = loadmat(path)
    for key in mat.keys():
        if key not in ('__header__', '__version__', '__globals__'):
            return mat[key]


def wash_dataset(dataset_path):
    images = [[os.path.join(dataset_path + f"/receivedpower_{i}MHz_mat", fname) for fname in sorted(os.listdir(dataset_path + f"/receivedpower_{i}MHz_mat"))] for i in (1750,2750,3750,4750,5750)]
    buildings = [os.path.join(dataset_path + '/buildings_position', fname) for fname in sorted(os.listdir(dataset_path + '/buildings_position'))]
    
    x, y = 0, 0
    for i in tqdm(range(len(images[0]))):
        shapes = [get_mat_numpy(images[n][i]).shape for n in range(5)]
        if len(set(shapes)) != 1 or shapes[0][0] <= 256 or shapes[0][1] <= 256:
            print('存在坏点或过小的数据：', shapes)
            for n in range(5):
                os.remove(images[n][i])
            os.remove(buildings[i])
            y += 1
        else:
            x += 1

    print(f"数据集已清洗完成，删除了{y}条数据，现在还有{x}条数据")


if __name__ == "__main__":
    dataset_path = "/root/autodl-tmp/dataset/root/autodl-tmp"
    wash_dataset(dataset_path)
      
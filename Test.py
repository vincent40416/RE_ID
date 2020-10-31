import os
import pickle
import glob
import numpy as np
from PIL import Image
HR_train_data_path = './CelebA-HQ-img/*'
edge_data = "./CelebA-HQ-img_CannyEdge/"


# def abc(path):
#     return glob.glob(path)[18000:18100]
# a = abc(HR_train_data_path)
# b = abc(HR_train_data_path)
# for i in range(0,100):
#     if a[i] == b[i]:
#         print(i)
label = np.zeros(200)
label[10] = 1
print(label)

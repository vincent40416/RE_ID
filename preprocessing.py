import glob
import numpy as np
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
Market1501 = PATH + '/Market-1501-v15.09.15/gt_bbox/*.jpg'


batch_size = 64
input_channels = 3


class ImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.transform = transforms.Compose(transform)
        self.files = imgs
        self.labels = labels

    def __getitem__(self, index):
        img = Image.fromarray(self.files[index].astype('uint8'), 'RGB')

        img = self.transform(img)
        label = np.zeros(1502, dtype=float)
        label[int(self.labels[index])] = 1

        # label = self.transform(label)
        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.files)


def load_jpg(data_folder, batch_size=batch_size, start_point=0, end_point=100000, shuffle=True, train=False):

    # print(data_folder)
    image_list = []
    label_list = []
    print("start loading hr data")
    i = 0
    for filename in glob.glob(data_folder)[start_point:end_point]:
        # get image as np array
        im = Image.open(filename)

        im = np.array(im)
        image_list.append(im)
        # print(im.shape)
        # get the identity of the image
        label = filename.split("/")[5].split("_")[0]
        label_list.append(label)
        # print(label)
        i = i + 1
        if i % 5000 == 0:
            print(i)

        # if i % 256 == 0:
        #     print(i)
        #     break
    image = np.asarray(image_list)
    label = np.asarray(label_list)

    transform = [
        transforms.Resize((384, 128), Image.BICUBIC),
        transforms.ToTensor()
    ]

    train_loader = torch.utils.data.DataLoader(ImageDataset(image, labels=label, transform=transform),
                                               batch_size=batch_size, shuffle=shuffle)

    return train_loader


# dataloader = load_jpg(Market1501, start_point=0, end_point=16000, train=True)

# test_data_set = load_jpg(HR_train_data_path, parsing_data, start_point=17000, end_point=18000, shuffle=False)
# print(dataloader)
print("end")

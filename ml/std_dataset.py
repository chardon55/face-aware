import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import cv2 as cv
from pathlib import Path

from image_classification import classifications, destination_dir


dataset_type = '300x300'
color_channel = 3

input_shape = (150, 150, color_channel)


dataset_dirs = {
    "face": Path("../dataset/ml/faces_class/"),
    "eye": Path("../dataset/eyes_out/"),
    "nose": Path("../dataset/noses_out/"),
    "mouth": Path("../dataset/mouths_out/"),
}

metas = {
    "face": Path("../dataset/ml/faces_class/meta.csv"),
    "eye": Path("../dataset/eye_meta.csv"),
    "nose": Path("../dataset/nose_meta.csv"),
    "mouth": Path("../dataset/mouth_meta.csv"),
}

sizes = {
    "face": (150, 150),
    "eye": (50, 50),
    "nose": (60, 55),
    "mouth": (80, 50),
}


def read_image(number: int, data_source='face') -> np.ndarray:
    return cv.imread(str(dataset_dirs[data_source] / dataset_type / f"{number}.png"))


def read_meta(data_source='face'):
    return pd.read_csv(metas[data_source], index_col='identity')


def preprocess(image: np.ndarray, source="face") -> np.ndarray:
    return tf.image.resize(image, sizes[source])


def label_convert(label):
    if label == 0 or label == 1:
        return 0
    elif label == 2 or label == 3:
        return 1
    else:
        return 1


def read_dataset(source="face"):
    raw = read_meta(source).sample(frac=1)
    data = raw[['like']]

    images = []

    for i in data.index:
        print(f"\rLoading image {i}...", end='')
        images.append(preprocess(read_image(i), source))

    images = np.array(images)
    print("\rLoading image done          ")

    return images, data['like'].values


def main():
    dataset = read_meta()

    like_count = dataset[['like']].value_counts()
    print(like_count)

    fig, _ = plt.subplots()
    like_count.plot.pie(autopct='%1.1f%%')
    plt.ylabel("")
    plt.title('Like ratio')
    plt.show()
    fig.savefig("../dataset_like.png")


if __name__ == '__main__':
    main()

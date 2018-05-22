import tensorflow as tf
import cv2  # resize images
import numpy as np  # arrays == ordered series
import os  # directories
from random import shuffle
from tqdm import tqdm  # percentage bar for tasks

# PART ONE: PREPROCESS DATA

# 1, Model details
TRAIN_DIR = "/Users/chester/github/Catdog/train"
TEST_DIR = "/Users/chester/github/Catdog/test"
IMG_SIZE = 50
LR = 1e-3  # what's this?

MODEL_NAME = "dogsvscats={}-{}.model".format(LR, "2conv-basic")
# to help remember which model is which

# 2, Convert images and labels (cat.1) to Array


def label_img(img):
    word_label = img.split(".")[-3]  # why -3? because c-a-t
    if word_label == "cat":
        return [1, 0]
    elif word_label == "dog":
        return [0, 1]

# 3, Process training info to Arrays


def create_train_data():
    training_data = []
    # for img in TEST_DIR:
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)  # 1 label the image
        path = os.path.join(TRAIN_DIR, img)  # 2 path of the TRAIN_DIR
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 3 cv2 to grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 4 cv2 to resize 50, 50
        # training_data.append(img)
        training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)  # shuffle modifies variable in place
        np.save("train_data.npy", training_data)
        return training_data
        # we will both save, and return the array data. This way
        # if we just change the neural network's structure
        # and not something with the images, like image size..etc
        # then we can just load the array file and save some processing time
        # if we dont , then everytime change NN  it will reprocess images


def process_test_data(img):
    testing_data = []
    path = (os.path.join(TEST_DIR), img)
    img_num = img.split(".")[0]  # not sure why we add a number afterward
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    testing_data.append([np.array(img)], img_num)
    shuffle(testing_data)
    np.save("test_data.npy", testing_data)
    return testing_data


train_data = create_train_data()

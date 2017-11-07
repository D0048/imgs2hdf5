#!/usr/bin/env python3

import tflearn
import cv2
"""
tflearn.data_utils.build_hdf5_image_dataset(
    target_path='./subclass_dummy',image_shape= [256, 256],
    output_path='./dataset.h5',
    mode='folder',
    categorical_labels=True,
    normalize=True,
    grayscale=False,
    files_extension=None,
    chunks=False)
"""
from random import shuffle
import glob
shuffle_data = False  # shuffle the addresses before saving
hdf5_path = './dataset.hdf5'  # address to where you want to save the hdf5 file
cat_dog_train_path = './subclass_dummy/*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:len(addrs)]
train_labels = labels[0:len(labels)]
"""
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]
val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]
"""

import numpy as np
import h5py

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
if data_order == 'tf':
    train_shape = (len(train_addrs), 256, 256, 3)
    #val_shape = (len(val_addrs), 224, 224, 3)
    #test_shape = (len(test_addrs), 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int8)
#hdf5_file.create_dataset("val_img", val_shape, np.int8)
#hdf5_file.create_dataset("test_img", test_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_addrs), ), np.int8)
hdf5_file["train_labels"][...] = train_labels
#hdf5_file.create_dataset("val_labels", (len(val_addrs), ), np.int8)
#hdf5_file["val_labels"][...] = val_labels
#hdf5_file.create_dataset("test_labels", (len(test_addrs), ), np.int8)
#hdf5_file["test_labels"][...] = test_labels

# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))
    pass
"""
# loop over validation addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Validation data: {}/{}'.format(i, len(val_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["val_img"][i, ...] = img[None]
# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Test data: {}/{}'.format(i, len(test_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]
"""
# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()

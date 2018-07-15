
# coding: utf-8

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
import pandas as pd
import skimage.transform
import imageio

def gen_batch_function(image_paths, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        #image_paths = glob(os.path.join(data_folder,'*.*'))
        labelsets = pd.read_csv("labels.csv",delimiter="|")
        labelsets = labelsets.iloc[:].values
        labels_dict = {}
        for key,value in labelsets:
            labels_dict[key]=value;

        
        while True:
            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                labels = []
                for image_file in image_paths[batch_i:batch_i+batch_size]:

                    image = skimage.transform.resize(imageio.imread(image_file), image_shape)
                    label = labels_dict[image_file]
                    images.append(image)
                    labels.append(label)
                yield np.array(images), np.array(labels)
    return get_batches_fn
#gen_train_batch = gen_batch_function("dataset",(819,460))
#train_gen = gen_train_batch(2)
#for x,y in train_gen:
#    print(y)
def generate_splits(image_paths, split_fractions):
    start_index = 0
    splits = []

    for i in range(len(split_fractions) - 1):
        end_index = int(split_fractions[i] * len(image_paths))
        splits.append(image_paths[start_index:start_index + end_index])
        start_index += end_index
    splits.append(image_paths[start_index:])
    return splits


image_paths = glob(os.path.join('dataset','*.*'))
labelsets = pd.read_csv("labels.csv",delimiter="|")
labelsets = labelsets.iloc[:].values
labels_dict = {}
for key,value in labelsets:
    labels_dict[key]=value;


train, val, test = generate_splits(image_paths, [0.5,0.33,0.16])

def load_model(sess, model_path, tensor_names,model_tag):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
#    vgg_tag = model_tag
#    vgg_input_tensor_name = 'image_input:0'
#    vgg_keep_prob_tensor_name = 'keep_prob:0'
#    vgg_layer3_out_tensor_name = 'layer3_out:0'
#    vgg_layer4_out_tensor_name = 'layer4_out:0'
#    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [model_tag], model_path)
    model_graph = tf.get_default_graph()
    tensors=[]
    for tensor_name in tensor_names:
        tensors.append(model_graph.get_tensor_by_name(tensor_name+":0"))
    return tensors
        
#tests.test_load_vgg(load_vgg, tf
def layers():
    # TODO: Implementing function
    # This Funciton is for removing the last layers from loaded model
    pass

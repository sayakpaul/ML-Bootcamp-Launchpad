# imports
import tensorflow as tf
import numpy as np
import re

# function to read the TFRecords, segregate the images and labels
def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "class": tf.io.FixedLenFeature([], tf.int64)
    }
    
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0  
    image = tf.reshape(image, [224, 224, 3]) 
    class_label = tf.cast(example['class'], tf.int32)
    
    return (image, class_label)

# a bit of data augmentation
def data_augment(image, class_label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0, 2)
    return (image, class_label)

# load the TFRecords and create tf.data.Dataset
def load_dataset(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=16) 
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset

# we need this to derive the steps for generator
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# batch, shuffle, and repeat the dataset and pre-fetch it
# well before the current epoch ends
def batch_dataset(filenames, batch_size, train):
    dataset = load_dataset(filenames)
    n = count_data_items(filenames)
    
    if train:
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(1000)
    else:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) 
    return (dataset, n//batch_size)
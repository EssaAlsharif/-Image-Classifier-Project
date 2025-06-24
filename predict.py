
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import json

from PIL import Image

def get_input_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str,
                        help='path to image')
    parser.add_argument('model', type=str,
                        help='path to model')
    parser.add_argument('--top_k', type=int, default = 5,
                        help='the top K most likely classes')
    parser.add_argument('--category_names', type=str,
                        help='Path to a JSON file mapping labels to flower names')
    return parser.parse_args()

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image /= 255
    image = image.numpy()
    return image

def predict (image_path,model_path,top_k,category_names=None):
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer': hub.KerasLayer})
    im = Image.open(image_path)
    required_image = np.asarray(im)

    processed_image = process_image(required_image)
    extended_image = np.expand_dims(processed_image, axis=0)

    ps = model.predict(extended_image)
    classes = np.argsort(ps[0])[-top_k:][::-1]
    probs = ps[0][classes]

    if category_names:
      with open(category_names, 'r') as f:
          class_names = json.load(f)
      flower_names = [class_names[str(i)] for i in classes]
    else:
        flower_names = [str(i) for i in classes]

    for i in range(top_k):
        print(f"{flower_names[i]} with probability of {probs[i]*100:.2f}%")

    return probs, classes

if __name__ == '__main__':
    in_arg = get_input_args()
    predict(in_arg.image,in_arg.model,in_arg.top_k,in_arg.category_names)


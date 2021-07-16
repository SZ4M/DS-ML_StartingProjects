import tensorflow as tf
import numpy as np
from PIL import Image
import time
import json
import sys
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
import sys
import argparse


def load_model(path):
    model = tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    model.build((None, 224, 224, 3))
    return model


def process_image(imageNp):
    #     Convert to TF
    imageNp = tf.cast(imageNp, tf.float32)
    imageNp = tf.image.resize(imageNp, (224, 224))
    imageNp /= 255
    #     Convert to NP
    imageNp = imageNp.numpy()
    return imageNp


def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)

    probs = model.predict(processed_test_image)
    probs = probs[0].tolist()
    prob, i = tf.math.top_k(probs, k=top_k)
    probs = prob.numpy().tolist()
    classes = i.numpy().tolist()
    return probs, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Add Image Path', type=str)
    model_path = parser.add_argument('model', default='./my_model.h5', type=str, help='Add model Path')
    parser.add_argument('--top_k', type=int, help='Top K Value', default=5)
    parser.add_argument('--category_names', type=str)
    #     labels = './label_map.json'
    #     model_path = './my_model.h5'
    args = parser.parse_args()
    print(args)

    image_path = args.image_path
    model = load_model(args.model)
    top_k = args.top_k
    labels = args.category_names
    if top_k is None:
        top_k = 5
    if labels is None:
        labels = './label_map.json'
    probs, classes = predict(image_path, model, top_k)
    with open(labels, 'r')as f:
        class_names = json.load(f)

    for i in range(top_k):
        print(class_names[str(classes[i] + 1)], ': ', probs[i])
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tf_keras as keras
from tf_keras import regularizers
from tf_keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image
from tf_keras import mixed_precision
import csv
import tensorflow_hub as hub
import dataprocessing

script_dir = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths so the script works when run from any cwd
model_path = os.path.join(script_dir, "ASLModel")
wnids_path = os.path.join(script_dir, "wnids.txt")
test_images_path = os.path.join(script_dir, "otherdataset", "test_images")

model = keras.models.load_model(model_path)
model.summary()
label_dict = {}
for i, line in enumerate(open(wnids_path, "r")):
    label_dict[line.rstrip("\n")] = int(i)

test_labels = dataprocessing.generate_test_labels(test_images_path)
test_int = [label_dict[x[0]] for x in test_labels]

batch_size = 100
img_size = 224

pred = dataprocessing.tta_prediction(model,
                                     batch_size,
                                     img_size,
                                     normalize=False,
                                     test_location=test_images_path)
y_predict_max = np.argmax(pred, axis=1)

print(np.asarray(test_int))
print(y_predict_max)

total = 0.0
correct = 0.0
for x, y in zip(test_int, y_predict_max):
    total += 1
    if x == y:
        correct += 1

accuracy = correct / total
print("Accuracy: " + str(accuracy))

# Use the train_images directory from otherdataset for evaluation if you
# don't have a separate Validation/ folder.
train_images_path = os.path.join(script_dir, "otherdataset", "train_images")
more_data, _ = dataprocessing.generate_nonaugmented_images(
    batch_size,
    img_size,
    normalize=False,
    valsplit=False,
    train_location=train_images_path,
    class_mode="categorical")
model.evaluate(more_data)
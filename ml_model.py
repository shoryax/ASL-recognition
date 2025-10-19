import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tf_keras as keras
from tf_keras import mixed_precision, regularizers
from tf_keras.optimizers import legacy as optim_legacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image
import csv
import tensorflow_hub as hub
import dataprocessing


def run():
    label_dict = {}
    for i, line in enumerate(open("/home/shoryax/asl/wnids.txt", "r")):
        label_dict[line.rstrip("\n")] = int(i)

    batch_size = 64
    img_size = 224
    num_classes = 26

    # Resolve dataset paths relative to this script's directory so the script
    # works regardless of the current working directory when invoked.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.join(script_dir, "otherdataset")

    ### PARSING TRAIN/VALIDATION FILES
    train_location = os.path.join(dataset_base, "train_images")
    test_images_location = os.path.join(dataset_base, "test_images")

    train_data, val_data = dataprocessing.generate_augmented_images(
        batch_size,
        img_size,
        normalize=False,
        train_location=train_location)

    ### PARSING TEST IMAGES
    test_labels = dataprocessing.generate_test_labels(
        test_location=test_images_location)
    test_int = [label_dict[x[0]] for x in test_labels]

    mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_model = mobilenet_v2

    feature_extractor_layer = hub.KerasLayer(feature_extractor_model,
                                             input_shape=(img_size, img_size,3),
                                             trainable=True)

    ### Optimized Neural Network
    model = keras.models.Sequential()

    # Model Layers
    model.add(keras.layers.Rescaling((1. / 255)))
    model.add(feature_extractor_layer)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='swish'))
    model.add(keras.layers.Dense(256, activation='swish'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.build(input_shape=(None, img_size, img_size, 3))
    model.summary()
    model.compile(optimizer=optim_legacy.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    num_epochs = 100
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=4,
                                                     verbose=1,
                                                     factor=0.4,
                                                     min_lr=0.0001)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.00001,
                                               patience=8,
                                               mode='auto',
                                               restore_best_weights=True)

    history = model.fit(train_data,
                        workers=16,
                        epochs=num_epochs,
                        validation_data=val_data,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stop, lr_reduction],
                        max_queue_size=30)

    print("Final Validation Accuracy:")
    model.evaluate(val_data)
    model.save("ASLModel")

    ##Matching Predictions with Correct Image ID
    pred = dataprocessing.tta_prediction(
        model,
        batch_size,
        img_size,
        data_format="channels_last",
        normalize=False,
        test_location="otherdataset/test_images")
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

    print("Test Accuracy: " + str(accuracy))
    '''
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, 100])
    plt.ylim([0.1, 1.0])
    plt.legend(loc='lower right')
    plt.show()
    '''
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "ASLModel")  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
run()
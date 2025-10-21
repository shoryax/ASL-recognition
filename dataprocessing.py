import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tf_keras as keras
from tf_keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import numpy as np
import pandas as pd


def generate_augmented_images(batch_size,
                              img_size,
                              normalize=True,
                              data_format="channels_last",
                              train_location="dataset/train_images/",
                              valsplit=True):

    if (normalize == True):
        aug_gens = ImageDataGenerator(
            rescale=1.0 / 255,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=15,      # Was 10
            shear_range=0.20,       # Was 0.15
            zoom_range=0.15,        # Was 0.1
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
            brightness_range=(0.9, 1.1),
            fill_mode="constant",
        )
    else:
        aug_gens = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=15,      # Was 10
            shear_range=0.20,       # Was 0.15
            zoom_range=0.15,        # Was 0.1
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
            brightness_range=(0.9, 1.1),
            fill_mode="constant",
        )
    if (valsplit):

        train_data = aug_gens.flow_from_directory(train_location,
                                                  subset="training",
                                                  seed=1447,
                                                  target_size=(img_size,
                                                               img_size),
                                                  batch_size=batch_size,
                                                  class_mode="categorical")

        val_data = aug_gens.flow_from_directory(train_location,
                                                subset="validation",
                                                seed=1447,
                                                target_size=(img_size,
                                                             img_size),
                                                batch_size=batch_size,
                                                class_mode="categorical")
    else:
        train_data = aug_gens.flow_from_directory(train_location,
                                                  subset=None,
                                                  seed=1447,
                                                  target_size=(img_size,
                                                               img_size),
                                                  batch_size=batch_size,
                                                  class_mode="categorical")

        val_data = None

    return train_data, val_data


def generate_nonaugmented_images(batch_size,
                                 img_size,
                                 normalize=True,
                                 train_location="dataset/train_images/",
                                 valsplit=True,
                                 class_mode="categorical"):

    if (normalize == True):
        aug_gens = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.1,
        )
    else:
        aug_gens = ImageDataGenerator(validation_split=0.1)
    if (valsplit):
        train_data = aug_gens.flow_from_directory(train_location,
                                                  subset="training",
                                                  seed=1447,
                                                  target_size=(img_size,
                                                               img_size),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode)

        val_data = aug_gens.flow_from_directory(train_location,
                                                subset="validation",
                                                seed=1447,
                                                target_size=(img_size,
                                                             img_size),
                                                batch_size=batch_size,
                                                class_mode=class_mode)
    else:
        train_data = aug_gens.flow_from_directory(train_location,
                                                  subset=None,
                                                  seed=1447,
                                                  target_size=(img_size,
                                                               img_size),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode)
        val_data = None
    return train_data, val_data


def generate_test_labels(test_location="dataset/test_images"):

    img_id = []
    for fileName in sorted(os.listdir(test_location)):
        if not fileName.startswith('.'):  # Skip hidden files
            img_id.append(fileName)
    return img_id


def tta_prediction(model,
                   batch_size,
                   img_size,
                   normalize=True,
                   data_format='channels_last',
                   test_location="dataset/test_images"):

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    filepaths = sorted([
        os.path.join(test_location, fname) for fname in os.listdir(test_location)
        if fname.lower().endswith(valid_exts)
    ])

    if not filepaths:
        raise ValueError(f"No image files found under {test_location}.")

    file_df = pd.DataFrame({"filepath": filepaths})

    if (normalize == True):
        aug_gens = ImageDataGenerator(
            rescale=1.0 / 255,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=15,      # Was 10
            shear_range=0.20,       # Was 0.15
            zoom_range=0.15,        # Was 0.1
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
            fill_mode="constant",
        )
    else:
        aug_gens = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=15,      # Was 10
            shear_range=0.20,       # Was 0.15
            zoom_range=0.15,        # Was 0.1
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
            fill_mode="constant",
        )

    tta_steps = 10
    predictions = []
    for i in tqdm(range(tta_steps)):
        preds = model.predict(aug_gens.flow_from_dataframe(
            dataframe=file_df,
            directory=None,
            x_col="filepath",
            y_col=None,
            class_mode=None,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False,
        ),
                              workers=16)
        predictions.append(preds)

    final_pred = np.mean(predictions, axis=0)
    return final_pred

# This does not work
def get_data(batch_size, img_size):
    traindf = pd.read_csv("otherdataset/train/_annotations.csv", dtype=str)
    valdf = pd.read_csv("otherdataset/valid/_annotations.csv", dtype=str)
    testdf = pd.read_csv("otherdataset/test/_annotations.csv", dtype=str)

    aug_gens = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,      # Was 10
        shear_range=0.20,       # Was 0.15
        zoom_range=0.15,        # Was 0.1
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
        data_format='channels_first',
        fill_mode='nearest',
    )

    train = aug_gens.flow_from_dataframe(dataframe=traindf,
                                         directory="otherdataset/train/",
                                         x_col='filename',
                                         y_col='class',
                                         batch_size=batch_size,
                                         seed=1447,
                                         class_mode='categorical',
                                         target_size=(img_size, img_size))

    valid = aug_gens.flow_from_dataframe(dataframe=valdf,
                                         directory="otherdataset/valid/",
                                         x_col='filename',
                                         y_col='class',
                                         batch_size=batch_size,
                                         seed=1447,
                                         class_mode='categorical',
                                         target_size=(img_size, img_size))

    test = aug_gens.flow_from_dataframe(dataframe=testdf,
                                        directory="otherdataset/test/",
                                        x_col='filename',
                                        y_col='class',
                                        batch_size=batch_size,
                                        seed=1447,
                                        class_mode='categorical',
                                        target_size=(img_size, img_size))

    return train, valid, test
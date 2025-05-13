import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
from numpy import mean, std
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.rc('axes', labelsize=16)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

dir = "../dogbreed_dataset"
train_paths = glob('../dogbreed_dataset/train/*.jpg')
test_paths = glob('../dogbreed_dataset/test/*.jpg')
labels_path = os.path.join(dir, "labels.csv")

labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique()
print("Number of Breed:", len(breed))

labels = []
for image_id in train_paths:
    image_id = image_id.split("\\")[-1].split(".")[0]  # Mac/Linux: "/"로 변경 필요
    breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
    labels.append(breed_name)

train_x, valid_x = train_test_split(train_paths, test_size=0.2, random_state=42)
train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

classes_name = labels
unique_classes = np.unique(classes_name, return_counts=True)
print(unique_classes)

plt.bar(unique_classes[0], unique_classes[1])
plt.xticks(rotation=45)
plt.show()

train_df = pd.DataFrame({'path': train_x, 'class_name': train_y})
test_df = pd.DataFrame({'path': valid_x, 'class_name': valid_y})
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

learning_rate = 1e-3
img_size = 224
input_shape = (img_size, img_size, 3)
num_classes = len(breed)
BATCH_SIZE = 64
initial_epochs = 10
fine_tune_epochs = 5
N_ch = 3
INIT_LR = 1e-3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='class_name',
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=12345,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='class_name',
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

inputs = Input((img_size, img_size, 3))
base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")

def build_model(size, num_classes):
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(200, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    return model

model = build_model(img_size, num_classes)
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate),
    metrics=["acc"]
)
model.summary()

scores, histories = list(), list()
annealer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1, min_lr=INIT_LR / 10)
checkpoint = ModelCheckpoint('models/dog_breed_model.keras', verbose=1, save_best_only=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=initial_epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[annealer, checkpoint]
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Accuracy 시각화
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1])
plt.title('Training and Validation Accuracy')

# Loss 시각화
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylim([0, 1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

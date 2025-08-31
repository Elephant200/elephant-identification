import numpy as np
import tensorflow as tf
import keras
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

print("Loading model...")
model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# for i, layer in enumerate(model.layers):
#     print(f"{i + 1}. {' ' * (5 - len(str(i + 1)))}{layer.name:<25}", end="")
#     try:
#         part_model = keras.Model(inputs=model.input, outputs=layer.output)
#         print(f" -> {str(part_model(np.zeros((1, 224, 224, 3))).shape):<20}", end="")
#     except Exception as e:
#         print(f"{"Couldn't get shape":<20}", end="")
#     print(f"(Type: {str(type(layer).__name__)})", end="")
#     print()
# all_layers = [layer.name for layer in model.layers]
# print(f"Number of layers: {len(all_layers)}")
print("Loaded model.")

root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

data = pd.read_csv(f"{root_dir}/dataset/train.csv")

conv4_layer = model.get_layer(name='conv3_block4_2_relu')

feature_extractor = keras.Model(inputs=model.input, outputs=conv4_layer.output)

svc = SVC(kernel='linear')

for index, row in data.iterrows():
    image_path = row["filepath"]
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = feature_extractor(image)

    print(features.shape)

    break

# X_train = features
# y_train = class mapping


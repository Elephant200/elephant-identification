import numpy as np
import keras
from keras.applications import ResNet50

layer_name = 'conv3_block4_2_relu'

model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
target_layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=target_layer.output)
print(feature_extractor.layers)

for i, layer in enumerate(feature_extractor.layers):
    print(f"{i + 1}. {' ' * (5 - len(str(i + 1)))}{layer.name:<25}", end="")
    try:
        part_model = keras.Model(inputs=model.input, outputs=layer.output)
        print(f" -> {str(part_model(np.zeros((1, 224, 224, 3))).shape):<20}", end="")
    except Exception as e:
        print(f"{"Couldn't get shape":<20}", end="")
    print(f"(Type: {str(type(layer).__name__)})", end="")
    print()
all_layers = [layer.name for layer in model.layers]
print(f"Number of layers: {len(all_layers)}")
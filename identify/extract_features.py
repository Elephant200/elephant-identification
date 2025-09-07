import numpy as np
import tensorflow as tf
import keras
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
import os
import pandas as pd
import pickle
import json
import argparse
from typing import List, Tuple, Dict
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess an image for ResNet50."""
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)


def extract_raw_features(data_df: pd.DataFrame, feature_extractor: keras.Model, layer_name: str = 'conv3_block4_2_relu', cache_dir: str = 'train_cache', force_retrain: bool = False) -> Tuple[List[np.ndarray], List[str]]:
    """Extract and cache raw features from images."""
    cache_file = f"{cache_dir}/raw_features_{layer_name}.pkl"

    if os.path.exists(cache_file) and not force_retrain:
        print(f"Loading cached raw features from {cache_file}")
        return pickle.load(open(cache_file, 'rb'))

    if force_retrain and os.path.exists(cache_file):
        print(f"Force retrain: Removing cached raw features from {cache_file}")
        os.remove(cache_file)
    

    features: List[np.ndarray] = []
    labels: List[str] = []

    print("Extracting raw features from images...")
    for _, row in data_df.iterrows():
        image = load_image(row['filepath'])
        feature_map = feature_extractor(image)
        features.append(feature_map.numpy().flatten())
        labels.append(str(row['name']))

    print(f"Saving raw features to {cache_file}")
    pickle.dump((features, labels), open(cache_file, 'wb'))
    return features, labels


def apply_pca(features: List[np.ndarray], n_components: int = 500, cache_dir: str = 'train_cache', layer_name: str = 'conv3_block4_2_relu', force_retrain: bool = False) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Apply PCA dimensionality reduction to features."""
    cache_file = f"{cache_dir}/pca_{n_components}_{layer_name}.pkl"

    if os.path.exists(cache_file) and not force_retrain:
        print(f"Loading cached PCA features from {cache_file}")
        return pickle.load(open(cache_file, 'rb'))

    if force_retrain and os.path.exists(cache_file):
        print(f"Force retrain: Removing cached PCA features from {cache_file}")
        os.remove(cache_file)

    print("Applying PCA dimensionality reduction...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Saving PCA features to {cache_file}")
    pickle.dump((X_pca, pca, scaler), open(cache_file, 'wb'))
    return X_pca, pca, scaler


def train_svm(X_train: np.ndarray, y_train: List[int], cache_dir: str = 'train_cache', layer_name: str = 'conv3_block4_2_relu', n_components: int = 500, force_retrain: bool = False) -> SVC:
    """Train SVM classifier on PCA features."""
    cache_file = f"{cache_dir}/svm_pca_{n_components}_{layer_name}.pkl"

    if os.path.exists(cache_file) and not force_retrain:
        print(f"Loading cached SVM from {cache_file}")
        return pickle.load(open(cache_file, 'rb'))

    if force_retrain and os.path.exists(cache_file):
        print(f"Force retrain: Removing cached SVM from {cache_file}")
        os.remove(cache_file)

    print("Training SVM classifier...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    print(f"Saving SVM to {cache_file}")
    pickle.dump(svm, open(cache_file, 'wb'))
    return svm


def predict_single_image(image_path: str, svm: SVC, pca: PCA, scaler: StandardScaler, feature_extractor: keras.Model, class_mapping: Dict[str, int]) -> Tuple[str, int]:
    """Predict elephant ID for a single image."""
    # Extract features
    image = load_image(image_path)
    feature_map = feature_extractor(image)
    raw_features = feature_map.numpy().flatten()

    # Apply PCA transformation
    features_scaled = scaler.transform([raw_features])
    features_pca = pca.transform(features_scaled)

    # Predict
    prediction_id = svm.predict(features_pca)[0]

    # Find the name corresponding to the predicted ID
    matching_names = [name for name, id_val in class_mapping.items() if id_val == prediction_id]

    if not matching_names:
        prediction_name = "UNKNOWN"
    else:
        prediction_name = matching_names[0]

    return prediction_name, prediction_id


def evaluate_on_set(dataset: pd.DataFrame, svm: SVC, pca: PCA, scaler: StandardScaler, feature_extractor: keras.Model, class_mapping: Dict[str, int]) -> float:
    """Evaluate SVM accuracy on dataset."""

    correct = 0
    total = len(dataset)

    for _, row in dataset.iterrows():
        predicted_name, _ = predict_single_image(
            row['filepath'], svm, pca, scaler, feature_extractor, class_mapping
        )
        if predicted_name == str(row['name']):
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate elephant identification model')
    parser.add_argument('--force', action='store_true', help='Force retrain by clearing all cached files')
    parser.add_argument('--force-pca', action='store_true', help='Force retrain by clearing all cached PCA files')
    parser.add_argument('--force-svm', action='store_true', help='Force retrain by clearing all cached SVM files')
    parser.add_argument('--n-components', type=int, default=350, help='Number of PCA components')
    args = parser.parse_args()

    print("Running single pipeline test...")
    if args.force:
        print("Force retrain: All cached files will be cleared and retrained")
    if args.force_pca:
        print("Force retrain: All cached PCA files will be cleared and retrained")
    if args.force_svm:
        print("Force retrain: All cached SVM files will be cleared and retrained")
    if args.n_components:
        print(f"Using {args.n_components} PCA components")

    # Training pipeline inline
    root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"
    train_data = pd.read_csv(f"{root_dir}/dataset/train.csv")

    # Load class mapping inline
    with open('dataset/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)

    layer_name = 'conv3_block4_2_relu'
    n_components = args.n_components # default 350

    print(f"Training pipeline: {layer_name} -> PCA({n_components}) -> SVM")

    # Create feature extractor
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    target_layer = model.get_layer(name=layer_name)
    max_pooling = keras.layers.MaxPooling2D(pool_size=(2, 2))(target_layer.output)
    feature_extractor = keras.Model(inputs=model.input, outputs=max_pooling)

    # Extract raw features
    raw_features, names = extract_raw_features(train_data, feature_extractor, layer_name, force_retrain=args.force)
    print(f"Raw features shape: {np.array(raw_features).shape}")

    # Apply PCA
    X_pca, pca, scaler = apply_pca(raw_features, n_components, layer_name=layer_name, force_retrain=args.force_pca or args.force)
    print(f"PCA features shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:10].sum():.3f}")

    # Convert names to class IDs
    y_train = [class_mapping[name] for name in names]

    # Train SVM
    svm = train_svm(X_pca, y_train, layer_name=layer_name, n_components=n_components, force_retrain=args.force_svm or args.force)
    print("SVM training completed")

    test_data = pd.read_csv(f"{root_dir}/dataset/test.csv")

    # print("Evaluating on train set...")
    # evaluate_on_set(train_data, svm, pca, scaler, feature_extractor, class_mapping)

    print("Evaluating on test set...")
    evaluate_on_set(test_data, svm, pca, scaler, feature_extractor, class_mapping)



# Accuracy: 0.268 with non-reflected
# Accuracy: 0.253 with reflected
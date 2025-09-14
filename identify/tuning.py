from identify.identify import run_and_evaluate
from utils import print_with_padding

if __name__ == "__main__":
    all_layer_names = ["conv3_block4_2_relu", "conv4_block6_2_relu", "conv5_block3_2_relu"]
    all_pool_sizes = [6]
    all_n_components = [1781]
    for layer_name in all_layer_names:
        for pool_size in all_pool_sizes:
            for n_components in all_n_components:
                print_with_padding(f"Tuning: ResNet50 until {layer_name} - Pool size {pool_size} - {n_components} PCA components")
                run_and_evaluate(
                    force_features=False,
                    force_pca=False,
                    force_svm=False,
                    batch_size=None,
                    layer_name=layer_name,
                    n_components=n_components,
                    pool_size=pool_size,
                    top_k_values=[1, 3, 5, 10]
                )
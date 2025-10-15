from identify.identify import run_and_evaluate
from utils import print_with_padding

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Tune the ResNet50 model for elephant identification'
    )
    parser.add_argument(
        '--optimal', 
        type=bool,
        help='Use the optimal layer and pool size'
    )
    args = parser.parse_args()
    if args.optimal:
        all_layer_names = ["conv4_block6_out"]
        all_pool_sizes = [6]
        all_n_components = [10000]
    else:
        all_layer_names = [
            "conv3_block4_2_relu",
            #"conv4_block6_2_relu",
            "conv4_block6_out", # 40th activation layer
            "conv5_block1_out", # 43rd activation layer
            #"conv5_block3_2_relu"
        ]
        all_pool_sizes = [1, 2, 4, 6]
        all_n_components = [10000] # Higher than needed for max possible PCA components

    accuracies = {}
    for layer_name in all_layer_names:
        for pool_size in all_pool_sizes:
            for n_components in all_n_components:
                print_with_padding(f"Tuning: ResNet50 until {layer_name} - Pool size {pool_size} - {n_components} PCA components")
                accuracies[f"{layer_name}_pool_{pool_size}_pca_{n_components}"] = run_and_evaluate(
                    force_features=False,
                    force_pca=False,
                    force_svm=False,
                    batch_size=None,
                    layer_name=layer_name,
                    n_components=n_components,
                    pool_size=pool_size,
                    top_k_values=[1, 3, 5, 10]
                )
    # Print as table with row for each test and column for each top_k
    print_with_padding("Tuning Results")
    print(f"| {'Test Name':<35} | {'Top 1':<20} | {'Top 3':<20} | {'Top 5':<20} | {'Top 10':<20} |")
    print(f"| {'-'*35} | {'-'*20} | {'-'*20} | {'-'*20} | {'-'*20} |")
    
    for test_name, test_accuracies in accuracies.items():
        print(f"| {test_name:<35} | ", end="")
        for _, accuracy in test_accuracies.items():
            print(f"{accuracy["accuracy"]*100:.1f}% ({accuracy["correct"]}/{accuracy["total"]})      | ", end="")
        print()

    print_with_padding("")
    
    print(f"| {'Test Name':<35} | {'Top 1':<10} | {'Top 3':<10} | {'Top 5':<10} | {'Top 10':<10} |")
    print(f"| {'-'*35} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} |")
    
    for test_name, test_accuracies in accuracies.items():
        print(f"| {test_name:<35} | ", end="")
        for _, accuracy in test_accuracies.items():
            print(f"{accuracy["accuracy"]*100:.1f}%      | ", end="")
        print()

    print_with_padding("")

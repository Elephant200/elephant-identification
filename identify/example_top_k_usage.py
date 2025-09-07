#!/usr/bin/env python3
"""Example usage of the top-k accuracy evaluation feature."""

def example_usage():
    """Demonstrate how to use the new top-k accuracy evaluation."""
    
    print("Example 1: Command line usage")
    print("-" * 50)
    print("# Evaluate with default top-k values (1, 3, 5, 10)")
    print("python -m identify.extract_features")
    print()
    print("# Evaluate with custom top-k values")
    print("python -m identify.extract_features --top-k 1 2 3 5 10")
    print()
    
    print("Example 2: Programmatic usage")
    print("-" * 50)
    print("""
from identify.extract_features import evaluate_on_set

# Evaluate with custom top-k values
accuracies = evaluate_on_set(
    test_data, 
    svm, 
    pca, 
    scaler, 
    feature_extractor, 
    class_mapping,
    batch_size=128,
    pool_size=2,
    top_k_values=[1, 3, 5, 10]  # Specify which k values to calculate
)

# Access results
print(f"Top-1 accuracy: {accuracies['top_1']:.3f}")
print(f"Top-3 accuracy: {accuracies['top_3']:.3f}")
print(f"Top-5 accuracy: {accuracies['top_5']:.3f}")
print(f"Top-10 accuracy: {accuracies['top_10']:.3f}")
""")
    
    print("\nExample 3: Understanding the results")
    print("-" * 50)
    print("""
The output will show accuracies for each k value:
- Top-1: The correct elephant is the model's first choice
- Top-3: The correct elephant is in the model's top 3 choices
- Top-5: The correct elephant is in the model's top 5 choices
- Top-10: The correct elephant is in the model's top 10 choices

Higher k values will always have equal or better accuracy than lower k values.
This helps understand how "close" the model is when it makes mistakes.
""")


if __name__ == "__main__":
    example_usage()
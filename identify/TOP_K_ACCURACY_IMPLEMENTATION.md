# Top-K Accuracy Implementation

## Overview
The elephant identification system has been enhanced to support top-k accuracy evaluation, allowing assessment of whether the correct elephant appears within the model's top k predictions.

## Key Changes

### 1. Modified `evaluate_on_set()` Function
- **Previous**: Returned single float accuracy value (top-1 only)
- **New**: Returns `Dict[str, float]` with accuracies for multiple k values
- Added `top_k_values` parameter (default: `[1, 3, 5]`)
- Uses `predict_proba()` instead of `predict()` to get probabilities for all classes
- Converts true names to IDs for efficient comparison

### 2. Updated `_evaluate_single_images()` Fallback Function
- Added support for top-k accuracy calculation
- Maintains compatibility with single-image processing
- Returns same dictionary format as batch evaluation

### 3. Implementation Details
```python
# Key logic for top-k accuracy:
# 1. Get prediction probabilities for all classes
prediction_probs = svm.predict_proba(features_pca)

# 2. Sort probabilities to get top-k predictions
top_k_predictions = np.argsort(prediction_probs, axis=1)[:, ::-1]

# 3. Convert true names to IDs
true_ids = [class_mapping.get(name, -1) for name in true_names]

# 4. Check if true ID is in top k predictions
for i, true_id in enumerate(true_ids):
    if true_id != -1 and true_id in top_k_predictions[i, :k]:
        correct += 1
```

### 4. Command Line Interface
Added `--top-k` argument:
```bash
# Default behavior (top 1, 3, 5, 10)
python -m identify.extract_features

# Custom k values
python -m identify.extract_features --top-k 1 2 3 5 10 20
```

### 5. Output Format
The evaluation now displays results for each k value:
```
Top-1 Accuracy: 0.268 (X/Y)
Top-3 Accuracy: 0.412 (X/Y)
Top-5 Accuracy: 0.523 (X/Y)
Top-10 Accuracy: 0.687 (X/Y)
```

## Benefits
1. **Better Model Understanding**: Shows how "close" incorrect predictions are
2. **Practical Applications**: In real-world use, showing top-3 or top-5 candidates can help human reviewers
3. **Performance Insights**: Identifies if the model is generally on the right track even when top-1 is wrong
4. **Unchanged Training**: No changes to model training - only evaluation is enhanced

## Usage Example
```python
from identify.extract_features import evaluate_on_set

# Evaluate with custom top-k values
accuracies = evaluate_on_set(
    test_data, svm, pca, scaler, feature_extractor, class_mapping,
    top_k_values=[1, 3, 5, 10, 20]
)

# Display results
for k_name, accuracy in accuracies.items():
    print(f"{k_name}: {accuracy:.3f}")
```
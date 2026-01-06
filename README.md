# Elephant Identification Research

This repository holds code used for research on identifying individual elephants from images. The project explores two complementary approaches to photo-identification: appearance-based deep learning and curvature-based ear recognition.

## Approaches

**Appearance-based identification** follows the baseline model from Körschens et al. (2018), using ResNet50 as a feature extractor, followed by PCA dimensionality reduction and SVM classification. Images are preprocessed using YOLOv11 for face detection and cropping. Future work will incorporate background removal techniques from Yu et al. (2024) to improve robustness.

**CurvRank** adapts the integral curvature methods developed by Weideman et al. (2017, 2020) for marine mammals to elephant ear identification. Ear contours are extracted using RF-DETR segmentation models, then processed to compute curvature descriptors along the curves. Local Naive Bayes Nearest Neighbor (LNBNN) matching identifies individuals by comparing these curvature signatures. Ears are processed separately (left and right) since they present different profiles.

## Dataset

Both methods are trained and evaluated on the ELPephants dataset (Körschens & Denzler, 2019), a fine-grained dataset designed for elephant re-identification research. Annotations for bounding box detection, keypoint extraction, and segmentation masks were created using Roboflow, with segmentation assisted by SAM3.

## Directory Structure

- `appearance/` - Deep learning identification pipeline (feature extraction, training, testing)
- `curvrank/` - Curvature-based identification using ear contours
- `data_prep/` - Scripts for dataset preparation, augmentation, and train/test splitting
- `dataset/` - Training and test data with metadata (not version controlled)
- `images/` - Source image collections from various datasets
- `processing/` - Preprocessing outputs for specific datasets
- `utils/` - Shared utilities for bounding boxes, contours, visualization, and file handling
- `archive/` - Earlier experimental code

## Background

Wildlife photo-identification helps researchers monitor populations without invasive tagging. While appearance-based deep learning has become dominant in computer vision, contour-based methods have proven effective for species with distinctive edge features. This project explores whether combining both approaches can improve identification accuracy for African elephants, particularly in challenging field conditions.

## Acknowledgements

This work builds on research in animal re-identification and utilizes several tools and datasets:

**Dataset**: Körschens, M., & Denzler, J. (2019). ELPephants: A Fine-Grained Dataset for Elephant Re-Identification. IEEE/CVF International Conference on Computer Vision Workshop (ICCVW).

**Appearance Model**: Körschens, M., Barz, B., & Denzler, J. (2018). Towards Automatic Identification of Elephants in the Wild. arXiv:1812.04418.

**Curvature Methods**: Weideman, H. J., Jablons, Z. M., Holmberg, J., et al. (2017). Integral Curvature Representation and Matching Algorithms for Identification of Dolphins and Whales. arXiv:1708.07785.

Weideman, H. J., Stewart, C. V., Parham, J. R., et al. (2020). Extracting identifying contours for African elephants and humpback whales using a learned appearance model. IEEE Winter Conference on Applications of Computer Vision (WACV).

**Background Removal**: Yu, Y., Vidit, V., Davydov, A., Engilberge, M., & Fua, P. (2024). Addressing the Elephant in the Room: Robust Animal Re-Identification with Unsupervised Part-Based Feature Alignment. arXiv:2405.13781.

**Computer Vision Tools**: Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11 (v11.0.0). Used for face detection, cropping, and keypoint extraction.

Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2025). RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. arXiv:2511.09554. Used for ear segmentation.

Carion, N., Gustafson, L., Hu, Y., et al. (2025). SAM 3: Segment Anything with Concepts. arXiv:2511.16719. Used for annotation assistance.

Annotation and model training workflows facilitated by Roboflow.

# Elephant Re-Identification using AI

![African Elephant](https://africageographic.com/wp-content/uploads/2020/01/Guest-Dr.jpg)

## Background

Reliable individual identification is fundamental to elephant conservation. Long-term population monitoring, tracking social dynamics, understanding movement patterns, and measuring the effectiveness of anti-poaching efforts all depend on our ability to recognize specific animals across time and space. Historically, this has meant either invasive tagging methods or manual identification by experts who memorize ear notches, tusk shapes, and vein patterns across hundreds of individuals. Neither scales well.

AI-based photo-identification offers a non-invasive alternative, and the proliferation of camera traps and citizen science platforms has created vast archives of elephant imagery. Manually processing these images remains difficult and time-consuming, but automated AI systems could transform how we conduct population surveys, allowing researchers to focus on analysis rather than image matching.

The challenge is that elephant re-identification is, well, challenging. Unlike human faces or car license plates, elephants are photographed at varying distances, angles, and lighting conditions. They're often partially occluded by vegetation, covered in mud, or facing the wrong way. Their appearance changes as they age, accumulate injuries, and lose tusks. Further, most identifying features are small and are oftne difficult to see, especially in low-quality images. Any practical system needs to handle this variability gracefully.

This project explores whether combining two complementary approaches can address these challenges better than either alone. Appearance-based deep learning excels at learning holistic visual patterns but can be sensitive to background clutter and pose variation. Contour-based methods focus on stable morphological features like ear shape, which remain consistent across conditions but require accurate segmentation. By developing both pipelines in parallel, we can evaluate their relative strengths and explore ensemble strategies.

## Approaches

**Appearance-based identification** follows the baseline model from Körschens et al. (2018), using ResNet50 as a feature extractor, followed by PCA dimensionality reduction and SVM classification. Images are preprocessed using YOLOv11 for face detection and cropping. Future work will incorporate background removal techniques from Yu et al. (2024) to improve robustness.

**CurvRank** adapts the integral curvature methods developed by Weideman et al. (2017, 2020) for marine mammals to elephant ear identification. Ear contours are extracted using RF-DETR segmentation models, then processed to compute curvature descriptors along the curves. Local Naive Bayes Nearest Neighbor (LNBNN) matching identifies individuals by comparing these curvature signatures. Ears are processed separately (left and right) since they present different profiles.

## Dataset

Both methods are trained and evaluated on the ELPephants dataset (Körschens & Denzler, 2019), a fine-grained dataset designed for elephant re-identification research. Annotations for bounding box detection, keypoint extraction, and segmentation masks were created using Roboflow, with segmentation assisted by SAM3.

**Current limitations**: The images currently available for training are not particularly high-quality, which constrains model performance. Access to larger, higher-resolution datasets from field researchers or conservation organizations would significantly improve accuracy and generalization. If you have access to elephant photos, please get in touch!

## Directory Structure

- `appearance/` - Deep learning identification pipeline (feature extraction, training, testing)
- `curvrank/` - Curvature-based identification using ear contours
- `data_prep/` - Scripts for dataset preparation, augmentation, and train/test splitting
- `dataset/` - Training and test data with metadata (not version controlled)
- `images/` - Source image collections from various datasets
- `processing/` - Preprocessing outputs for specific datasets
- `utils/` - Shared utilities for bounding boxes, contours, visualization, and file handling
- `archive/` - Earlier experimental code

## Project Status

This is an active research project. Both identification pipelines are functional but ongoing work focuses on improving robustness, testing ensemble methods, and evaluating performance across different image quality conditions. I have currently achieved 68% top-1 accuracy and 96.7% top-10 accuracy using the ResNet50 deep feature extractor.

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

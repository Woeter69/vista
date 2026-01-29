# Vista Hackathon To-Do List

## Immediate Actions
- [ ] **Data:** Locate and inspect the dataset (images and labels).
- [ ] **EDA:** Analyze class distribution and image difficulty (as suggested in overview).
- [ ] **Env:** Setup Python environment (PyTorch/TensorFlow, OpenCV, etc.).

## Modeling
- [x] **Baseline:** Create a simple object detection/classification pipeline to verify submission format. (Created `train.py` with FasterRCNN)
- [x] **Model Selection:** Choose an architecture suitable for "low to mid resource" (Selected FasterRCNN-ResNet50).
- [ ] **Training:** Train model on provided dataset. (Ready for Colab)

## Submission Pipeline
- [ ] **Post-Processing:** Implement the strict sorting logic (ascending categories, ascending image_ids).
- [ ] **Validation:** Create a local script to verify output format (JSON validity, sorting) before uploading.

## Optimization
- [ ] **Resource Check:** Evaluate inference time/model size to satisfy the "practicality" requirement.

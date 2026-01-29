# Dataset Analysis Report

## Summary
- **Training Set (`instances_train.json`):** 
  - **Images:** 53,739
  - **Type:** Single-object images (1 annotation per image).
  - **Use Case:** Learning object features/classification.
- **Test/Validation Set (`instances_test.json`):** 
  - **Images:** 18,000
  - **Type:** Multi-object scenes (Average 12.24 objects/image).
  - **Labels:** **Present** in the json file (useful for local validation).
  - **Difficulty:** Significantly harder than train set due to occlusion/clutter.
- **Categories:** 
  - **Total:** 200 distinct object categories.
  - Defined in `Categories.json`.

## Detailed Statistics

### Training Data
- **Count:** 53,739 images.
- **Objects per Image:** Always 1.
- **Class Balance:** Roughly balanced (Top classes ~640 images, others ~320-480).

### Test/Validation Data
- **Count:** 18,000 images.
- **Objects per Image:**
  - **Average:** 12.24
  - **Max:** 20
  - **Min:** 3
- **Total Annotations:** 220,360

## Key Insight for Modeling
The training data differs significantly from the test data distribution. 
- **Train:** Clean, single objects.
- **Test:** Dense, multi-object "checkout" scenarios.

**Recommendation:** 
- Use synthetic data generation (Mosaic/MixUp) during training to simulate multi-object scenes.
- Validate strictly on `instances_test.json` as it mimics the final challenge format.

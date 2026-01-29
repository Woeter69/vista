# Vista Hackathon - Codefest'26 (IIT BHU)

## Overview
**Goal:** Automate the checkout process in malls and supermarkets using Computer Vision.
**Task:** Identify objects placed on a table from images.
**Context:** Part of Codefest'26, aiming for practical low-to-mid resource solutions.
**Time Remaining:** ~2 days (as of Jan 29, 2026).

## Problem Statement
Develop a model to identify objects in images. The solution should be practical for real-world checkout scenarios, implying efficiency on low/mid-resource hardware is valued alongside raw accuracy.

## Evaluation
- **Primary Metric:** Average row score based on correct object identification.
- **Critical Constraint:** If the *count* of predicted objects differs from the ground truth count for an image, the score for that image is **0**.
- **Final Ranking:** Determined by a private leaderboard on hidden validation data. Ties broken by earlier submission time.
- **Qualitative:** Final judgments consider the practicality of the solution (resource usage).

## Submission Format
- **File Type:** CSV
- **Columns:** `image_id`, `categories`
    - `image_id`: Integer unique identifier.
    - `categories`: JSON-formatted string of predicted category IDs (e.g., `"[12,13,16]"`).

### Strict Sorting Requirements
1. **Category IDs:** Must be sorted in **ascending order** within the list (e.g., `[2, 4, 10]`).
2. **Rows:** Must be sorted by `image_id` in **ascending numerical order**.

### Important Notes
- Every `image_id` in the validation set must appear exactly once.
- Missing or duplicate `image_id`s will cause evaluation failure.
- Extra `image_id`s are ignored.

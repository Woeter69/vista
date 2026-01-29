# Gemini Project Context: Vista

## Project Identity
- **Name:** Vista (Computer Vision Hackathon)
- **Type:** Object Detection / Multi-Label Classification
- **Critical Goal:** Automate supermarket checkout.

## Key Constraints for Assistant
1. **Submission Strictness:** ALWAYS ensure any submission generation code enforces:
   - Ascending sort of Category IDs.
   - Ascending sort of Image IDs.
   - Valid JSON string format for categories.
2. **Scoring Logic:** Remind user that *count mismatch = 0 score*. The model must be precise in counting.
3. **Resource Awareness:** Prefer suggesting lightweight/efficient models (MobileNet, YOLOv8n, etc.) over massive transformers, unless accuracy demands otherwise, due to the "low to mid resource" judging criterion.

## User Preferences (Global)
- **Environment:** Linux.
- **Philosophy:** UX First, Optionality (from global context, keep in mind if building UI).
- **Editor:** VS Code (implied by file usage patterns, confirm if needed).

import os
import argparse
import json
import torch
import torchvision
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Get the best threshold from training if available, else default to 0.5
    threshold = checkpoint.get('best_t', 0.5)
    return model, threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--img-size', default=640, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Test Data Info
    test_json = None
    for r, d, f in os.walk(args.data_dir):
        if "instances_test.json" in f:
            test_json = os.path.join(r, "instances_test.json")
            img_dir = r
            break
    
    if not test_json:
        raise FileNotFoundError("Could not find instances_test.json")

    with open(test_json, 'r') as f:
        data = json.load(f)
    
    # Sort images by image_id (Ascending) as per instructions
    images = sorted(data['images'], key=lambda x: x['id'])
    
    # 2. Load Model
    model, threshold = get_model(201, args.model_path, device)
    print(f"Generating submission using threshold: {threshold}")

    results = []
    
    # 3. Inference
    with torch.no_grad():
        for img_info in tqdm(images, desc="Inference"):
            img_id = img_info['id']
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            # Use same loading as training for consistency
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.img_size, args.img_size))
            image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float().to(device)
            
            outputs = model([image_tensor])
            
            # Filter by optimized threshold
            p = outputs[0]
            mask = p['scores'] > threshold
            # Hardcoded mapping fix: we know IDs are 1-200
            preds = p['labels'][mask].detach().cpu().tolist()
            
            # STRICT REQUIREMENT: Sort categories in ascending order
            preds = sorted(list(preds))
            
            # Format as JSON string: "[1, 2, 3]"
            results.append({
                'image_id': img_id,
                'categories': json.dumps(preds).replace(" ", "")
            })

    # 4. Save CSV
    df = pd.DataFrame(results)
    # STRICT REQUIREMENT: Sort rows by image_id
    df = df.sort_values('image_id')
    df.to_csv(args.output, index=False)
    print(f"Final submission saved to {args.output}")

if __name__ == "__main__":
    main()

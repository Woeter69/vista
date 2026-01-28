import os
import argparse
import json
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# 1. Transforms (Removed forced resize to use default image sizes)
# -----------------------------------------------------------------------------
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transforms():
    return A.Compose([
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# -----------------------------------------------------------------------------
# 2. Dataset
# -----------------------------------------------------------------------------
class VistaDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        
        json_name = "instances_train.json" if split == 'train' else "instances_test.json"
        self.ann_file = self._find_file(root, json_name)
        if not self.ann_file: raise FileNotFoundError(f"Missing {json_name}")
        
        self.img_dir = self._find_dir(root, split) or os.path.dirname(self.ann_file)

        with open(self.ann_file, 'r') as f:
            data = json.load(f)
            
        self.ids = [img['id'] for img in data['images']]
        self.img_info = {img['id']: img for img in data['images']}
        self.img_to_anns = {}
        for ann in data['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

    def _find_file(self, path, name):
        for r, d, f in os.walk(path):
            if name in f: return os.path.join(r, name)
        return None

    def _find_dir(self, path, name):
        for r, d, f in os.walk(path):
            if name in d: return os.path.join(r, name)
        return None

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_data = self.img_info[img_id]
        path = os.path.join(self.img_dir, img_data['file_name'])
        
        image = np.array(Image.open(path).convert("RGB"))
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# -----------------------------------------------------------------------------
# 3. Metrics (Accuracy & Dice)
# -----------------------------------------------------------------------------
def calculate_metrics(preds, targets):
    total_dice, total_acc, count = 0, 0, 0
    for p, t in zip(preds, targets):
        if len(t['boxes']) == 0: continue
        if len(p['boxes']) == 0:
            count += 1
            continue
        
        iou = torchvision.ops.box_iou(p['boxes'], t['boxes'])
        max_iou, matched_idx = iou.max(dim=1)
        
        dice = (2 * max_iou) / (1 + max_iou + 1e-6)
        total_dice += dice.mean().item()
        
        correct = (p['labels'] == t['labels'][matched_idx]) & (max_iou > 0.5)
        total_acc += correct.float().mean().item()
        count += 1
    return (total_acc / count, total_dice / count) if count > 0 else (0, 0)

# -----------------------------------------------------------------------------
# 4. Training & Evaluation
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', default='resnet', choices=['resnet', 'mobilenet'])
    parser.add_argument('--batch-size', default=4, type=int) # Lowered for ResNet memory
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_ds = VistaDataset(args.data_dir, 'train', get_train_transforms())
    val_ds = VistaDataset(args.data_dir, 'test', get_valid_transforms())
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    if args.model == 'resnet':
        print("Using ResNet50 FPN V2 (Default)...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    else:
        print("Using MobileNetV3-Large (Fast)...")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 201)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            if scaler:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            
            train_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        # Validation
        model.eval()
        val_acc, val_dice = 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                acc, dice = calculate_metrics(outputs, targets)
                val_acc += acc
                val_dice += dice
        
        print(f"Epoch {epoch+1} Results: Loss: {train_loss/len(train_loader):.4f} | Acc: {val_acc/len(val_loader):.4f} | Dice: {val_dice/len(val_loader):.4f}")

if __name__ == "__main__":
    main()

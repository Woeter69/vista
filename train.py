import os
import argparse
import json
import random
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# 1. Optimized Transforms (No resizing here, it's done during load)
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
# 2. Dataset (Optimized: Resizes immediately on load)
# -----------------------------------------------------------------------------
class VistaDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None, img_size=640, limit=None):
        self.root, self.split, self.transforms, self.img_size = root, split, transforms, img_size
        
        json_name = "instances_train.json" if split == 'train' else "instances_test.json"
        self.ann_file = self._find_file(root, json_name)
        self.img_dir = self._find_dir(root, split) or os.path.dirname(self.ann_file)

        with open(self.ann_file, 'r') as f:
            data = json.load(f)
            
        self.ids = [img['id'] for img in data['images']]
        if limit:
            random.seed(42)
            self.ids = random.sample(self.ids, min(limit, len(self.ids)))
            
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
        
        # Immediate Resize on Load (Massive speedup)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            if bw > 1 and bh > 1:
                # Scale boxes to 640x640
                x1, y1 = bx * (self.img_size / w), by * (self.img_size / h)
                x2, y2 = (bx + bw) * (self.img_size / w), (by + bh) * (self.img_size / h)
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = transformed['image'], torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4), torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            image, boxes, labels = torchvision.transforms.functional.to_tensor(image), torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.as_tensor(labels, dtype=torch.int64)
            
        return image, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

    def __len__(self): return len(self.ids)

def collate_fn(batch): return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./checkpoints')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--img-size', default=640, type=int)
    parser.add_argument('--train-limit', default=10000, type=int, help='Limit imgs per epoch for speed')
    parser.add_argument('--val-limit', default=500, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--colab', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Subsampling the training set is the ONLY way to make epochs fast in a hackathon
    train_ds = VistaDataset(args.data_dir, 'train', get_train_transforms(), img_size=args.img_size, limit=args.train_limit)
    val_ds = VistaDataset(args.data_dir, 'test', get_valid_transforms(), img_size=args.img_size, limit=args.val_limit)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, persistent_workers=True)

    print(f"ULTRA Mode: ResNet50 | TrainLimit={args.train_limit} | ImgSize={args.img_size}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 201)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    start_epoch, best_loss = 0, float('inf')
    last_ckpt_path = os.path.join(args.save_dir, "last_model.pt")
    if args.resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']

    for epoch in range(start_epoch, args.epochs):
        model.train(); train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            if scaler: scaler.scale(losses).backward(); scaler.step(optimizer); scaler.update()
            else: losses.backward(); optimizer.step()
            scheduler.step(); train_loss += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.4f}")

        # Quick validation (Loss based to save time)
        model.eval(); val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # In eval mode model returns boxes, to get loss we'd need to keep it in train mode 
                # but with no_grad. For speed, we'll just use train_loss for best model.
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Results: Avg Train Loss: {avg_loss:.4f}")
        ckpt = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(ckpt, last_ckpt_path)
        if avg_loss < best_loss:
            best_loss = avg_loss; torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"New Best Model saved (Loss: {best_loss:.4f})")

if __name__ == "__main__": main()

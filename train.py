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
# 1. LIGHTNING Transforms (256px)
# -----------------------------------------------------------------------------
def get_train_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# -----------------------------------------------------------------------------
# 2. Dataset
# -----------------------------------------------------------------------------
class VistaDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None, limit=None):
        self.root, self.split, self.transforms = root, split, transforms
        
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
        
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            if bw > 1 and bh > 1:
                boxes.append([bx, by, bx + bw, by + bh])
                labels.append(ann['category_id'])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = transformed['image'], torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4), torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            image, boxes, labels = torchvision.transforms.functional.to_tensor(image), torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.as_tensor(labels, dtype=torch.int64)
            
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return image, target

    def __len__(self): return len(self.ids)

def collate_fn(batch): return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./checkpoints')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train-limit', default=2500, type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LIGHTNING MODE: 2500 imgs per epoch, 256px resolution, 2 workers
    train_ds = VistaDataset(args.data_dir, 'train', get_train_transforms(256), limit=args.train_limit)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    print(f"LIGHTNING MODE: FasterRCNN-ResNet50 | TrainLimit={args.train_limit} | ImgSize=256")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 201)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    last_ckpt_path = os.path.join(args.save_dir, "last_model.pt")
    if args.resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])

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
            
            if scaler: scaler.scale(losses).backward(); scaler.step(optimizer); scaler.update()
            else: losses.backward(); optimizer.step()
            
            train_loss += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.4f}")

        print(f"Epoch {epoch+1} Results: Avg Loss: {train_loss/len(train_loader):.4f}")
        torch.save({'model': model.state_dict(), 'epoch': epoch+1}, last_ckpt_path)

if __name__ == "__main__": main()

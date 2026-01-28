import os
import argparse
import json
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# 1. Transforms
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
    def __init__(self, root, split='train', transforms=None, use_mosaic=False, img_size=1024):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.use_mosaic = use_mosaic
        self.img_size = img_size
        
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

    def load_image_and_boxes(self, index):
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
        return image, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def load_mosaic(self, index):
        indices = [index] + random.choices(range(len(self)), k=3)
        s = self.img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_boxes, mosaic_labels = [], []
        yc, xc = s, s

        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_image_and_boxes(idx)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w, pad_h = x1a - x1b, y1a - y1b
            if len(boxes) > 0:
                boxes[:, [0, 2]] += pad_w
                boxes[:, [1, 3]] += pad_h
                mosaic_boxes.append(boxes)
                mosaic_labels.append(labels)

        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.concatenate(mosaic_boxes)
            mosaic_labels = np.concatenate(mosaic_labels)
            np.clip(mosaic_boxes, 0, 2 * s, out=mosaic_boxes)
        else:
            mosaic_boxes = np.zeros((0, 4))
            mosaic_labels = np.zeros((0,))
        
        return mosaic_img, mosaic_boxes, mosaic_labels

    def __getitem__(self, index):
        if self.split == 'train' and self.use_mosaic and random.random() < 0.5:
            image, boxes, labels = self.load_mosaic(index)
        else:
            image, boxes, labels = self.load_image_and_boxes(index)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = torchvision.transforms.functional.to_tensor(image)
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([self.ids[index]])}
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# -----------------------------------------------------------------------------
# 3. Metrics
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
# 4. Main
# -----------------------------------------------------------------------------
def setup_colab():
    if not os.path.exists('/content/drive'):
        print("Warning: Google Drive not mounted.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./checkpoints')
    parser.add_argument('--model', default='resnet', choices=['resnet', 'mobilenet', 'retinanet'])
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mosaic', action='store_true')
    parser.add_argument('--colab', action='store_true')
    args = parser.parse_args()

    if args.colab: setup_colab()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_ds = VistaDataset(args.data_dir, 'train', get_train_transforms(), use_mosaic=args.mosaic)
    val_ds = VistaDataset(args.data_dir, 'test', get_valid_transforms())
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"Loading Model: {args.model}")
    if args.model == 'resnet':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 201)
    elif args.model == 'mobilenet':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 201)
    elif args.model == 'retinanet':
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(256, num_anchors, 201)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    start_epoch, best_acc = 0, 0
    last_ckpt_path = os.path.join(args.save_dir, "last_model.pt")
    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt.get('acc', 0)

    for epoch in range(start_epoch, args.epochs):
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
            scheduler.step()
            train_loss += losses.item()
            pbar.set_postfix(loss=losses.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        total_val_acc, total_val_dice, val_count = 0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                acc, dice = calculate_metrics(outputs, targets)
                total_val_acc += acc
                total_val_dice += dice
                val_count += 1
        
        avg_val_acc = total_val_acc / val_count
        print(f"Epoch {epoch+1} Results: Loss: {train_loss/len(train_loader):.4f} | Acc: {avg_val_acc:.4f} | Dice: {total_val_dice/val_count:.4f}")

        ckpt = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'acc': avg_val_acc}
        torch.save(ckpt, last_ckpt_path)
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"New Best Model saved (Acc: {best_acc:.4f})")

if __name__ == "__main__":
    main()
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
# 1. Pro Winning Transforms (640px + Heavy Augmentation)
# -----------------------------------------------------------------------------
def get_train_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.3),
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# -----------------------------------------------------------------------------
# 2. Dataset (Optimized with Mosaic for Winning Accuracy)
# -----------------------------------------------------------------------------
class VistaDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None, use_mosaic=False, img_size=640, limit=None):
        self.root, self.split, self.transforms, self.img_size = root, split, transforms, img_size
        self.use_mosaic = use_mosaic
        
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
            
        all_cats = sorted(list(set([ann['category_id'] for ann in data['annotations']]))) # noqa
        self.cat_to_idx = {cat: i + 1 for i, cat in enumerate(all_cats)}

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
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            if bw > 1 and bh > 1:
                boxes.append([bx, by, bx + bw, by + bh])
                labels.append(self.cat_to_idx[ann['category_id']])
        return image, np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64), img_id

    def load_mosaic(self, index):
        indices = [index] + random.choices(range(len(self)), k=3)
        s = self.img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_boxes, mosaic_labels = [], []
        yc, xc = s, s
        main_img_id = None
        for i, idx in enumerate(indices):
            img, boxes, labels, img_id = self.load_image_and_boxes(idx)
            if i == 0: main_img_id = img_id
            h, w = img.shape[:2]
            img = cv2.resize(img, (s, s))
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= (s / w)
                boxes[:, [1, 3]] *= (s / h)
            h, w = s, s
            
            if i == 0: x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = max(xc-w,0), max(yc-h,0), xc, yc, w-(xc-max(xc-w,0)), h-(yc-max(yc-h,0)), w, h
            elif i == 1: x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = xc, max(yc-h,0), min(xc+w,s*2), yc, 0, h-(yc-max(yc-h,0)), min(w, min(xc+w,s*2)-xc), h
            elif i == 2: x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = max(xc-w,0), yc, xc, min(yc+h,s*2), w-(xc-max(xc-w,0)), 0, w, min(h, min(yc+h,s*2)-yc)
            else: x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = xc, yc, min(xc+w,s*2), min(yc+h,s*2), 0, 0, min(w, min(xc+w,s*2)-xc), min(h, min(yc+h,s*2)-yc)
            
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w, pad_h = x1a - x1b, y1a - y1b
            if len(boxes) > 0:
                boxes[:, [0, 2]] += pad_w; boxes[:, [1, 3]] += pad_h
                mosaic_boxes.append(boxes); mosaic_labels.append(labels)
        
        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.concatenate(mosaic_boxes); mosaic_labels = np.concatenate(mosaic_labels)
            mosaic_boxes[:, [0, 2]] = np.clip(mosaic_boxes[:, [0, 2]], 0, 2 * s)
            mosaic_boxes[:, [1, 3]] = np.clip(mosaic_boxes[:, [1, 3]], 0, 2 * s)
            valid = (mosaic_boxes[:, 2] > mosaic_boxes[:, 0] + 1.0) & (mosaic_boxes[:, 3] > mosaic_boxes[:, 1] + 1.0)
            mosaic_boxes, mosaic_labels = mosaic_boxes[valid], mosaic_labels[valid]
        else: mosaic_boxes, mosaic_labels = np.zeros((0, 4)), np.zeros((0,))
        
        mosaic_img = cv2.resize(mosaic_img, (s, s))
        mosaic_boxes *= 0.5
        return mosaic_img, mosaic_boxes, mosaic_labels, main_img_id

    def __getitem__(self, index):
        if self.split == 'train' and self.use_mosaic and random.random() < 0.5:
            image, boxes, labels, img_id = self.load_mosaic(index)
        else:
            image, boxes, labels, img_id = self.load_image_and_boxes(index)
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = transformed['image'], torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4), torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            image, boxes, labels = torchvision.transforms.functional.to_tensor(image), torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.as_tensor(labels, dtype=torch.int64)
            
        return image, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

    def __len__(self): return len(self.ids)

def collate_fn(batch): return tuple(zip(*batch))

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
        
        # Ensure boxes are on CPU for torchvision metrics calculation
        p_boxes = p['boxes'].detach().cpu()
        t_boxes = t['boxes'].detach().cpu()
        p_labels = p['labels'].detach().cpu()
        t_labels = t['labels'].detach().cpu()
        
        # Compute IoU between predicted and ground truth boxes
        iou = torchvision.ops.box_iou(p_boxes, t_boxes)
        max_iou, matched_idx = iou.max(dim=1)
        
        # Dice approx for boxes: 2*IoU / (1 + IoU)
        dice = (2 * max_iou) / (1 + max_iou + 1e-6)
        total_dice += dice.mean().item()
        
        # Accuracy: correct class if IoU > 0.5
        correct = (p_labels == t_labels[matched_idx]) & (max_iou > 0.5)
        total_acc += correct.float().mean().item()
        count += 1
    return (total_acc / count, total_dice / count) if count > 0 else (0, 0)

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./checkpoints')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--accum', default=8, type=int, help='Gradient accumulation steps')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--img-size', default=640, type=int)
    parser.add_argument('--mosaic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--kaggle', action='store_true')
    args = parser.parse_args()

    if args.kaggle:
        args.data_dir = "/kaggle/input/vista-dataset" if args.data_dir == "/content/vista_data/" else args.data_dir
        args.save_dir = "/kaggle/working/checkpoints" if args.save_dir == "./checkpoints" else args.save_dir
        print("Running in Kaggle environment.")
    elif args.colab:
        setup_colab()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = VistaDataset(args.data_dir, 'train', get_train_transforms(args.img_size), use_mosaic=args.mosaic, img_size=args.img_size)
    val_ds = VistaDataset(args.data_dir, 'test', get_valid_transforms(args.img_size), img_size=args.img_size, limit=500)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"WINNING MODE: ResNet50 | Size={args.img_size} | Mosaic={args.mosaic} | Accum={args.accum}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 201)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader)//args.accum, epochs=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    start_epoch, best_acc = 0, 0
    last_ckpt_path = os.path.join(args.save_dir, "last_model.pt")
    
    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming from checkpoint found in: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt.get('acc', 0)
        print(f"Successfully resumed from Epoch {start_epoch} (Best Acc so far: {best_acc:.4f})")
    elif args.resume:
        print(f"Warning: --resume was passed but no checkpoint was found at {last_ckpt_path}. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs):
        model.train(); train_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]; targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / args.accum
            
            if scaler: scaler.scale(losses).backward()
            else: losses.backward()
            
            if (i + 1) % args.accum == 0:
                if scaler: scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(); scheduler.step()
            
            train_loss += losses.item() * args.accum
            pbar.set_postfix(loss=f"{losses.item()*args.accum:.4f}")

        # --- Validation Step ---
        model.eval()
        total_val_acc, total_val_dice, val_count = 0, 0, 0
        print(f"Calculating Accuracy and Dice for Epoch {epoch+1}...")
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                acc, dice = calculate_metrics(outputs, targets)
                total_val_acc += acc
                total_val_dice += dice
                val_count += 1
        
        avg_acc = total_val_acc / val_count
        avg_dice = total_val_dice / val_count
        
        # This is what you requested: telling you the results after every epoch
        print(f"\n" + "="*40)
        print(f"EPOCH {epoch+1} COMPLETE")
        print(f"Average Loss: {train_loss/len(train_loader):.4f}")
        print(f"Accuracy:     {avg_acc:.4f}")
        print(f"Mean Dice:    {avg_dice:.4f}")
        print("="*40 + "\n")

        ckpt = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'acc': avg_acc, 'dice': avg_dice}
        torch.save(ckpt, last_ckpt_path)
        print(f"Checkpoint saved: {last_ckpt_path}")
        
        if avg_acc > best_acc:
            best_acc = avg_acc; torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"*** NEW BEST MODEL SAVED with Accuracy: {best_acc:.4f} ***")

if __name__ == "__main__": main()
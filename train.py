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
        
        if not self.ann_file:
            print(f"Warning: Could not find {json_name} in {root}. Searching for any JSON...")
            for r, d, f in os.walk(root):
                for file in f:
                    if file.endswith('.json') and split in file.lower():
                        self.ann_file = os.path.join(r, file)
                        break
                if self.ann_file: break

        if not self.ann_file:
            raise FileNotFoundError(f"Could not locate {json_name} in {root}")
            
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
            
        # FIXED: Use a hardcoded or globally sorted list of category IDs (1-200)
        # This is CRITICAL. Without this, Train and Val IDs won't match.
        self.cat_to_idx = {i: i for i in range(1, 201)} 

    def _find_file(self, path, name):
        for r, d, f in os.walk(path):
            for file in f:
                if file.lower() == name.lower():
                    return os.path.join(r, file)
        return None

    def _find_dir(self, path, name):
        for r, d, f in os.walk(path):
            for folder in d:
                if folder.lower() == name.lower():
                    return os.path.join(r, folder)
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

    def load_mega_mosaic(self, index):
        """
        Winning Feature: Take 12 random single-object images and tile them into a 4x3 grid.
        This forces the model to learn multi-object scenes (avg 12 objects) like the test set.
        """
        rows, cols = 4, 3
        indices = [index] + random.choices(range(len(self)), k=(rows * cols) - 1)
        s = self.img_size
        
        # Sub-image size
        sw, sh = s // cols, s // rows
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        mosaic_boxes, mosaic_labels = [], []
        
        for i, idx in enumerate(indices):
            img, boxes, labels, _ = self.load_image_and_boxes(idx)
            img = cv2.resize(img, (sw, sh))
            
            # Position in grid
            r, c = i // cols, i % cols
            x1, y1 = c * sw, r * sh
            
            mosaic_img[y1:y1+sh, x1:x1+sw] = img
            
            if len(boxes) > 0:
                # Scale boxes to sub-image size and shift to grid position
                orig_h, orig_w = self.img_info[self.ids[idx]]['height'], self.img_info[self.ids[idx]]['width']
                boxes[:, [0, 2]] *= (sw / orig_w)
                boxes[:, [1, 3]] *= (sh / orig_h)
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                mosaic_boxes.append(boxes)
                mosaic_labels.append(labels)
        
        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.concatenate(mosaic_boxes); mosaic_labels = np.concatenate(mosaic_labels)
        else:
            mosaic_boxes, mosaic_labels = np.zeros((0, 4)), np.zeros((0,))
            
        return mosaic_img, mosaic_boxes, mosaic_labels, self.ids[index]

    def __getitem__(self, index):
        # Increased probability to 80% to force the model to learn multi-object scenes
        if self.split == 'train' and self.use_mosaic and random.random() < 0.8:
            image, boxes, labels, img_id = self.load_mega_mosaic(index)
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
# 3. Official Hackathon Metric (IIT BHU Codefest '26)
# -----------------------------------------------------------------------------
def calculate_metrics(preds, targets, threshold=0.5):
    """
    Official Metric: If predicted count != ground truth count, score is 0.
    Else, score is (correctly identified objects) / total objects.
    """
    total_score = 0
    count = 0
    
    for p, t in zip(preds, targets):
        mask = p['scores'] > threshold
        p_labels = p['labels'][mask].detach().cpu().tolist()
        t_labels = t['labels'].detach().cpu().tolist()
        
        # RULE: Exact count required
        if len(p_labels) != len(t_labels):
            total_score += 0 
        else:
            # Calculate match score (intersection of multiset)
            matches = 0
            temp_t = t_labels.copy()
            for label in p_labels:
                if label in temp_t:
                    matches += 1
                    temp_t.remove(label)
            total_score += matches / len(t_labels) if len(t_labels) > 0 else 1.0
        
        count += 1
        
    return total_score / count if count > 0 else 0

def find_best_threshold(model, val_loader, device):
    """Find the threshold that maximizes the hackathon score."""
    model.eval()
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    scores = {t: 0 for t in thresholds}
    
    print("Optimizing confidence threshold...")
    with torch.no_grad():
        # Use a small subset for speed
        for i, (images, targets) in enumerate(val_loader):
            if i > 20: break 
            images = [img.to(device) for img in images]
            outputs = model(images)
            for t in thresholds:
                scores[t] += calculate_metrics(outputs, targets, threshold=t)
    
    best_t = max(scores, key=scores.get)
    return best_t, scores[best_t] / min(len(val_loader), 21)

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
    parser.add_argument('--lr', default=0.0002, type=float) # Doubled for faster convergence
    parser.add_argument('--img-size', default=640, type=int)
    parser.add_argument('--mosaic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--kaggle', action='store_true')
    args = parser.parse_args()

    # --- Setup Device ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # ... [setup code] ...
    
    train_ds = VistaDataset(args.data_dir, 'train', get_train_transforms(args.img_size), use_mosaic=args.mosaic, img_size=args.img_size)
    val_ds = VistaDataset(args.data_dir, 'test', get_valid_transforms(args.img_size), img_size=args.img_size, limit=100) # Reduced to 100 for speed
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"WINNING MODE: ResNet50 | Size={args.img_size} | Mosaic={args.mosaic} | Accum={args.accum}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 201)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader)//args.accum, epochs=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    start_epoch = 0
    best_acc = 0.0 # Track the best accuracy globally
    last_ckpt_path = os.path.join(args.save_dir, "last.pth")
    
    # --- Smart Resume Logic ---
    if args.resume:
        found_path = None
        if os.path.exists(last_ckpt_path):
            found_path = last_ckpt_path
        else:
            # Kaggle specific: search for model in inputs if not in working dir
            print(f"Searching for checkpoint in Kaggle inputs...")
            for r, d, f in os.walk("/kaggle/input"):
                if "last.pth" in f:
                    found_path = os.path.join(r, "last.pth")
                    break
        
        if found_path:
            print(f"Resuming from: {found_path}")
            try:
                ckpt = torch.load(found_path, map_location=device)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
                start_epoch = ckpt['epoch']
                # CRITICAL: Load the best_acc from the checkpoint so we can compare against it
                best_acc = ckpt.get('acc', 0.0)
                print(f"Successfully resumed from Epoch {start_epoch} (Previous Best Acc: {best_acc:.4f})")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Warning: Checkpoint architecture mismatch or corruption detected. Starting fresh.")
                start_epoch = 0
                best_acc = 0.0
        else:
            print(f"Warning: --resume was passed but last.pth was not found anywhere. Starting fresh.")

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
        # --- Official Validation Step ---
        model.eval()
        best_t, official_score = find_best_threshold(model, val_loader, device)
        
        print(f"\n" + "="*40)
        print(f"EPOCH {epoch+1} COMPLETE")
        print(f"Average Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Official Codefest Score: {official_score:.4f} (at threshold {best_t})")
        print("="*40 + "\n")

        ckpt = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(), 'acc': official_score, 'best_t': best_t}
        torch.save(ckpt, last_ckpt_path)
        print(f"Checkpoint saved: {last_ckpt_path}")
        
        if official_score > best_acc:
            best_acc = official_score; torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"*** NEW BEST MODEL SAVED with Codefest Score: {best_acc:.4f} ***")

if __name__ == "__main__": main()
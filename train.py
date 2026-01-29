import os
import argparse
import json
import random
import time
import math
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image, ImageDraw
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# 1. Albumentations & Transforms
# -----------------------------------------------------------------------------
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Color/Noise Augmentations
        A.OneOf([
            A.MotionBlur(p=1),
            A.MedianBlur(blur_limit=3, p=1),
            A.Blur(blur_limit=3, p=1),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        # Geometric
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
        # Final formatting
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transforms():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# -----------------------------------------------------------------------------
# 2. Dataset & Mosaic Augmentation
# -----------------------------------------------------------------------------
class VistaDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None, use_mosaic=False, img_size=1024):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.use_mosaic = use_mosaic
        self.img_size = img_size
        
        # Discovery Logic
        json_name = "instances_train.json" if split == 'train' else "instances_test.json"
        self.ann_file = self._find_file(root, json_name)
        if not self.ann_file:
            raise FileNotFoundError(f"Could not find {json_name} in {root}")
        
        self.img_dir = self._find_dir(root, split)
        if not self.img_dir:
            self.img_dir = os.path.dirname(self.ann_file)
            
        print(f"[{split.upper()}] Anns: {self.ann_file} | Imgs: {self.img_dir}")

        with open(self.ann_file, 'r') as f:
            self.coco = json.load(f)
            
        self.ids = [img['id'] for img in self.coco['images']]
        self.img_info = {img['id']: img for img in self.coco['images']}
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

    def _find_file(self, search_path, filename):
        for root, dirs, files in os.walk(search_path):
            if filename in files: return os.path.join(root, filename)
        return None

    def _find_dir(self, search_path, dirname):
        possible = os.path.join(search_path, dirname)
        if os.path.isdir(possible): return possible
        for root, dirs, files in os.walk(search_path):
            if dirname in dirs: return os.path.join(root, dirname)
        return None

    def load_image_and_boxes(self, index):
        img_id = self.ids[index]
        img_data = self.img_info[img_id]
        path = os.path.join(self.img_dir, img_data['file_name'])
        
        if not os.path.exists(path):
            path = self._find_file(self.root, img_data['file_name'])
        
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h]) # Pascal VOC
                labels.append(ann['category_id'])
                
        return image, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def load_mosaic(self, index):
        # 4-tile Mosaic
        indices = [index] + random.choices(range(len(self)), k=3)
        
        # Target size (can be larger than single image)
        s = self.img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []
        
        # Center point
        yc, xc = s, s

        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_image_and_boxes(idx)
            h, w = img.shape[:2]
            
            # Place image in specific quadrant
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            if len(boxes) > 0:
                # Adjust boxes
                boxes[:, [0, 2]] += pad_w
                boxes[:, [1, 3]] += pad_h
                
                # Clip boxes to mosaic image
                np.clip(boxes[:, 0], 0, 2 * s, out=boxes[:, 0])
                np.clip(boxes[:, 1], 0, 2 * s, out=boxes[:, 1])
                np.clip(boxes[:, 2], 0, 2 * s, out=boxes[:, 2])
                np.clip(boxes[:, 3], 0, 2 * s, out=boxes[:, 3])
                
                # Filter invalid
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                mosaic_boxes.append(boxes[valid])
                mosaic_labels.append(labels[valid])

        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.concatenate(mosaic_boxes)
            mosaic_labels = np.concatenate(mosaic_labels)
        else:
            mosaic_boxes = np.zeros((0, 4))
            mosaic_labels = np.zeros((0,))
            
        return mosaic_img, mosaic_boxes, mosaic_labels

    def __getitem__(self, index):
        # 50% chance of Mosaic in training
        if self.split == 'train' and self.use_mosaic and random.random() < 0.5:
            img, boxes, labels = self.load_mosaic(index)
        else:
            img, boxes, labels = self.load_image_and_boxes(index)

        # Transforms (Albumentations)
        if self.transforms:
            # Albumentations requires boxes to be list of lists
            # and non-empty for some operations.
            if len(boxes) > 0:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img = transformed['image']
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.tensor(transformed['labels'], dtype=torch.int64)
            else:
                # Just transform image
                transformed = self.transforms(image=img, bboxes=[], labels=[])
                img = transformed['image']
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.tensor([], dtype=torch.int64)
        else:
            # Fallback tensor conversion
            img = torchvision.transforms.functional.to_tensor(img)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Normalize 0-1 to tensor is done by ToTensorV2 in albumentations
        # Just ensure simple ToTensor if no transforms
        if isinstance(img, np.ndarray):
             img = torchvision.transforms.functional.to_tensor(img)

        # Handle empty boxes after transform clipping
        if len(boxes) > 0 and boxes.ndim == 2:
             # Ensure x2>x1, y2>y1
             keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
             boxes = boxes[keep]
             labels = labels[keep]
        
        # Reshape if became 1D or empty
        if boxes.ndim != 2 or len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([self.ids[index]])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0)
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# -----------------------------------------------------------------------------
# 3. Setup Helper
# -----------------------------------------------------------------------------
def setup_colab():
    """Checks if Google Drive is already mounted."""
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ Google Drive detected and mounted.")
    else:
        print("⚠️ Warning: Google Drive not detected at /content/drive/MyDrive.")
        print("Please run 'from google.colab import drive; drive.mount(\"/content/drive\")' in a notebook cell.")

# -----------------------------------------------------------------------------
# 4. Main Training Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/vista', help='Dataset root')
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--num-classes', default=201, type=int)
    parser.add_argument('--mosaic', action='store_true', help='Enable Mosaic Augmentation')
    args = parser.parse_args()

    if args.colab: setup_colab()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device} | Mosaic: {args.mosaic} | AMP: True")

    # Load Data
    print("Loading Data...")
    try:
        train_ds = VistaDataset(
            args.data_dir, split='train', 
            transforms=get_train_transforms(), 
            use_mosaic=args.mosaic
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    # Model: Faster R-CNN ResNet50 FPN V2 (Better weights than V1)
    print("Loading Model (ResNet50 FPN V2)...")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # Cosine Scheduler for smoother convergence
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists("last_run.pt"):
        print("Resuming...")
        ckpt = torch.load("last_run.pt", map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # AMP Forward
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # AMP Backward
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = losses.item()
            epoch_loss += loss_val
            if not math.isfinite(loss_val):
                print(f"Loss is {loss_val}, stopping")
                return

            pbar.set_postfix({'Loss': f"{loss_val:.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.6f}"})

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Avg Loss: {avg_loss:.4f}")

        # Checkpoint
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss
        }
        torch.save(ckpt, "last_run.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt['best_loss'] = best_loss
            torch.save(ckpt, "best_model.pt")
            print(f"Saved Best Model (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()
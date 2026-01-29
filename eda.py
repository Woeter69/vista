import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def run_eda(json_path, name):
    print(f"\n--- Analyzing {name} ---")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    num_images = len(images)
    num_anns = len(annotations)
    num_cats = len(categories)
    
    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_anns}")
    print(f"Number of categories: {num_cats}")
    
    # Category distribution
    cat_ids = [ann['category_id'] for ann in annotations]
    cat_counts = Counter(cat_ids)
    
    # Objects per image
    img_id_to_anns = Counter([ann['image_id'] for ann in annotations])
    objs_per_img = list(img_id_to_anns.values())
    
    if objs_per_img:
        avg_objs = sum(objs_per_img) / len(objs_per_img)
        max_objs = max(objs_per_img)
        min_objs = min(objs_per_img)
        print(f"Average objects per image: {avg_objs:.2f}")
        print(f"Max objects per image: {max_objs}")
        print(f"Min objects per image: {min_objs}")
    
    return {
        'num_images': num_images,
        'num_anns': num_anns,
        'cat_counts': cat_counts,
        'objs_per_img': objs_per_img
    }

base_path = "data/Vistas Dataset Public/Vistas Dataset Public"
train_json = os.path.join(base_path, "instances_train.json")
# Val json is in root data/ and nested. Let's check both or prefer nested.
val_json = os.path.join(base_path, "instances_val.json")
if not os.path.exists(val_json):
    val_json = "data/instances_val.json"

train_stats = run_eda(train_json, "Train")
val_stats = run_eda(val_json, "Validation")

# Check Categories mapping
with open(os.path.join(base_path, "Categories.json"), 'r') as f:
    cats_data = json.load(f)
    print(f"\nTotal Categories in Categories.json: {len(cats_data['categories'])}")

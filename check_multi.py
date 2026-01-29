import json
from collections import Counter

with open("data/Vistas Dataset Public/Vistas Dataset Public/instances_train.json", 'r') as f:
    data = json.load(f)

img_ids = [ann['image_id'] for ann in data['annotations']]
counts = Counter(img_ids)
most_common = counts.most_common(5)
print(f"Most common image_id counts: {most_common}")

num_multi = sum(1 for c in counts.values() if c > 1)
print(f"Number of images with multiple objects: {num_multi}")

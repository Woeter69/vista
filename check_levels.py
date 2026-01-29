import json
from collections import Counter

with open("data/instances_val.json", 'r') as f:
    data = json.load(f)

levels = [img.get('level', 'unknown') for img in data['images']]
print(f"Level distribution: {Counter(levels)}")

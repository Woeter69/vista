import pandas as pd
import json
import argparse
import sys

def verify_submission(csv_path):
    print(f"Verifying {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

    # 1. Check Columns
    required_cols = {'image_id', 'categories'}
    if not required_cols.issubset(df.columns):
        print(f"❌ Missing columns. Found {df.columns}, expected {required_cols}")
        return False
    
    # 2. Check Sorting of Rows (image_id ascending)
    image_ids = df['image_id'].tolist()
    if image_ids != sorted(image_ids):
        print("❌ Rows are NOT sorted by image_id ascending!")
        # Find first violation
        for i in range(len(image_ids)-1):
            if image_ids[i] > image_ids[i+1]:
                print(f"   Violation at index {i}: {image_ids[i]} > {image_ids[i+1]}")
                break
        return False
    
    # 3. Check Categories format and sorting
    all_passed = True
    for idx, row in df.iterrows():
        cats_str = row['categories']
        try:
            cats = json.loads(cats_str)
        except json.JSONDecodeError:
            print(f"❌ Row {idx} (Img {row['image_id']}): Invalid JSON format '{cats_str}'")
            all_passed = False
            continue
            
        if not isinstance(cats, list):
            print(f"❌ Row {idx} (Img {row['image_id']}): Categories is not a list '{cats_str}'")
            all_passed = False
            continue
            
        if not all(isinstance(c, int) for c in cats):
            print(f"❌ Row {idx} (Img {row['image_id']}): Contains non-integers '{cats_str}'")
            all_passed = False
            continue
            
        # Check sorting within list
        if cats != sorted(cats):
            print(f"❌ Row {idx} (Img {row['image_id']}): Category IDs not sorted '{cats_str}'")
            all_passed = False
            
    if all_passed:
        print(f"✅ Submission {csv_path} passed all checks!")
        return True
    else:
        print(f"❌ Submission {csv_path} FAILED checks.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='Path to submission CSV file')
    args = parser.parse_args()
    
    success = verify_submission(args.csv_file)
    sys.exit(0 if success else 1)

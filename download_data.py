import gdown
import os
import argparse

def download_from_drive(url, output_path='data'):
    # If downloading a single file, ensure the directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print(f"Downloading dataset from Google Drive link: {url}")
    try:
        # gdown.download handles single file links (regular and 'uc?id=')
        # fuzzy=True helps extract the ID from a sharing link
        gdown.download(url, output=output_path, quiet=False, fuzzy=True)
        print(f"Download complete: {output_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a file from Google Drive.')
    parser.add_argument('url', type=str, help='The full Google Drive file sharing URL')
    parser.add_argument('--output', type=str, default='dataset.zip', help='Output filename (default: dataset.zip)')
    
    args = parser.parse_args()
    download_from_drive(args.url, args.output)

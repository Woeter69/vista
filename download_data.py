import gdown
import os
import argparse

def download_from_drive(url, output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Downloading dataset from Google Drive link: {url}")
    try:
        # gdown.download_folder handles the full URL and downloads all contents
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading folder: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a folder from Google Drive.')
    parser.add_argument('url', type=str, help='The full Google Drive folder URL')
    parser.add_argument('--output', type=str, default='data', help='Output directory (default: data)')
    
    args = parser.parse_args()
    download_from_drive(args.url, args.output)
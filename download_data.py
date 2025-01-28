import gdown
import os

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
    # Replace with your actual Google Drive folder share link
    DRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE?usp=sharing'
    download_from_drive(DRIVE_FOLDER_URL)

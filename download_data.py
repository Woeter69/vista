import gdown
import os

def download_from_drive(folder_id, output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download the entire folder from Google Drive
    # Note: folder_id needs to be publicly accessible or authenticated
    url = f'https://drive.google.com/drive/folders/{folder_id}?usp=sharing'
    
    print(f"Downloading dataset from Google Drive folder: {folder_id}")
    try:
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading folder: {e}")

if __name__ == "__main__":
    # Placeholder folder ID - Replace with actual Vista dataset folder ID
    DRIVE_FOLDER_ID = '1-XYZ-PLACEHOLDER-ID' 
    download_from_drive(DRIVE_FOLDER_ID)

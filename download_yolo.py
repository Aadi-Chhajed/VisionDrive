"""
YOLOv3 Files Downloader for ADAS Project
Downloads: yolov3.weights, yolov3.cfg, coco.names
"""

import os
import urllib.request
import sys


def download_file(url, filename):
    """Download file with progress bar"""

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '-' * (bar_length - filled)
        sys.stdout.write(f"\r{filename}: |{bar}| {percent:.1f}% ")
        sys.stdout.flush()

    try:
        print(f"\nDownloading {filename}...")
        urllib.request.urlretrieve(url, filename, progress)
        print(f"\n✓ {filename} downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {e}")
        return False


def main():
    print("=" * 70)
    print("YOLOv3 FILES DOWNLOADER FOR ADAS")
    print("=" * 70)

    # Files to download
    files = [
        {
            'url': 'https://pjreddie.com/media/files/yolov3.weights',
            'filename': 'yolov3.weights',
            'description': 'YOLOv3 Weights',
            'size': '~248 MB'
        },
        {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'filename': 'yolov3.cfg',
            'description': 'YOLOv3 Configuration',
            'size': '~8 KB'
        },
        {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
            'filename': 'coco.names',
            'description': 'COCO Class Names',
            'size': '~1 KB'
        }
    ]

    # Check existing files
    print("\n📋 Checking files...")
    for file_info in files:
        if os.path.exists(file_info['filename']):
            size_mb = os.path.getsize(file_info['filename']) / (1024 * 1024)
            print(f"  ✓ {file_info['filename']} exists ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {file_info['filename']} missing")

    print("\n" + "=" * 70)
    response = input("\nDownload missing files? (y/n): ")

    if response.lower() != 'y':
        print("Download cancelled.")
        return

    print("\n🚀 Starting downloads...\n")

    # Download files
    success_count = 0
    for file_info in files:
        if os.path.exists(file_info['filename']):
            print(f"\n⏭ Skipping {file_info['filename']} (already exists)")
            success_count += 1
        else:
            if download_file(file_info['url'], file_info['filename']):
                success_count += 1

    # Final verification
    print("\n" + "=" * 70)
    print("FINAL CHECK")
    print("=" * 70)

    all_present = True
    for file_info in files:
        if os.path.exists(file_info['filename']):
            size_mb = os.path.getsize(file_info['filename']) / (1024 * 1024)
            print(f"✓ {file_info['filename']:<20} ({size_mb:>8.2f} MB)")
        else:
            print(f"✗ {file_info['filename']:<20} MISSING!")
            all_present = False

    print("=" * 70)

    if all_present and success_count == 3:
        print("\n🎉 SUCCESS! All YOLOv3 files are ready!")
        print("\n📹 Next step: Add your 'driving_video.mp4' to this folder")
        print("📹 Then run: python main2.py")
    else:
        print("\n⚠ INCOMPLETE! Some files failed to download.")
        print("Please check your internet connection and try again.")

    print("=" * 70)


if __name__ == "__main__":
    main()
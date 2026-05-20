import os
import sys
import urllib.request
import subprocess

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading: {read_so_far / (1024*1024):.1f} MB")
    sys.stdout.flush()

def main():
    url = "https://ndownloader.figshare.com/files/45206104"
    dest = "LungHist700.rar"
    extract_dir = "data"

    print("--- LungHist700 Dataset Downloader & Extractor ---")

    # Step 1: Download dataset
    if not os.path.exists(dest):
        print(f"Downloading dataset from Figshare ({url})...")
        try:
            urllib.request.urlretrieve(url, dest, download_progress)
            print("\nDownload completed successfully!")
        except Exception as e:
            print(f"\nError downloading file: {e}")
            sys.exit(1)
    else:
        print(f"Dataset archive '{dest}' already exists. Skipping download.")

    # Step 2: Verify unar installation
    print("Checking if 'unar' is available for extraction...")
    try:
        subprocess.run(["unar", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("'unar' is available.")
    except FileNotFoundError:
        # Check standard Homebrew path
        brew_unar = "/opt/homebrew/bin/unar"
        if os.path.exists(brew_unar):
            print(f"'unar' found at {brew_unar}.")
            # Add to PATH temporarily
            os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
        else:
            print("Error: 'unar' is not installed or not in PATH. Please wait for the Homebrew install to finish.")
            sys.exit(1)

    # Step 3: Extract archive
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting '{dest}' to '{extract_dir}/' directory using 'unar'...")
    try:
        # unar extracts into extract_dir. -f forces overwrite. -o specifies output directory.
        cmd = ["unar", "-f", "-o", extract_dir, dest]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Extraction completed successfully!")
            print("Contents of extract directory:")
            for item in os.listdir(extract_dir)[:10]:
                print(f" - {item}")
            if len(os.listdir(extract_dir)) > 10:
                print(f" ... and {len(os.listdir(extract_dir)) - 10} more items.")
        else:
            print(f"Error during extraction (exit code {result.returncode}):")
            print(result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to execute unar: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

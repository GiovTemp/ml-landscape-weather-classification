import os
import shutil

src_dir = "../dataset_train"
dest_dir = "../shine"

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through all the files in the source directory
for filename in os.listdir(src_dir):
    if "shine" in filename:
        # If the filename contains "rain", move the file to the destination directory
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(src_path, dest_path)
        print(f"Moved {filename} to {dest_dir}")

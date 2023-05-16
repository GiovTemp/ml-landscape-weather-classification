import os
import shutil

src_dir = "./datasets/dataset2/train/rain"
dest_dir = "./datasets/dataset2/train/sunrise"


# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through all the files in the source directory
for filename in os.listdir(src_dir):
    if "sunrise" in filename:
        # If the filename contains "rain", move the file to the destination directory
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(src_path, dest_path)
        print(f"Moved {filename} to {dest_dir}")

import os
import shutil

# Source directory and target directory
source_dir = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA_ysl/ICAS（重）/复发/173、赵现华-4(new)/Series_005_Arch"
target_dir = "/home/lwt/code_data/data/vessel/DIAS/full_sequence"
os.makedirs(target_dir,exist_ok=True)

mame = "1.3.12.2.1107.5.4.5.154061.30000017030100350167100000774.4_Frame"
new_prefix = "60"

# Get the list of all files in the source directory
file_list = os.listdir(source_dir)

# Iterate through the list of files
for filename in file_list:
    # Check if the file is a file and not a directory
    if os.path.isfile(os.path.join(source_dir, filename)):
        # Get the filename and extension
        base_name, ext = os.path.splitext(filename)
        if mame in base_name:
            # New file name prefix (here using "_new_" as the prefix, you can modify it as needed)
            # Construct the new file name
            new_filename = new_prefix +"_" +base_name.split(mame)[1] + ext
        
            # Construct the full path of the source file and the full path of the target file
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_filename)
        
            # Copy the file to the target directory
            shutil.copy(source_path, target_path)

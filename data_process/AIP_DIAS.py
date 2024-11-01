import cv2
import os
import numpy as np

# Define the folder path
dataset = 'test'
folder_path = f"/data/DIAS/{dataset}/images"

# Get all files in the folder
files = os.listdir(folder_path)
# Create a dictionary to store the image list for each sequence ID
sequence_images = {}

# Iterate through each file in the folder
for file in files:
    if file.endswith(".db"):
        continue
    # Split the file name to get the sequence ID and image ID
    sequence_id = file.split("_")[1]

    # Read the image
    image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)

    # If the sequence ID is not in the dictionary, add it
    if sequence_id not in sequence_images:
        sequence_images[sequence_id] = [image]
    else:
        sequence_images[sequence_id].append(image)

# Create a new folder to save the merged images
output_folder = f"/data/DIAS/{dataset}/aip_npy"
os.makedirs(output_folder, exist_ok=True)

# Merge the images for each sequence ID and perform maximum intensity projection
for sequence_id, images in sequence_images.items():
    # Stack the images together
    stacked_images = np.stack(images, axis=0)

    # Compute the maximum intensity projection
    max_density_projection = np.mean(stacked_images, axis=0)
    # max_density_projection = np.where(max_density_projection > 100,255,0)

    # Save the maximum intensity projection image
    # output_file = os.path.join(output_folder, f"{sequence_id}.jpg")
    # cv2.imwrite(output_file, max_density_projection)
    output_file = os.path.join(output_folder, f"image_{sequence_id}.npy")
    np.save(output_file, max_density_projection)

print("Merging and saving completed.")
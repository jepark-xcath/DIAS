import cv2
import os, json
import numpy as np
import pydicom as pyd

# Define the folder path
data_type = 'Cerebral_DERIVED_S'
data_list_path = f'/home/jepark/codes/Algorithms_Vessel_2D/Dataset/{data_type}.json'
with open(data_list_path, 'r') as f:
    files = json.load(f)
image_root = '/data/aiminer-neuroangio-june17-2024-all'

save_dir = os.path.join(image_root, 'preprocessed_dias', data_type, 'aip_npy')
os.makedirs(save_dir, exist_ok=True)
print(f"Save directory: {save_dir}")

# Iterate through each file in the folder
for file in files:
    if not file.endswith(".dcm"):
        continue

    p_name = file.split('/')[1] + '_' + file.split('/')[-1].split('.')[0]
    # Load image
    sequence = pyd.dcmread(os.path.join(image_root,file)).pixel_array

    # Z-score normalization
    sequence = (sequence - np.mean(sequence)) / np.std(sequence)
    sequence = (sequence - np.min(sequence))*255 / (np.max(sequence) - np.min(sequence))

    # MIP
    max_density_projection = np.mean(sequence, axis=0)

    # Save image
    output_file = os.path.join(save_dir, f"image_{p_name}.npy")
    np.save(output_file, max_density_projection)

    print("Completed.", p_name, sequence.shape, max_density_projection.shape)
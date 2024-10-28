import cv2
import os
import numpy as np

# Define input and output folder paths
input_folder = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA/MIP_labels"
output_folder = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA/RMS_labels"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Define the maximum width of small vessels to be removed (in pixels)
# max_vessel_width = 1

# Get all files in the input folder
# files = os.listdir(input_folder)

# Iterate through each file in the folder
# for file in files:
#     # Read binary blood vessel image
#     input_path = os.path.join(input_folder, file)
#     blood_vessel_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) // 255

#     # Use distance transform to determine the distance of each pixel to the nearest background pixel
#     dist_transform = cv2.distanceTransform(blood_vessel_image, cv2.DIST_L2, 3)

#     # Determine the regions of small vessels based on the distance transform results
#     small_vessels = (dist_transform < max_vessel_width).astype(np.uint8)

#     # Remove small vessels from the original image
#     removed_small_vessels = cv2.subtract(blood_vessel_image, small_vessels)

#     # Save the processed image to the output folder
#     output_path = os.path.join(output_folder, file)
#     cv2.imwrite(output_path, removed_small_vessels*255)
#     # cv2.imwrite(output_path, np.uint16(dist_transform/dist_transform.max()*255))
 
# print("Processing complete, images have been saved to the output folder.")

# Define the kernel size and number of iterations for morphological operations
min_width = 1

# Get all files in the input folder
files = os.listdir(input_folder)

# Iterate through each file in the folder
for file in files:
    # Read binary blood vessel image
    input_path = os.path.join(input_folder, file)
    blood_vessel_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)// 255

    # Perform morphological operations to remove small vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_width * 2 + 1, min_width * 2 + 1))
    removed_small_vessels = cv2.morphologyEx(blood_vessel_image, cv2.MORPH_OPEN,kernel)

    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, removed_small_vessels*255)

print("Processing complete, images have been saved to the output folder.")

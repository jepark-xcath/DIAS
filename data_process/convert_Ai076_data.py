import os 
import json
import cv2
import numpy as np
import pydicom as pyd

def save_image(data, save_path):
	""" Save an image to a given path """
	np.save(save_path, data)
	print(f"Saved image to {save_path}")

if __name__ == '__main__':
	data_type = 'Cerebral_ORIGINAL_S' #'Cerebral_DERIVED_S'
	data_list_path = f'/home/jepark/codes/Algorithms_Vessel_2D/Dataset/{data_type}.json'
	with open(data_list_path, 'r') as f:
		data_list = json.load(f)
	image_root = '/mnt/d/data/aiminer-neuroangio-june17-2024-all'


	save_dir = os.path.join(image_root, 'preprocessed_dias', data_type, 'images_npy')
	os.makedirs(save_dir, exist_ok=True)

	for data_path in data_list:
		p_name = data_path.split('/')[1] + '_' + data_path.split('/')[-1].split('.')[0]
		# Load images
		images = pyd.dcmread(os.path.join(image_root,data_path)).pixel_array
		# Z-score normalization
		images = (images - np.mean(images)) / np.std(images)
		images = (images - np.min(images))*255 / (np.max(images) - np.min(images))
		print(f"Patient: {p_name}, Num images: {len(images)}")

		# Save images
		save_path = os.path.join(save_dir, f"image_{p_name}.npy")
		save_image(images, save_path)
import os 
import cv2
import numpy as np

def read_images(image_dir, image_lst, p_name):
	""" Read images and return a list of images """
	image_lst = [f for f in image_lst if p_name in f]
	images = []
	for image_name in sorted(image_lst):
		image_path = os.path.join(image_dir, image_name)
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		images.append(image)
	return images

def save_image(data, save_path):
	""" Save an image to a given path """
	np.save(save_path, data)
	print(f"Saved image to {save_path}")

if __name__ == '__main__':
	dataset = 'test'
	image_dir = f'/mnt/d/data/DIAS/{dataset}/images'
	image_lst = [f for f in os.listdir(image_dir) if f.endswith('.png')]
	p_names = np.unique([f.split('_')[1] for f in image_lst])

	save_dir = f'/mnt/d/data/DIAS/{dataset}/images_npy'
	os.makedirs(save_dir, exist_ok=True)
	for p_name in p_names:
		images = read_images(image_dir, image_lst, p_name)
		print(f"Patient: {p_name}, Num images: {len(images)}")

		# Save images
		save_path = os.path.join(save_dir, f"image_{p_name}.npy")
		save_image(images, save_path)
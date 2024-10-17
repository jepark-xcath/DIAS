import os 
import numpy as np

if __name__ == '__main__':
	dataset = 'test'
	image_path = f'/mnt/d/data/DIAS/{dataset}/images_npy'
	image_list = os.listdir(image_path)

	step = 4
	for image in image_list:
		img = np.load(os.path.join(image_path, image))
		for i in range(0, img.shape[0]-step+1):
			sub_img = img[i:i+step]
			save_path = os.path.join(image_path, f'{image.split(".")[0]}_{i}.npy')
			np.save(save_path, sub_img)
			print(f'Saved image to {save_path}', img.shape)
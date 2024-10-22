import os 
import numpy as np

if __name__ == '__main__':
	dataset = 'test'
	# image_path = f'/mnt/d/data/DIAS/{dataset}/images_npy'
	image_path = '/mnt/d/data/aiminer-neuroangio-june17-2024-all/preprocessed_dias/Cerebral_DERIVED_S/images_npy'
	image_list = os.listdir(image_path)

	step = 1
	os.makedirs(os.path.join(image_path, f'step{int(step)}'), exist_ok=True)
	for image in image_list:
		img_path = os.path.join(image_path, image)
		if not os.path.isfile(img_path):
			continue
		img = np.load(img_path)
		for i in range(0, img.shape[0]-step+1):
			sub_img = img[i:i+step]
			save_path = os.path.join(image_path, f'step{int(step)}', f'{image.split(".")[0]}_{i}.npy')
			np.save(save_path, sub_img)
			print(f'Saved image to {save_path}', img.shape)
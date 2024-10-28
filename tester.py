import time
import shutil
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files,to_cuda,recompone_overlap
from utils.metrics import get_metrics, get_metrics, count_connect_component,get_color,AverageMeter
from utils.postprocessing import *
from configs.config import save_config
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd

class Tester(Trainer):
    def __init__(self,config, test_loader, model, save_dir, is_2d,  model_name):
        # super(Trainer, self).__init__()
        self.config = config
        self.test_loader = test_loader
        self.model = model
        self.is_2d = is_2d
        self.model_name = model_name
        name = config.MODEL_PATH.split('/')[-1] if not config.MODEL_PATH.endswith('/') else config.MODEL_PATH.split('/')[-2]
        self.save_path = os.path.join(config.INFERENCE_RESULT_PATH, name)
        self.labels_path = config.DATASET.TEST_LABEL_PATH
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        dir_exists(self.save_path)
        remove_files(self.save_path)
        shutil.copyfile(
            os.path.join(config.MODEL_PATH, 'config.yaml'),
            os.path.join(self.save_path, 'config.yaml'))
        
        cudnn.benchmark = True

        self.has_labels = True if self.labels_path is not None else False

    def test(self):
        
        self.model.eval()
        self._reset_metrics()
        self.VC=AverageMeter()
        # gts = self.get_labels()
        tbar = tqdm(self.test_loader, ncols=150)

        pres = []
        with torch.no_grad():
            
            for img in tbar:
               
                img = to_cuda(img)
                if not self.is_2d:
                    img = img.unsqueeze(1)
                with torch.amp.autocast('cuda',enabled=self.config.AMP):
                    pre = self.model(img)
            
               
                pre = torch.softmax(pre, dim=1)[:,1,:,:]
        
                pres.extend(pre)
        

        pres = torch.stack(pres, 0).cpu()

        # Recover the original image shape using either GT labels or the Test_dataset
        if self.has_labels:
            gts = self.get_labels()
            H, W = gts[0].shape
            image_id = range(len(gts))
        else:
            H, W = self.test_loader.img_list[0].shape[1:]  # Recover shape from the dataset
            image_id = self.test_loader.image_files

        pad_h = self.stride - (H - self.patch_size[0]) % self.stride
        pad_w = self.stride - (W - self.patch_size[1]) % self.stride
        new_h = H + pad_h
        new_w = W + pad_w
        pres = recompone_overlap(np.expand_dims(pres.cpu().detach().numpy(),axis=1), new_h, new_w, self.stride, self.stride)  # predictions
        predict = pres[:,0,0:H,0:W]
        predict_b = np.where(predict >= 0.4, 1, 0)

        # If labels are available, compute metrics and save results
        if self.has_labels:
            for j, id_ in enumerate(image_id):
                self._save_predictions(j, gts[j], predict[j], predict_b[j])
                self._update_metrics(*get_metrics(predict[j], gts[j], run_clDice=True).values())
                self.VC.update(count_connect_component(predict_b[j], gts[j]))

            self._log_metrics()
        else:
            # Save predictions only (no ground truth available)
            for j, id_ in enumerate(image_id):
                self._save_predictions(id_, None, predict[j], predict_b[j])

    def _save_predictions(self, index, gt, predict, predict_b):
        """Save predictions and optional GT as images."""
        # Postprocessing 
        predict_b = remove_small_vessles(predict_b)
        predict_b = get_connect_components(predict_b, min_size=1024)

        cv2.imwrite(self.save_path + f"/pre_{index}.png", np.uint8(predict * 255))
        cv2.imwrite(self.save_path + f"/pre_b{index}.png", predict_b)
        if gt is not None:
            cv2.imwrite(self.save_path + f"/gt_{index}.png", np.uint8(gt * 255))
            cv2.imwrite(self.save_path + f"/color_b{index}.png", get_color(predict_b, gt))

    def _log_metrics(self):
        """Log the evaluation metrics."""
        mean_data = list(self._get_metrics_mean().values())
        std_data = list(self._get_metrics_std().values())
        mean_data.append(self.VC.mean)
        std_data.append(self.VC.std)
        columns = list(self._get_metrics_mean().keys())
        columns.append("VC")

        formatted_data = [f"{mean}$\pm${std}" for mean, std in zip(mean_data, std_data)]

        # Create a DataFrame and save to CSV
        data_dict = {col: [val] for col, val in zip(columns, formatted_data)}
        df = pd.DataFrame(data_dict)
        df.to_csv(join(self.save_path, f"{self.model_name}_result.csv"))

        # Log metrics to console
        for k, v in self._get_metrics_mean().items():
            logger.info(f'{str(k):5s}: {v}')
        for k, v in self._get_metrics_std().items():
            logger.info(f'{str(k):5s}: {v}')

        logger.info(f'VC_mean: {self.VC.mean}')
        logger.info(f'VC_std: {self.VC.std}')

    def get_labels(self):
        """Load ground truth labels."""
        labels = subfiles(self.labels_path, join=False, suffix='png')
        label_list = []
        for i in range(len(labels)):
            gt = cv2.imread(os.path.join(self.labels_path, f'label_s{i}.png'), 0)
            gt = np.array(gt / 255)
            label_list.append(gt)
        return label_list
   
        
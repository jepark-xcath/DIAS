from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from data.dataset import Train_dataset, Valid_dataset, Test_dataset
from prefetch_generator import BackgroundGenerator
import torch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_train_loader(config):
    train_dataset = Train_dataset(
        config, images_path=config.DATASET.TRAIN_IMAGE_PATH, labels_path=config.DATASET.TRAIN_LABEL_PATH)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    val_dataset = Valid_dataset(
        config, images_path=config.DATASET.VAL_IMAGE_PATH, labels_path=config.DATASET.VAL_LABEL_PATH)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset) if config.DIS else None
    val_loader = DataLoaderX(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    return train_loader, val_loader


def build_test_loader(config):
    test_dataset = Test_dataset(
        config, images_path=config.DATASET.TEST_IMAGE_PATH, 
        # labels_path=config.DATASET.TEST_LABEL_PATH
        )

    test_loader = DataLoaderX(
        test_dataset,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )
    test_loader.img_list = test_dataset.img_list

    return test_loader

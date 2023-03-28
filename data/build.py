from torch.utils import data
from .datasets import CocoDataset

def build_datasets(data_folder, ann_file, training=True):
    datasets = CocoDataset(path=data_folder, annotations=ann_file, training=training)
    return datasets

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    datasets = build_datasets(cfg.INPUT.FOLDER, cfg.INPUT.ANN_FILE, is_train)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
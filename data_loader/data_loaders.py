from torch.utils.data import DataLoader
from data_loader.dataset import GSC_Dataset


def GSC_loader(config):
    train_dataset = GSC_Dataset(config, stage='train')
    train_loader = DataLoader(train_dataset,
                              shuffle=config.shuffle,
                              batch_size=config.batch_size,
                              num_workers=config.num_worker,
                              pin_memory=config.pin_memory,
                              drop_last=True)

    dev_dataset = GSC_Dataset(config, stage='validation')
    dev_loader = DataLoader(dev_dataset,
                            shuffle=config.shuffle,
                            batch_size=config.batch_size, 
                            num_workers=config.num_worker,
                            pin_memory=config.pin_memory,
                            drop_last=True)

    test_dataset = GSC_Dataset(config, stage='test')
    test_loader = DataLoader(test_dataset,
                             shuffle=config.shuffle,
                             batch_size=config.batch_size, 
                             num_workers=config.num_worker,
                             pin_memory=config.pin_memory,
                             drop_last=True)

    return train_loader, dev_loader, test_loader


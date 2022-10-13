from data.dataset import drop_train, drop_valid
from torch.utils.data import DataLoader


def load_dataset(opt):

    print("\n===> Loading Dataset")

    train_set = drop_train(opt.images_dir, augm=opt.augm)
    val_set = drop_valid(opt.images_dir)

    training_data_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
    validation_data_loader = DataLoader(
        dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False, pin_memory=False)

    print('\nDataset: {} \
            \nNumber of images: {} \
            \nNumber of iteration per epoch: {} \
            \nData augmentation: {}'.format(
        opt.images_dir,
        len(train_set),
        int(len(train_set) / opt.batch_size),
        opt.augm
    ))

    return training_data_loader, validation_data_loader

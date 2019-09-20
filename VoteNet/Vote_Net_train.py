"""
Train a 3D UNet
"""

import os
import sys
from Dataset import *
import Transform as bio_transform
from Vote_Net_model import *
from tools import *
import utils as helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        default=
                        # "ckpoints/VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_02262019_225006/checkpoint.pth.tar", type=str, metavar='PATH',
                        # "ckpoints/VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_problem3839_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_02262019_232115/checkpoint.pth.tar",
                        # type=str, metavar='PATH',

                        "",
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', default='ckpoints', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--logdir', default='logs', type=str, metavar='PATH',
                        help='path to tensorboard log file (default: none)')
    parser.add_argument('--GPU', default=0, type=int,
                        help='index of used GPU')
    parser.add_argument('--random-seed', default=1234, type=int,
                        help='seed for deterministic random generater')
    parser.add_argument('--debug', default=0, type=int,
                        help='if debug mode')
    args = parser.parse_args()

    class_ratio = np.ones(57)  # ratio of number of voxels with different class label

    config = dict(
        debug_mode=args.debug,

        training_list_file=os.path.realpath("../Data/LPBA40_data_sep/LPBA40_fold_1_train_VoteNet_hist.txt"), #"../Data/LPBA40_data_sep/LPBA40_fold_1_train_VoteNet.txt"
        validation_list_file=os.path.realpath("../Data/LPBA40_data_sep/LPBA40_fold_1_validate_VoteNet_hist.txt"), #"../Data/LPBA40_data_sep/LPBA40_fold_1_validate_VoteNet.txt"
        data_dir=os.path.realpath(""), #"../Data/LPBA40/"
        valid_data_dir=os.path.realpath(""), #"../Data/LPBA40/"
        patch_size=(72, 72, 72),  # size of 3d patch cropped from the original image (229, 193, 229)
        n_epochs=300,
        batch_size=2,
        valid_batch_size=2,
        print_batch_period=1,
        valid_epoch_period=1,
        save_ckpts_epoch_period=1,

        model='VoteNet3D',

        n_classes=1,
        n_channels=2,
        if_bias=1,
        if_batchnorm=1,

        # set random seed
        np_rand_state=np.random.RandomState(seed=args.random_seed),

        # weighted loss function
        if_weight_loss=False,
        weight_mode='const',
        weight_const=torch.FloatTensor(1/class_ratio/np.sum(1/class_ratio)),

        # config loss and optimizer
        # if_dice_loss= False,
        loss='dice2',  # cross_entropy/dice/focal_loss/binary_cross_entropy
        learning_rate=1e-3,
        lr_mode='multiStep',  # const/plateau/...
    )

    # optimizer scheduler setting
    if config['lr_mode'] == 'plateau':
        plateau_threshold = 0.003
        plateau_patience = 100
        plateau_factor = 0.2

    config['print_batch_period'] = 10 #max(120 // config['batch_size'] // 2, 1)

    # essential transforms
    sample_threshold = 0.005
    # balanced_random_crop = bio_transform.BalancedRandomCrop(config['patch_size'], threshold=sample_threshold)
    random_crop = bio_transform.RandomCrop(config['patch_size'])
    sitk_to_tensor = bio_transform.SitkToTensor()
    partition = bio_transform.Partition(config['patch_size'], overlap_size=(16, 16, 16))

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # load training data and validation data
    print("Initializing dataloader")

    # set up data loader

    train_transforms = [random_crop, sitk_to_tensor]
    valid_transforms = [partition]


    if config['debug_mode']:
        print("Debug mode")
        config['valid_epoch_period'] = 1

    config['experiment_name'] = 'Hist_dice2_{}{}{}{}{}{}{}{}{}'.format(
        '{}{}{}_'.format(config['model'], '_bias' if config['if_bias'] else '', '_BN' if config['if_batchnorm'] else ''),
        os.path.basename(config['data_dir']),
        '_{}'.format(os.path.basename(config['training_list_file']).split('.')[0]),
        '_patch_{}_{}_{}'.format(config['patch_size'][0], config['patch_size'][1], config['patch_size'][2]),
        '_batch_{}'.format(config['batch_size']),
        '_sampleThreshold_{}'.format(sample_threshold),
        '_weighted-loss_{}{}'.format(config['weight_mode'],
                                     '-{}-{}-{}'.format(class_ratio[0], class_ratio[1], class_ratio[2])
                                     if config['weight_mode'] == 'const' else '') if config['if_weight_loss'] else '',
        '_lr_{}'.format(config['learning_rate']),
        '_scheduler_{}'.format(config['lr_mode']) if not config['lr_mode'] == 'const' else ''
    )

    print("Training {}".format(config['experiment_name']))

    train_transform = transforms.Compose(train_transforms)

    valid_transform = transforms.Compose(valid_transforms)

    training_data = NiftiDataset(config['training_list_file'], config['data_dir'], mode="training", preload=False if config['debug_mode'] else True, transform=train_transform)

    training_data_loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    validation_data = NiftiDataset(config['validation_list_file'], config['valid_data_dir'], mode="training", preload=False if config['debug_mode'] else True, transform=valid_transform)

    validation_data_loader = DataLoader(validation_data, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)


    # build unet
    model_type = globals()[config['model']]
    model = model_type(in_channel=config['n_channels'], n_classes=config['n_classes'], bias=config['if_bias'], BN=config['if_batchnorm'])
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config['lr_mode'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   patience=plateau_patience // config['valid_epoch_period'],
                                                   factor=plateau_factor, verbose=True, threshold_mode='abs',
                                                   threshold=plateau_threshold,
                                                   min_lr=1e-5)

    if config['lr_mode'] == 'multiStep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, config['n_epochs'] - 50], gamma=0.1)

    # training
    train_model(model, training_data_loader, validation_data, optimizer, args, config, scheduler=scheduler if not config['lr_mode'] == 'const' else None)




if __name__ == '__main__':
    main()

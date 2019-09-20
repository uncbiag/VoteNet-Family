import os
import sys

from torchvision import transforms
import Transform as bio_transform
from Dataset import *
from Vote_Net_model import *
from tools import *
import torch.nn.functional as F
from temperature_scaling import ModelWithTemperature

def main():
    parser = argparse.ArgumentParser()
    # folder 1 parameters
    parser.add_argument('--resume',
                        default='ckpoints/'
                                # 'Dice2_VoteNet3D_bias_BN_IBSR18_IBSR18_fold_3_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03192019_163223/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03202019_160213/'
                                'BCE_VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03212019_132903/'
                                # 'BCE_VoteNet3D_bias_BN_LPBA40_LPBA40_fold_2_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03212019_141354/'
                                # 'Dice2_VoteNet3D_bias_BN_LPBA40_LPBA40_fold_2_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03212019_173759/'
                                'model_best.pth.tar',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='results', type=str, metavar='PATH',
                        help='path to tensorboard log file (default: none)')
    parser.add_argument('--GPU', default=2, type=int,
                        help='index of used GPU')
    args = parser.parse_args()

    n_classes = 1
    n_channels = 2

    model_name = args.resume.split('/')[1]
    params = model_name.split('_')

    overlap_size = (16, 16, 16)
    test_batch_size = 4
    testing_list_file = os.path.realpath("../Data/LPBA40_data_sep/LPBA40_fold_1_validate_VoteNet.txt") # "../Data/LPBA40_data_sep/LPBA40_fold_2_test_VoteNet.txt"
    data_dir = os.path.realpath("../Data/LPBA40/") # "../Data/LPBA40/"

    patch_size = (int(params[params.index('patch') + 1]), int(params[params.index('patch') + 2]), int(params[params.index('patch') + 3]))

    padding_mode = 'reflect'
    if_vote = False
    test_config = "{}_{}{}".format(overlap_size[0], padding_mode, '_vote' if if_vote else '')
    print("Test config: " + test_config)
    partition = bio_transform.Partition(patch_size, overlap_size, padding_mode=padding_mode, mode='pred')
    test_transforms = [partition]

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # load training data and validation data
    print("load data from {}".format(data_dir))

    test_transform = transforms.Compose(test_transforms)

    testing_data = NiftiDataset(testing_list_file, data_dir, "training", transform=test_transform)

    # build 3D VoteNet
    model = VoteNet3D(in_channel=n_channels, n_classes=n_classes, BN='BN' in params, bias='bias' in params)
    model.cuda()

    # resume checkpoint or initialize
    training_epochs = 0
    if args.resume is None:
        raise ValueError("The path of trained model is needed.")
    else:
        training_epochs, _ = initialize_model(model, ckpoint_path=args.resume)

    model.eval()

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(testing_data)


if __name__ == '__main__':
    main()
import os
import sys

from torchvision import transforms
import Transform as bio_transform
from Dataset import *
from Vote_Net_model import *
from tools import *
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    # folder 1 parameters
    parser.add_argument('--resume',
                        default='ckpoints/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_02052019_205306/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_02282019_015948/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03032019_221839/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03042019_205820/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_All_regs_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_02262019_000311/'
                                # 'VoteNet3D_bias_BN_LPBA40_LPBA40_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03062019_113624/'
                                # 'VoteNet3D_bias_BN_IBSR18_IBSR18_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03062019_155916/'
                                # 'VoteNet3D_bias_BN_IBSR18_IBSR18_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03062019_205858/'
                                # 'VoteNet3D_bias_BN_IBSR18_IBSR18_fold_1_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03092019_005523/'
                                # 'VoteNet3D_bias_BN_IBSR18_IBSR18_fold_2_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03182019_205500/'
                                # 'VoteNet3D_bias_BN_IBSR18_IBSR18_fold_3_train_VoteNet_patch_72_72_72_batch_2_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_03182019_210046/'
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
    parser.add_argument('--GPU', default=1, type=int,
                        help='index of used GPU')
    args = parser.parse_args()

    n_classes = 1
    n_channels = 2

    model_name = args.resume.split('/')[1]
    params = model_name.split('_')

    overlap_size = (16, 16, 16)
    test_batch_size = 4
    testing_list_file = os.path.realpath("../Data/LPBA40_data_sep/LPBA40_fold_1_test_VoteNet.txt") # "../Data/LPBA40_data_sep/LPBA40_fold_2_test_VoteNet.txt"
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

    # build UNet
    model = VoteNet3D(in_channel=n_channels, n_classes=n_classes, BN='BN' in params, bias='bias' in params)
    model.cuda()

    # resume checkpoint or initialize
    training_epochs = 0
    if args.resume is None:
        raise ValueError("The path of trained model is needed.")
    else:
        training_epochs, _ = initialize_model(model, ckpoint_path=args.resume)

    model.eval()

    save_dir = os.path.join(args.results_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    starting_time = time.time()
    for i in range(len(testing_data)):
        temp_starting_time = time.time()
        image_tiles, seg, prior, name = testing_data[i]

        message_tmp = "[{}] Segmenting {}".format(i, name)
        print(message_tmp)

        cat_input = torch.cat((image_tiles, prior), 1)
        prediction = pred_iter(model, cat_input, sub_size=test_batch_size)

        scaled_prediction = torch.add(prediction * 0.845, 0.018)
        # prediction_label = torch.max(prediction, 1)[1]
        # prediction_label = (F.sigmoid(prediction) > 0.5).squeeze()
        prediction_probs = (F.sigmoid(scaled_prediction)).squeeze()
        # prediction_label = 1 - prediction_label
        # prediction_softmax = F.softmax(prediction, dim=1)
        # prediction_softmax_choose_probs = prediction_softmax[:,1,...] > 0.9
        # entropy_pre = -prediction_softmax * torch.log2(prediction_softmax)
        # entropy_pre[entropy_pre != entropy_pre] = 0.0
        # entropy = torch.sum(entropy_pre, dim=1)
        # uncertainty = torch.max(prediction_softmax, 1)[0]
        # pred_image = partition.assemble(prediction_label, is_vote=if_vote)
        pred_probs_map = partition.assemble(prediction_probs, is_vote=if_vote)
        # pred_image = partition.assemble(prediction_softmax_choose_probs, is_vote=if_vote)
        # uncertainty_map = partition.assemble(uncertainty, is_vote=if_vote)
        # entropy_map = partition.assemble(entropy, is_vote=if_vote)

        # save prediction as image
        if args.results_dir is not "":
            print("Saving images")
            # seg_image_file = name + "_Seg_dice2_VoteNet_{}.nii.gz".format(test_config)
            # uncertainty_image_file = name + "_uncertainty_CU_{}.nii.gz".format(test_config)
            # entropy_image_file = name + "_entropy_CU_{}.nii.gz".format(test_config)
            probs_image_file = name + "_scaled_probs_VoteNet_{}.nii.gz".format(test_config)
            # sitk.WriteImage(pred_image, os.path.join(save_dir, seg_image_file))
            # sitk.WriteImage(uncertainty_map, os.path.join(save_dir, uncertainty_image_file))
            # sitk.WriteImage(entropy_map, os.path.join(save_dir, entropy_image_file))
            sitk.WriteImage(pred_probs_map, os.path.join(save_dir, probs_image_file))
            print(os.path.join(save_dir, probs_image_file))
            print("Single image time: {}".format(time.time() - temp_starting_time))

    print("Total time: {}".format(time.time() - starting_time))


if __name__ == '__main__':
    main()
import numpy as np
import os
import sys
import shutil
import time
import datetime
import gc
import subprocess
import argparse
# import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as utils
from torch.autograd import Variable
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from skimage import color
import SimpleITK as sitk

sys.path.append(os.path.realpath(".."))
import loss as bio_loss
import evalMetrics as metrics



class write_and_print():
    def __init__(self, if_write, save_dir, log_name):
        self.if_write = if_write
        if if_write:
            self.log = open(os.path.join(save_dir, log_name), 'a')

    def write(self, text):
        if self.if_write:
            self.log.write(text + '\n')
        print(text)

    def close(self):
        if self.if_write:
            self.log.close()


def get_params_num(model):
    """
    Get number of parameters of a model
    :param model: a pytorch modual
    :return: num:int, total number of parameters
    """
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num


def one_hot_encoder(input, num_classes):
    """
    Convert a label tensor/variable into one-hot format
    :param input: A pytorch tensor of size [batchSize,]
    :param num_classes: number of label classes
    :return: output: A pytorch tensor of shape [batchSize x num_classes]
    """
    variable = False
    if isinstance(input, torch.autograd.Variable):
        input = input.data
        variable = True
    output = torch.zeros(input.shape + (num_classes,))
    output.scatter_(1, input.unsqueeze(1), 1.0)
    if variable:
        output = torch.autograd.Variable(output)
    return output


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar', max_keep=1):
    save_path = os.path.join(path, prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    name = filename
    torch.save(state, os.path.join(save_path, name))
    if is_best:
        shutil.copyfile(os.path.join(save_path, name), os.path.join(save_path, 'model_best.pth.tar'))


def make_image_summary(images, truths, raw_output, maxoutput=4, overlap=True):
    """make image summary for tensorboard

    :param images: torch.Variable, NxCxDxHxW, 3D image volume (C:channels)
    :param truths: torch.Variable, NxDxHxW, 3D label mask
    :param raw_output: torch.Variable, NxCxHxWxD: prediction for each class (C:classes)
    :param maxoutput: int, number of samples from a batch
    :param overlap: bool, overlap the image with groundtruth and predictions
    :return: summary_images: list, a maxoutput-long list with element of tensors of Nx
    """
    slice_ind = images.size()[2] // 2
    images_2D = images.data[:maxoutput, :, slice_ind, :, :]
    truths_2D = truths.data[:maxoutput, slice_ind, :, :]
    predictions_2D = torch.max(raw_output.data, 1)[1][:maxoutput, slice_ind, :, :]

    grid_images = utils.make_grid(images_2D, pad_value=1)
    grid_truths = utils.make_grid(labels2colors(truths_2D, images=images_2D, overlap=overlap), pad_value=1)
    grid_preds = utils.make_grid(labels2colors(predictions_2D, images=images_2D, overlap=overlap), pad_value=1)

    return torch.cat([grid_images, grid_truths, grid_preds], 1)


def labels2colors(labels, images=None, overlap=False):
    """Turn label masks into color images
    :param labels: torch.tensor, NxMxN
    :param images: torch.tensor, NxMxN or NxMxNx3
    :param overlap: bool
    :return: colors: torch.tensor, Nx3xMxN
    """
    colors = []
    if overlap:
        if images is None:
            raise ValueError("Need background images when overlap is True")
        else:
            for i in range(images.size()[0]):
                image = images.squeeze(dim=1)[i, :, :]
                label = labels[i, :, :]
                colors.append(color.label2rgb(label.cpu().numpy(), image.cpu().numpy(), bg_label=0, alpha=0.7))
    else:
        for i in range(images.size()[0]):
            label = labels[i, :, :]
            colors.append(color.label2rgb(label.numpy(), bg_label=0))

    return torch.Tensor(np.transpose(np.stack(colors, 0), (0, 3, 1, 2)))


def weight_from_truth(truths, n_classes):
    ratio_inv = torch.zeros(n_classes)
    for i_class in range(n_classes):
        try:
            ratio_inv[i_class] = len(truths.view(-1)) / torch.sum(truths == i_class)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            ratio_inv[i_class] = 0
            pass
    loss_weight = ratio_inv / torch.sum(ratio_inv)

    return loss_weight


def initialize_model(model, optimizer=None, ckpoint_path=None):
    """
    Initialize a model with saved checkpoints, or random values
    :param model: a pytorch model to be initialized
    :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
    :param ckpoint_path: The path of saved checkpoint
    :return: current epoch and best validation score
    """
    finished_epoch = 0
    best_score = 0
    if ckpoint_path:
        if os.path.isfile(ckpoint_path):
            print("=> loading checkpoint '{}'".format(ckpoint_path))
            checkpoint = torch.load(ckpoint_path)
            best_score = checkpoint['best_score']
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if optimizer and checkpoint.__contains__('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            finished_epoch = finished_epoch + checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpoint_path, checkpoint['epoch']))
            del checkpoint
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))
    else:
        model.weights_init()
    return finished_epoch, best_score


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def train_model(model, training_data_loader, validation_data, optimizer, args, config, scheduler=None):
    # log writer
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)

    total_validation_time = 0
    if not config['debug_mode']:
        writer = SummaryWriter(os.path.join(args.logdir, now_date, config['experiment_name'] + '_' + now_time))
    if config['loss'] == 'focal_loss':
        FC = bio_loss.FocalLoss(config['n_classes'])

    # resume checkpoint or initialize
    finished_epochs, best_score = initialize_model(model, optimizer, args.resume)
    current_epoch = finished_epochs + 1

    iters_per_epoch = len(training_data_loader.dataset)
    print("Start Training:")
    while current_epoch <= config['n_epochs']:
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        if not config['lr_mode'] == 'const' and not config['lr_mode'] == 'plateau':
            scheduler.step(epoch=current_epoch)

        for i, (images, segs, names) in enumerate(training_data_loader):
            global_step = (current_epoch - 1) * iters_per_epoch + (i + 1) * config['batch_size']  # current globel step

            # weight for loss (inverse ratio of all labels)
            if config['if_weight_loss']:
                if config['weight_mode'] == 'dynamic':
                    loss_weight = weight_from_truth(segs, config['n_classes'])
                elif config['weight_mode'] == 'const':
                    loss_weight = config['weight_const']

            model.train()

            images = Variable(images)
            segs = Variable(segs.long())

            optimizer.zero_grad()

            output = model(images.cuda())

            output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, config['n_classes'])
            segs_flat = segs.view(segs.numel())


            if config['loss'] == 'dice':
                loss = bio_loss.dice_loss(output_flat, segs_flat)
            elif config['loss'] == 'cross_entropy':
                loss = F.cross_entropy(output_flat, segs_flat.cuda(), weight=loss_weight.cuda() if config['if_weight_loss'] else None)
            elif config['loss'] == 'focal_loss':
                loss = FC(output_flat, segs_flat.cuda())

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # average loss over 10 batches
            if i % config['print_batch_period'] == config['print_batch_period'] - 1:  # print every 10 mini-batches
                duration = time.time() - start_time
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.3f} lr:{} ({:.3f} sec/batch) {}'.format
                      (current_epoch, (i + 1) * config['batch_size'], iters_per_epoch,
                       (i + 1) * config['batch_size'] / iters_per_epoch * 100,
                       running_loss / config['print_batch_period'] if i > 0 else running_loss,
                       optimizer.param_groups[0]['lr'],
                       duration / config['print_batch_period'],
                       datetime.datetime.now().strftime("%D %H:%M:%S")
                       ))
                if not config['debug_mode']:
                    writer.add_scalar('loss/training', running_loss / config['print_batch_period'], global_step=global_step)  # data grouping by `slash`
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=global_step)  # data grouping by `slash`
                running_loss = 0.0
                start_time = time.time()  # log running time

            if config['debug_mode']:
                image_summary = make_image_summary(images, segs, output.cpu())

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            image_summary = make_image_summary(images, segs, output.cpu())
            writer.add_image("training", image_summary, global_step=global_step)

        # validation
        if current_epoch % config['valid_epoch_period'] == 0:
            model.eval()
            dice_all = np.zeros(57)
            start_time = time.time()  # log running time
            for j in range(len(validation_data) if best_score > 0.7 else 4):
                # for j in range(len(validation_data)):
                image_tiles, segs, names = validation_data[j]
                if config['if_weight_loss']:
                    if config['weight_mode'] == 'dynamic':
                        loss_weight = weight_from_truth(segs, config['n_classes'])
                    elif config['weight_mode'] == 'const':
                        loss_weight = config['weight_const']

                valid_batch_size = config['valid_batch_size']
                outputs = []  # a list of lists that store output at each step

                for i in range(0, np.ceil(float(image_tiles.size()[0]) / float(valid_batch_size)).astype(int)):
                    temp_input = image_tiles.narrow(0, valid_batch_size * i,
                                                    valid_batch_size if valid_batch_size * (i + 1) <= image_tiles.size()[0]
                                                    else image_tiles.size()[0] - valid_batch_size * i)

                    with torch.no_grad():
                        temp_input = Variable(temp_input)

                    # predict through model 1
                    temp_output = model(temp_input.cuda()).cpu().data

                    outputs.append(temp_output)
                    del temp_input

                predictions = torch.cat(outputs, dim=0)

                pred_assemble = validation_data.transform.transforms[0].assemble(torch.max(predictions, 1)[1], if_itk=False)

                for ind in range(57):
                    dice_all[ind] += metrics.metricEval('dice', pred_assemble == ind, segs.numpy() == ind, num_labels=2)

            dice_all = dice_all / (j + 1)
            dice_avg = np.sum(dice_all) / len(dice_all)

            if config['lr_mode'] == 'plateau':
                scheduler.step(dice_avg, epoch=current_epoch)

            is_best = False
            if dice_avg > best_score:
                is_best = True
                best_score = dice_avg

            if not config['debug_mode']:
                # writer.add_scalar('loss/validation', valid_loss_avg,
                #                   global_step=global_step)  # data grouping by `slash`
                writer.add_scalar('validation/dice_avg', dice_avg, global_step=global_step)
                for e in range(len(dice_all)):
                    writer.add_scalar('validation/dice_'+str(e), dice_all[e], global_step=global_step)

            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: Dice Avg: {:.3f} Dice_lowest: {:.3f} Dice_highest:{:.3f} ({:.3f} sec) {}'.format
                  (current_epoch, dice_avg, np.min(dice_all[1:]),
                   np.max(dice_all[1:]), time.time() - start_time,
                   datetime.datetime.now().strftime("%D %H:%M:%S")))
            total_validation_time += time.time() - start_time

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            save_checkpoint({'epoch': current_epoch,
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_score': best_score},
                            is_best, args.save, config['experiment_name'] + '_' + now_date + '_' + now_time)

        current_epoch += 1

    writer.close()
    print('Finished Training: {}_{}_{}'.format(config['experiment_name'], now_date, now_time))
    print('Total validation time: {}'.format(datetime.timedelta(seconds=total_validation_time)))



# eval input using model by iteratively evaluating subsets of a large patch
def pred_iter(model, input, sub_size=4):
    output = []

    for i in range(0, np.ceil(float(input.size()[0]) / float(sub_size)).astype(int)):
        temp_input = input.narrow(0, sub_size * i,
                                  sub_size if sub_size * (i + 1) <= input.size()[0]
                                  else input.size()[0] - sub_size * i)

        with torch.no_grad():
            temp_input = Variable(temp_input)

        temp_output = model(temp_input.cuda()).data
        output.append(temp_output.cpu())
        del temp_output

    return torch.cat(output, dim=0)


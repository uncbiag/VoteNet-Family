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
import utils.loss as bio_loss
import utils.evalMetrics as metrics


# def weights_init(m):
#     class_name = m.__class__.__name__
#     if class_name.find('Conv') != -1:
#         if not m.weight is None:
#             nn.init.xavier_normal(m.weight.data)
#         if not m.bias is None:
#             nn.init.xavier_normal(m.bias.data)
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
    # name = '_'.join([str(state['epoch']), filename])
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
    Initilaize a model with saved checkpoins, or random values
    :param model: a pytorch model to be initialized
    :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
    :param ckpoint_path: The path of saved checkpoint
    :return: currect epoch and best validation score
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
        # model.apply(weights_init)
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

        # training_data_iter = training_data_loader.__iter__()
        for i, (images, truths, name) in enumerate(training_data_loader):
            # images, truths, name = training_data_iter.__next__()
            global_step = (current_epoch - 1) * iters_per_epoch + (i + 1) * config['batch_size']  # current globel step

            # weight for loss （inverse ratio of all labels）
            if config['if_weight_loss']:
                if config['weight_mode'] == 'dynamic':
                    loss_weight = weight_from_truth(truths, config['n_classes'])
                elif config['weight_mode'] == 'const':
                    loss_weight = config['weight_const']

            model.train()

            # wrap inputs in Variable
            images = Variable(images)
            truths = Variable(truths.long())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(images.cuda())

            # start_time = time.time()
            # print(time.time() - start_time)

            output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, config['n_classes'])
            truths_flat = truths.view(truths.numel())

            # loss = criterion(output_flat, truths_flat)

            if config['loss'] == 'dice':
                loss = bio_loss.dice_loss(output_flat, truths_flat)
            elif config['loss'] == 'cross_entropy':
                loss = F.cross_entropy(output_flat, truths_flat.cuda(),
                                       weight=loss_weight.cuda() if config['if_weight_loss'] else None)
            elif config['loss'] == 'focal_loss':
                loss = FC(output_flat, truths_flat.cuda())
            # del output_flat, truths_flat
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]  # average loss over 10 batches
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
                    writer.add_scalar('loss/training', running_loss / config['print_batch_period'],
                                      global_step=global_step)  # data grouping by `slash`
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                      global_step=global_step)  # data grouping by `slash`
                running_loss = 0.0
                start_time = time.time()  # log running time

            if config['debug_mode']:
                image_summary = make_image_summary(images, truths, output.cpu())

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            image_summary = make_image_summary(images, truths, output.cpu())
            writer.add_image("training", image_summary, global_step=global_step)
        # images, truths = validation_data_loader.

        # del images, truths, output, loss
        # gc.collect()

        # validation
        if current_epoch % config['valid_epoch_period'] == 0:
            model.eval()
            dice_FC = 0
            dice_TC = 0
            start_time = time.time()  # log running time
            for j in range(len(validation_data) if best_score > 0.7 else 2):
                # for j in range(len(validation_data)):
                image_tiles, truths, name = validation_data[j]
                if config['if_weight_loss']:
                    if config['weight_mode'] == 'dynamic':
                        loss_weight = weight_from_truth(truths, config['n_classes'])
                    elif config['weight_mode'] == 'const':
                        loss_weight = config['weight_const']

                valid_batch_size = config['valid_batch_size']
                outputs = []  # a list of lists that store output at each step
                # for i in range(1):
                for i in range(0, np.ceil(image_tiles.size()[0] / valid_batch_size).astype(int)):
                    temp_input = image_tiles.narrow(0, valid_batch_size * i,
                                                    valid_batch_size if valid_batch_size * (i + 1) <=
                                                                        image_tiles.size()[
                                                                            0]
                                                    else image_tiles.size()[0] - valid_batch_size * i)

                    # temp_input = Variable(temp_input, volatile=True)
                    temp_input = Variable(temp_input, volatile=True)

                    # predict through model 1
                    temp_output = model(temp_input.cuda()).cpu().data

                    outputs.append(temp_output)
                    del temp_input

                predictions = torch.cat(outputs, dim=0)

                pred_assemble = validation_data.transform.transforms[0].assemble(torch.max(predictions, 1)[1],
                                                                                 if_itk=False)

                dice_FC += metrics.metricEval('dice', pred_assemble == 1, truths.numpy() == 1,
                                              num_labels=2)
                dice_TC += metrics.metricEval('dice', pred_assemble == 2, truths.numpy() == 2,
                                              num_labels=2)
            dice_FC = dice_FC / (j + 1)
            dice_TC = dice_TC / (j + 1)
            dice_avg = (dice_FC + dice_TC) / 2
            # valid_loss_avg = valid_loss / len(validation_data)

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
                writer.add_scalar('validation/dice_FC', dice_FC, global_step=global_step)
                writer.add_scalar('validation/dice_TC', dice_TC, global_step=global_step)

            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: Dice Avg: {:.3f} Dice_FC: {:.3f} Dice_TC:{:.3f} ({:.3f} sec) {}'.format
                  (current_epoch, dice_avg, dice_FC,
                   dice_TC, time.time() - start_time,
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
    print('Total validation time: {}'.str(datetime.timedelta(seconds=total_validation_time)))


def train_auto_context(pre_model, auto_context_model, training_data_loader,
                       validation_data, optimizer, args, config, scheduler=None):
    # log writer
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)

    if not config['debug_mode']:
        writer = SummaryWriter(os.path.join(args.logdir, now_date, config['experiment_name'] + '_' + now_time))

    # resume checkpoint or initialize
    _, _ = initialize_model(pre_model, ckpoint_path=args.resume1)
    finished_epochs, best_score = initialize_model(auto_context_model, optimizer=optimizer, ckpoint_path=args.resume2)

    current_epoch = finished_epochs + 1

    pre_model.eval()
    auto_context_model.train()

    iters_per_epoch = len(training_data_loader.dataset)
    print("Start Training:")
    while current_epoch <= config['n_epochs']:
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        if not config['lr_mode'] == 'const' and not config['lr_mode'] == 'plateau':
            scheduler.step(epoch=current_epoch)

        for i, (images, truths, name) in enumerate(training_data_loader):
            # images, truths, name = training_data_iter.__next__()
            global_step = (current_epoch - 1) * iters_per_epoch + i + 1  # current globel step

            # weight for loss （inverse ratio of all labels）
            if config['if_weight_loss']:
                if config['weight_mode'] == 'dynamic':
                    loss_weight = weight_from_truth(truths, config['n_classes'])
                elif config['weight_mode'] == 'const':
                    loss_weight = config['weight_const']

            # wrap inputs in Variable
            images = Variable(images, volatile=True)
            truths = Variable(truths.long(), volatile=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # output = forward_auto_context(pre_model, auto_context_model, images, use_residual=config['context_residual'])

            # get output from pre-model
            pre_output = pre_model(images.cuda())
            pre_output.detach()
            pre_output.volatile = False
            if config['if_AC']:
                input_context = Variable(
                    torch.cat([F.softmax(pre_output, dim=1).data, images.data.cuda()], dim=1).cuda(), volatile=False)
            else:
                input_context = Variable(images.data.cuda(), volatile=False)
            # del pre_output
            # forward + backward + optimize
            if config['context_residual'] == 0:
                output = auto_context_model(input_context)
            if config['context_residual'] == 1:
                output = auto_context_model(input_context) * config['residual_scale'] + pre_output
            elif config['context_residual'] == 2:
                output = auto_context_model(input_context) + F.softmax(pre_output, dim=1)
            else:
                ValueError("Wrong context_residual code {}".format(config['context_residual']))

            # compute loss
            output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(output.numel() // config['n_classes'],
                                                                          config['n_classes'])
            truths_flat = truths.view(truths.numel())

            # loss = criterion(output_flat, truths_flat)

            if config['context_residual'] == 2:
                loss = F.mse_loss(output_flat, one_hot_encoder(truths_flat, config['n_classes']).cuda())
            elif config['loss'] == 'dice':
                loss = bio_loss.dice_loss(output_flat, truths_flat)
            elif config['loss'] == 'cross_entropy':
                loss = F.cross_entropy(output_flat, truths_flat.cuda(),
                                       weight=loss_weight.cuda() if config['if_weight_loss'] else None)
            elif config['loss'] == 'focal_loss':
                FC = bio_loss.FocalLoss(config['n_classes'])
                loss = FC(output_flat, truths_flat.cuda())


            # del output_flat, truths_flat
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]  # average loss over 10 batches
            if i % config['print_batch_period'] == config['print_batch_period'] - 1:  # print every 10 mini-batches
                duration = time.time() - start_time
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.4f} lr:{} ({:.3f} sec/batch) {}'.format
                      (current_epoch, (i + 1) * config['batch_size'], iters_per_epoch,
                       (i + 1) * config['batch_size'] / iters_per_epoch * 100,
                       running_loss / config['print_batch_period'] if i > 0 else running_loss,
                       optimizer.param_groups[0]['lr'],
                       duration / config['print_batch_period'], datetime.datetime.now().strftime("%D %H:%M:%S")))

                if not config['debug_mode']:
                    writer.add_scalar('loss/training', running_loss / config['print_batch_period'],
                                      global_step=global_step)  # data grouping by `slash`
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                      global_step=global_step)  # data grouping by `slash`

                running_loss = 0.0
                start_time = time.time()  # log running time

            if (not config['debug_mode']) and current_epoch % config['valid_epoch_period'] == 0 and i == 0:
                image_summary = make_image_summary(images.cpu(), truths.cpu(), output.cpu())
                writer.add_image("training", image_summary, global_step=global_step)

            del images, truths, output, loss
            gc.collect()

            # del images, truths, output, loss
            # gc.collect()

        # validation
        if current_epoch % config['valid_epoch_period'] == 0:
            auto_context_model.eval()
            dice_FC = 0
            dice_TC = 0
            start_time = time.time()  # log running time
            for j in range(len(validation_data)):
                image_tiles, truths, name = validation_data[j]
                if config['if_weight_loss']:
                    if config['weight_mode'] == 'dynamic':
                        loss_weight = weight_from_truth(truths, config['n_classes'])
                    elif config['weight_mode'] == 'const':
                        loss_weight = config['weight_const']

                test_batch_size = config['valid_batch_size']
                outputs = []  # a list of lists that store output at each step
                # for i in range(1):
                for i in range(0, np.ceil(image_tiles.size()[0] / test_batch_size).astype(int)):

                    temp_input = image_tiles.narrow(0, test_batch_size * i,
                                                    test_batch_size if test_batch_size * (i + 1) <= image_tiles.size()[
                                                        0]
                                                    else image_tiles.size()[0] - test_batch_size * i)

                    # temp_input = Variable(temp_input, volatile=True)
                    temp_input = Variable(temp_input, volatile=True)

                    # predict through model 1
                    temp_output = pre_model(temp_input.cuda()).cpu()
                    # outputs[0].append(F.softmax(temp_output, dim=1).data)

                    # predict through model 2
                    if config['if_AC']:
                        temp_context_input = Variable(
                            torch.cat([F.softmax(temp_output, dim=1).data, temp_input.data], dim=1), volatile=True)
                    else:
                        temp_context_input = Variable(temp_input.data.cuda(), volatile=True)

                    # temp_context_input = Variable(torch.cat([temp_output, temp_input.data.cuda()], dim=1), volatile=True)
                    if config['context_residual'] == 0:
                        temp_output = F.softmax(auto_context_model(temp_context_input.cuda()), dim=1).cpu().data
                    elif config['context_residual'] == 1:
                        temp_output = F.softmax(auto_context_model(temp_context_input.cuda()) * config[
                            'residual_scale'] + temp_output.cuda(),
                                                dim=1).cpu().data
                    elif config['context_residual'] == 2:
                        temp_output = F.softmax(temp_output, dim=1).data + auto_context_model(
                            temp_context_input.cuda()).cpu().data
                    else:
                        ValueError("Wrong context_residual code {}".format(config['context_residual']))

                    outputs.append(temp_output)
                    del temp_output, temp_input

                predictions = torch.cat(outputs, dim=0)

                # predictions = pred_iter_auto_context(pre_model, auto_context_model, image_tiles,
                #                                      iterations=1, sub_size=6,
                #                                      residual=config['context_residual'])[1]
                pred_assemble = validation_data.transform.transforms[0].assemble(torch.max(predictions, 1)[1],
                                                                                 if_itk=False)

                dice_FC += metrics.metricEval('dice', pred_assemble == 1, truths.numpy() == 1,
                                              num_labels=2)
                dice_TC += metrics.metricEval('dice', pred_assemble == 2, truths.numpy() == 2,
                                              num_labels=2)
            dice_FC = dice_FC / len(validation_data)
            dice_TC = dice_TC / len(validation_data)
            dice_avg = (dice_FC + dice_TC) / 2
            # valid_loss_avg = valid_loss / len(validation_data)

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
                writer.add_scalar('validation/dice_FC', dice_FC, global_step=global_step)
                writer.add_scalar('validation/dice_TC', dice_TC, global_step=global_step)

            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: Dice Avg: {:.3f} Dice_FC: {:.3f} Dice_TC:{:.3f} ({:.3f} sec) {}'.format
                  (current_epoch, dice_avg, dice_FC,
                   dice_TC, time.time() - start_time,
                   datetime.datetime.now().strftime("%D %H:%M:%S")))

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            save_checkpoint({'epoch': current_epoch,
                             'model_state_dict': auto_context_model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_score': best_score},
                            is_best, args.save, config['experiment_name'] + '_' + now_date + '_' + now_time)

        current_epoch += 1

    writer.close()
    print('Finished Training: {}_{}_{}'.format(config['experiment_name'], now_date, now_time))


# eval input using model by iteratively evaluating subsets of a large patch
def pred_iter_auto_context(pre_model, auto_context_model, input, iterations=1, sub_size=4, residual=False):
    outputs = [[] for i in range(iterations + 1)]  # a list of lists that store output at each step
    # for i in range(1):
    for i in range(0, np.ceil(input.size()[0] / sub_size).astype(int)):

        temp_input = input.narrow(0, sub_size * i,
                                  sub_size if sub_size * (i + 1) <= input.size()[0]
                                  else input.size()[0] - sub_size * i)

        # temp_input = Variable(temp_input, volatile=True)
        temp_input = Variable(temp_input, volatile=True)

        # predict through model 1
        temp_output = pre_model(temp_input.cuda()).cpu()
        outputs[0].append(F.softmax(temp_output, dim=1).data)

        # predict through model 2 iteratively
        for j in range(1, iterations + 1):
            temp_context_input = Variable(torch.cat([F.softmax(temp_output, dim=1).data, temp_input.data], dim=1),
                                          volatile=True)
            # temp_context_input = Variable(torch.cat([temp_output, temp_input.data.cuda()], dim=1), volatile=True)
            if residual:
                temp_output = F.softmax(auto_context_model(temp_context_input.cuda()) + temp_output.cuda(),
                                        dim=1).cpu().data
                # temp_output = F.softmax(auto_context_model(temp_context_input) +\
                #               temp_context_input[:, :-1, :, :, :], dim=1).data
            else:
                temp_output = F.softmax(auto_context_model(temp_context_input), dim=1).cpu().data
            outputs[j].append(temp_output)

    return [torch.cat(outputs[j], dim=0) for j in range(iterations + 1)]


def forward_auto_context(pre_model, auto_context_model, images, use_residual):
    """
    One step forward for auto-context model
    :param pre_model: the pre-model that predicts probability from only image
    :param auto_context_model: the auto-context model that refines probability from both image and prior
    :param images(Variable, NxCxHxWxH): image batches
    :param use_residual(bool): if the auto-context model generate residual that will be added to the output of pre-model
    :return:output(Variable, NxCxHxWxH)
    """

    # # get output from pre-model
    # pre_output = pre_model(images)
    # pre_output_prop = F.softmax(pre_output, dim=1).data
    # # pre_output.detach_()
    # # make input varibale for auto-context model
    # input_context = Variable(torch.cat([pre_output_prop, images.data], dim=1), volatile=False).cuda()
    #
    # # forward
    # if use_residual:
    #     pre_output.volatile = False
    #     output = auto_context_model(input_context) + pre_output
    #     del pre_output
    # else:
    #     output = auto_context_model(input_context)

    # get output from pre-model
    pre_output = F.softmax(pre_model(images), dim=1)

    input_context = Variable(torch.cat([pre_output.data, images.data], dim=1).cuda(), volatile=False)
    del pre_output
    # forward + backward + optimize
    if use_residual:
        output = auto_context_model(input_context) + input_context[:, :-1, :, :, :]
    else:
        output = auto_context_model(input_context)

    return output


def train_auto_context_rnn(pre_model, auto_context_model, training_data_loader,
                           validation_data, optimizer, args, config, scheduler=None):
    # log writer
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)

    if not config['debug_mode']:
        writer = SummaryWriter(os.path.join(args.logdir, now_date, config['experiment_name'] + '_' + now_time))

    # resume checkpoint or initialize
    _, _ = initialize_model(pre_model, ckpoint_path=args.resume1)
    finished_epochs, best_score = initialize_model(auto_context_model, optimizer=optimizer, ckpoint_path=args.resume2)

    current_epoch = finished_epochs + 1

    pre_model.eval()
    auto_context_model.train()

    iters_per_epoch = len(training_data_loader.dataset)
    print("Start Training:")
    while current_epoch <= config['n_epochs']:
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        if not config['lr_mode'] == 'const' and not config['lr_mode'] == 'plateau':
            scheduler.step(epoch=current_epoch)

        for i, (images, truths, name) in enumerate(training_data_loader):
            # images, truths, name = training_data_iter.__next__()
            global_step = (current_epoch - 1) * iters_per_epoch + i + 1  # current globel step

            # weight for loss （inverse ratio of all labels）
            if config['if_weight_loss']:
                if config['weight_mode'] == 'dynamic':
                    loss_weight = weight_from_truth(truths, config['n_classes'])
                elif config['weight_mode'] == 'const':
                    loss_weight = config['weight_const']

            # wrap inputs in Variable
            images = Variable(images, volatile=False)
            truths = Variable(truths.long(), volatile=False)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss = 0

            # prediction of premodel
            hidden = pre_model(images.cuda())

            # forward + backward + optimize
            # rnn run in loops
            for step in range(config['auto_context_steps']):
                output, hidden = auto_context_model(images.cuda(), hidden)

                # compute loss
                output_flat = hidden.permute(0, 2, 3, 4, 1).contiguous().view(-1, config['n_classes'])
                truths_flat = truths.view(-1)

                # loss = criterion(output_flat, truths_flat)
                if config['loss'] == 'dice':
                    loss += bio_loss.dice_loss(output_flat, truths_flat)
                elif config['loss'] == 'cross_entropy':
                    loss += F.cross_entropy(output_flat, truths_flat.cuda(),
                                           weight=loss_weight.cuda() if config['if_weight_loss'] else None)
                elif config['loss'] == 'focal_loss':
                    FC = bio_loss.FocalLoss(config['n_classes'])
                    loss += FC(output_flat, truths_flat.cuda())

            # del output_flat, truths_flat
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]  # average loss over 10 batches
            if i % config['print_batch_period'] == config['print_batch_period'] - 1:  # print every 10 mini-batches
                duration = time.time() - start_time
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.3f} lr:{} ({:.3f} sec/batch) {}'.format
                      (current_epoch, (i + 1) * config['batch_size'], iters_per_epoch,
                       (i + 1) * config['batch_size'] / iters_per_epoch * 100,
                       running_loss / config['print_batch_period'] if i > 0 else running_loss,
                       optimizer.param_groups[0]['lr'],
                       duration / config['print_batch_period'], datetime.datetime.now().strftime("%D %H:%M:%S")))

                if not config['debug_mode']:
                    writer.add_scalar('loss/training', running_loss / config['print_batch_period'],
                                      global_step=global_step)  # data grouping by `slash`
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                      global_step=global_step)  # data grouping by `slash`

                running_loss = 0.0
                start_time = time.time()  # log running time

            if not config['debug_mode'] and current_epoch % config['valid_epoch_period'] == 0 and i == 0:
                image_summary = make_image_summary(images.cpu(), truths.cpu(), output.cpu())
                writer.add_image("training", image_summary, global_step=global_step)

            del images, truths, output, loss
            gc.collect()

        # validation
        if current_epoch % config['valid_epoch_period'] == 0:
            auto_context_model.eval()
            dice_FC = 0
            dice_TC = 0
            start_time = time.time()  # log running time
            for j in range(len(validation_data)):
                image_tiles, truths, name = validation_data[j]
                if config['if_weight_loss']:
                    if config['weight_mode'] == 'dynamic':
                        loss_weight = weight_from_truth(truths, config['n_classes'])
                    elif config['weight_mode'] == 'const':
                        loss_weight = config['weight_const']

                test_batch_size = config['valid_batch_size']
                outputs = []

                for i in range(0, np.ceil(image_tiles.size()[0] / test_batch_size).astype(int)):

                    temp_input = image_tiles.narrow(0, test_batch_size * i,
                                                    test_batch_size if test_batch_size * (i + 1) <= image_tiles.size()[
                                                        0]
                                                    else image_tiles.size()[0] - test_batch_size * i)

                    # temp_input = Variable(temp_input, volatile=True)
                    temp_input = Variable(temp_input, volatile=True).cuda()

                    # predict through model 1
                    temp_hidden = pre_model(temp_input)

                    # predict through model 2
                    for step in range(config['auto_context_steps']):
                        temp_output, temp_hidden = auto_context_model(temp_input, temp_hidden)

                    outputs.append(temp_output.data.cpu())
                    del temp_output, temp_input

                predictions = torch.cat(outputs, dim=0)

                pred_assemble = validation_data.transform.transforms[0].assemble(torch.max(predictions, 1)[1],
                                                                                 if_itk=False)

                dice_FC += metrics.metricEval('dice', pred_assemble == 1, truths.numpy() == 1,
                                              num_labels=2)
                dice_TC += metrics.metricEval('dice', pred_assemble == 2, truths.numpy() == 2,
                                              num_labels=2)

            dice_FC = dice_FC / len(validation_data)
            dice_TC = dice_TC / len(validation_data)
            dice_avg = (dice_FC + dice_TC) / 2

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
                writer.add_scalar('validation/dice_FC', dice_FC, global_step=global_step)
                writer.add_scalar('validation/dice_TC', dice_TC, global_step=global_step)

            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: Dice Avg: {:.3f} Dice_FC: {:.3f} Dice_TC:{:.3f} ({:.3f} sec) {}'.format
                  (current_epoch, dice_avg, dice_FC,
                   dice_TC, time.time() - start_time,
                   datetime.datetime.now().strftime("%D %H:%M:%S")))

            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            save_checkpoint({'epoch': current_epoch,
                             'model_state_dict': auto_context_model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_score': best_score},
                            is_best, args.save, config['experiment_name'] + '_' + now_date + '_' + now_time)

        current_epoch += 1

    writer.close()
    print('Finished Training: {}_{}_{}'.format(config['experiment_name'], now_date, now_time))


# eval input using model by iteratively evaluating subsets of a large patch
def pred_iter(model, input, sub_size=4):
    output = []

    for i in range(0, np.ceil(input.size()[0] / sub_size).astype(int)):
        temp_input = input.narrow(0, sub_size * i,
                                  sub_size if sub_size * (i + 1) <= input.size()[0]
                                  else input.size()[0] - sub_size * i)

        temp_input = Variable(temp_input, volatile=True)

        temp_output = model(temp_input.cuda()).data
        output.append(temp_output.cpu())
        del temp_output

    return torch.cat(output, dim=0)


def train_cascaded(cascaded_model, training_data_loader, validation_data, optimizer, args, config, scheduler=None):
    # log writer
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    total_validation_time = 0
    total_training_time = 0

    if not config['debug_mode']:
        writer = SummaryWriter(os.path.join(args.logdir, now_date, config['experiment_name'] + '_' + now_time))

    # resume checkpoint or initialize
    print("Initialize models")
    num_cascaded_models_resume = 0
    if config['ckpoint']:
        params = config['ckpoint'].split('/')[-2].split('_')
        # print(params)
        num_cascaded_models_resume = int(params[params.index('Cascaded') + 1])

    # load premodels if not endtoend training
    if num_cascaded_models_resume < config['num_cascaded_models']:
        # if num_cascaded_models_resume == 1:
        #     _, best_score = initialize_model(cascaded_model.models[0], optimizer=None,
        #                                                    ckpoint_path=config['ckpoint'])
        # else:
        _, best_score = initialize_model(cascaded_model, optimizer=None, ckpoint_path=config['ckpoint'])

    # reload interrupted training
    else:
        if config['restore_optimizer']:
            finished_epochs, best_score = initialize_model(cascaded_model, optimizer=optimizer,
                                                           ckpoint_path=config['ckpoint'])
        else:
            finished_epochs, best_score = initialize_model(cascaded_model, optimizer=None,
                                                           ckpoint_path=config['ckpoint'])
    print("Previous best validation score: {}".format(best_score))

    current_epoch = finished_epochs + 1
    iters_per_epoch = len(training_data_loader.dataset)

    print("Start Training:")
    while current_epoch <= config['n_epochs']:
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        if not config['lr_mode'] == 'const' and not config['lr_mode'] == 'plateau':
            scheduler.step(epoch=current_epoch)

        cascaded_model.cascaded_train()

        for batch_ind, (images, truths, name) in enumerate(training_data_loader):
            # images, truths, name = training_data_iter.__next__()
            global_step = (current_epoch - 1) * iters_per_epoch + batch_ind + 1  # current globel step

            # weight for loss （inverse ratio of all labels）
            if config['if_weight_loss']:
                if config['weight_mode'] == 'dynamic':
                    loss_weight = weight_from_truth(truths, config['n_classes'])
                elif config['weight_mode'] == 'const':
                    loss_weight = config['weight_const']

            # wrap inputs in Variable
            images = Variable(images, volatile=False)
            truths = Variable(truths.long(), volatile=False)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss = 0

            # forward + backward + optimize
            output = cascaded_model(images.cuda(), train=True,
                                    multi_output=config['if_end2end'] and config['if_multi_output'])

            # compute loss
            if config['loss'] == 'dice':
                loss_F = bio_loss.DiceLoss()
            elif config['loss'] == 'cross_entropy':
                loss_F = nn.CrossEntropyLoss(weight=loss_weight.cuda() if config['if_weight_loss'] else None)
            elif config['loss'] == 'focal_loss':
                loss_F = bio_loss.FocalLoss(config['n_classes'])

            truths_flat = truths.view(-1)
            if config['if_end2end'] and config['if_multi_output']:
                for temp_output in output:
                    temp_output_flat = temp_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, config['n_classes'])
                    loss += loss_F(temp_output_flat, truths_flat.cuda())
                    del temp_output
            else:
                output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, config['n_classes'])
                loss += loss_F(output_flat, truths_flat.cuda())

            # loss = criterion(output_flat, truths_flat)
            # if config['loss'] == 'dice':
            #     loss += bio_loss.dice_loss(output_flat, truths_flat)
            # elif config['loss'] == 'cross_entropy':
            #     loss += F.cross_entropy(output_flat, truths_flat.cuda(),
            #                            weight=loss_weight.cuda() if config['if_weight_loss'] else None)
            # elif config['loss'] == 'focal_loss':
            #     FC = bio_loss.FocalLoss(config['n_classes'])
            #     loss += FC(output_flat, truths_flat.cuda())

            # del output_flat, truths_flat
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]  # average loss over 10 batches
            if batch_ind % config['print_batch_period'] == config[
                'print_batch_period'] - 1:  # print every 10 mini-batches
                duration = time.time() - start_time
                total_training_time += duration
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.3f} lr:{} time:{:.0f} sec ({:.3f} sec/batch) {}'.format
                      (current_epoch, (batch_ind + 1) * config['batch_size'], iters_per_epoch,
                       (batch_ind + 1) * config['batch_size'] / iters_per_epoch * 100,
                       running_loss / config['print_batch_period'] if batch_ind > 0 else running_loss,
                       optimizer.param_groups[0]['lr'], duration,
                       duration / config['print_batch_period'], datetime.datetime.now().strftime("%D %H:%M:%S")))

                if not config['debug_mode']:
                    writer.add_scalar('loss/training', running_loss / config['print_batch_period'],
                                      global_step=global_step)  # data grouping by `slash`
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                      global_step=global_step)  # data grouping by `slash`

                running_loss = 0.0
                start_time = time.time()  # log running time

            if config['debug_mode']:
                image_summary = make_image_summary(images.cpu(), truths.cpu(), output[-1].cpu()
                if config['if_end2end'] and config['if_multi_output'] else output.cpu())

            if not config['debug_mode'] and current_epoch % config['valid_epoch_period'] == 0 and batch_ind == 0:
                image_summary = make_image_summary(images.cpu(), truths.cpu(), output[-1].cpu()
                if config['if_end2end'] and config['if_multi_output'] else output.cpu())
                writer.add_image("training", image_summary, global_step=global_step)

            del images, truths, output, loss
            gc.collect()

        # validation
        if current_epoch % config['valid_epoch_period'] == 0 and (
                current_epoch > config['n_epochs'] // 3 or config['debug_mode']):
            cascaded_model.cascaded_eval()
            dice_FC = 0
            dice_TC = 0
            start_time = time.time()  # log running time
            valid_threshold = 0.81 + 0.01 * config['num_cascaded_models'] if 'UNet_light4' in config['model'] else 0.85
            for valid_ind in range(len(validation_data) if best_score > valid_threshold else 2):
                image_tiles, truths, name = validation_data[valid_ind]
                if config['if_weight_loss']:
                    if config['weight_mode'] == 'dynamic':
                        loss_weight = weight_from_truth(truths, config['n_classes'])
                    elif config['weight_mode'] == 'const':
                        loss_weight = config['weight_const']

                test_batch_size = config['valid_batch_size']
                outputs = []

                for batch_ind in range(0, np.ceil(image_tiles.size()[0] / test_batch_size).astype(int)):
                    temp_input = image_tiles.narrow(0, test_batch_size * batch_ind,
                                                    test_batch_size if test_batch_size * (batch_ind + 1) <=
                                                                       image_tiles.size()[0]
                                                    else image_tiles.size()[0] - test_batch_size * batch_ind)

                    # temp_input = Variable(temp_input, volatile=True)
                    temp_input = Variable(temp_input, volatile=True).cuda()
                    temp_output = cascaded_model(temp_input)

                    outputs.append(temp_output.data.cpu())
                    del temp_output, temp_input

                predictions = torch.cat(outputs, dim=0)

                pred_assemble = validation_data.transform.transforms[0].assemble(torch.max(predictions, 1)[1],
                                                                                 if_itk=False)

                crop_size = config['overlap_size']
                pred_assemble = pred_assemble[crop_size[2]:-crop_size[2], crop_size[0]:-crop_size[0],
                                crop_size[1]:-crop_size[1]]
                truths = truths.numpy()[:, crop_size[2]:-crop_size[2], crop_size[0]:-crop_size[0],
                         crop_size[1]:-crop_size[1]]

                dice_FC += metrics.metricEval('dice', pred_assemble == 1, truths == 1,
                                              num_labels=2)
                dice_TC += metrics.metricEval('dice', pred_assemble == 2, truths == 2,
                                              num_labels=2)

            dice_FC = dice_FC / (valid_ind + 1)
            dice_TC = dice_TC / (valid_ind + 1)
            dice_avg = (dice_FC + dice_TC) / 2

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
                writer.add_scalar('validation/dice_FC', dice_FC, global_step=global_step)
                writer.add_scalar('validation/dice_TC', dice_TC, global_step=global_step)

            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: Dice Avg: {:.3f} Dice_FC: {:.3f} Dice_TC:{:.3f} ({:.3f} sec) {}'.format
                  (current_epoch, dice_avg, dice_FC,
                   dice_TC, time.time() - start_time,
                   datetime.datetime.now().strftime("%D %H:%M:%S")))
            total_validation_time += time.time() - start_time
            # image_summary = make_image_summary(images, truths, output)
            # writer.add_image("validation", image_summary, global_step=global_step)

        if not config['debug_mode'] and current_epoch % config['save_ckpts_epoch_period'] == 0:
            save_checkpoint({'epoch': current_epoch,
                             'model_state_dict': cascaded_model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'best_score': best_score},
                            is_best, args.save, config['experiment_name'] + '_' + now_date + '_' + now_time)

        current_epoch += 1

    if not config['debug_mode']:
        writer.close()
    print('Finished Training: {}_{}_{}'.format(config['experiment_name'], now_date, now_time))
    print('Total training time: {}'.format(str(datetime.timedelta(seconds=total_training_time))))
    print('Total validation time: {}'.format(str(datetime.timedelta(seconds=total_validation_time))))
    return os.path.join(args.save, '{}_{}_{}'.format(config['experiment_name'], now_date, now_time))

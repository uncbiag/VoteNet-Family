import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from itertools import repeat
import numpy as np

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

# TODO: dice loss for multi-class

class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input, None

def dice_loss(input, target):
    return DiceLoss()(input, target)

def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU


def mydice_loss(pred, target):
    """
        This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
    """
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    # iflat = pred.contiguous().view(-1)
    # tflat = target.contiguous().view(-1)
    iflat = F.sigmoid(pred.squeeze())
    tflat = target
    intersection = (iflat * tflat).sum()
    # A_sum = torch.sum(iflat * iflat)
    # B_sum = torch.sum(tflat * tflat)
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def mydice_loss2(pred, target):
    """
        This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
    """
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    # iflat = pred.contiguous().view(-1)
    # tflat = target.contiguous().view(-1)
    iflat = F.sigmoid(pred.squeeze())
    tflat = target
    intersection_1 = (iflat * tflat).sum()
    intersection_0 = ((1.0 - iflat) * (1.0 - tflat)).sum()
    # A_sum = torch.sum(iflat * iflat)
    # B_sum = torch.sum(tflat * tflat)
    A_sum_1 = torch.sum(iflat)
    A_sum_0 = torch.sum(1.0 - iflat)
    B_sum_1 = torch.sum(tflat)
    B_sum_0 = torch.sum(1.0 - tflat)

    return 1 - ((intersection_1 + smooth) / (A_sum_1 + B_sum_1 + smooth)) - ((intersection_0 + smooth) / (A_sum_0 + B_sum_0 + smooth))

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score



class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, softmax=False, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.softmax = softmax
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        pass

    def forward(self, pred, truth):
        """

        :param pred: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param truth: Tensor, ground truth mask when of size BxDxMxN;
                             or class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert pred.shape[0] == truth.shape[0]
        assert pred.shape[-3:] == truth.squeeze().shape[-3:]

        shape = list(pred.shape)

        if self.softmax:
            pred = F.softmax(pred, dim=1)

        # flat the spatial dimensions
        pred_flat = pred.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if len(truth.shape) == len(shape)-1:
            truth_flat = mask_to_one_hot(truth.view(shape[0], 1, -1), shape[1])
        elif truth.shape[1] == shape[1]:
            truth_flat = truth.view(shape[0], shape[1], -1)
        else:
            truth_flat = None
            raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(truth.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))
        # True positive
        tp = torch.sum(pred_flat * truth_flat, dim=2)

        # Weighted sum of false negtive and false positive
        fp_and_fn = self.alpha * torch.sum(pred_flat * (1 - truth_flat), 2) + \
                    self.beta * torch.sum((1 - pred_flat) * truth_flat, 2)

        # Tversky score
        score = (tp + self.eps) / ((tp + self.eps) + fp_and_fn)

        return 1 - score.mean()



class FocalLoss(nn.Module):
    """
        This criterion is an implementation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double): gamma > 0; reduces the relative loss for well-classified examples (p > .5),
                                    putting more focus on hard, misclassified examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        """

        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        # inputs = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, inputs.size(1))
        # targets = targets.view(-1)

        # TODO avoid softmax in focal loss
        P = F.softmax(inputs, dim=1)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[targets.data.view(-1)].view(-1)

        log_p = - F.cross_entropy(inputs, targets, reduce=False)
        probs = F.nll_loss(P, targets, reduce=False)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalLoss2(nn.Module):
    """
    https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    """

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)
import torch
from torch import nn, optim
from torch.nn import functional as F
from tools import *
import Transform as bio_transform
from Dataset import *
import SimpleITK as sitk
import numpy as np

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature_a = nn.Parameter(torch.ones(1)*(0.845))
        self.temperature_b = nn.Parameter(torch.ones(1)*(0.018))

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature_a = self.temperature_a.expand(logits.size()).double()
        temperature_b = self.temperature_b.expand(logits.size()).double()
        return torch.mul(logits, temperature_a) + temperature_b

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        testing_list_file = os.path.realpath("../Data/LPBA40_data_sep/LPBA40_fold_1_validate_VoteNet.txt")
        data_dir = os.path.realpath("../Data/LPBA40/")
        overlap_size = (16, 16, 16)
        patch_size = (72, 72, 72)
        padding_mode = 'reflect'
        partition = bio_transform.Partition(patch_size, overlap_size, padding_mode=padding_mode, mode='pred')
        test_transforms = [partition]
        test_transform = transforms.Compose(test_transforms)
        testing_data = NiftiDataset(testing_list_file, data_dir, "training", transform=test_transform)

        self.cuda()
        # nll_criterion = nn.CrossEntropyLoss().cuda()
        bnll_criterion = nn.BCEWithLogitsLoss().cuda()
        # ece_criterion = _ECELoss().cuda()

        print("Collecting Logits")
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []

        # add mask to remove background info
        I_18_mask = sitk.GetArrayFromImage(sitk.ReadImage('/playpen/zpd/remote/MAS/Data/LPBA40/corr_label/s18.nii'))
        I_18_mask[I_18_mask != 0] = 1
        I_19_mask = sitk.GetArrayFromImage(sitk.ReadImage('/playpen/zpd/remote/MAS/Data/LPBA40/corr_label/s19.nii'))
        I_19_mask[I_19_mask != 0] = 1
        I_20_mask = sitk.GetArrayFromImage(sitk.ReadImage('/playpen/zpd/remote/MAS/Data/LPBA40/corr_label/s20.nii'))
        I_20_mask[I_20_mask != 0] = 1
        I_universal_mask = I_18_mask + I_19_mask + I_20_mask
        I_universal_mask = I_universal_mask.reshape((1, -1))

        with torch.no_grad():
            for i in range(len(testing_data)):
                src, seg, tar, name = testing_data[i]
                print(name)
                cat_input = torch.cat((src, tar), 1)
                prediction = pred_iter(self.model, cat_input, sub_size=4)
                logits = partition.assemble(prediction.squeeze(), is_vote=False, if_itk=False)
                logits = logits.reshape((1, -1))
                logits = logits[I_universal_mask > 0]
                seg = seg.cpu().detach().numpy()
                seg = seg.reshape((1, -1))
                seg = seg[I_universal_mask > 0]
                logits = torch.from_numpy(logits)
                seg = torch.from_numpy(seg)
                logits_list.append(logits.double())
                labels_list.append(seg.double())
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        print("Computing Loss")
        # Calculate NLL and ECE before temperature scaling
        before_temperature_bnll = bnll_criterion(logits, labels).item()
        # before_temperature_ece = ece_criterion(logits, labels).item()
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        print('Before temperature - BNLL: %.5f' % (before_temperature_bnll))

        # Next: optimize the temperature w.r.t. NLL
        # optimizer = optim.LBFGS([self.temperature_a, self.temperature_b], lr=0.002, max_iter=200)
        optimizer = optim.Adam([self.temperature_a, self.temperature_b], lr=0.00001)

        def eval():
            loss = bnll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        for i in range(2000):
            optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_bnll = bnll_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.5f, %.5f' % (self.temperature_a.item(), self.temperature_b.item()))
        print('After temperature - BNLL: %.5f' % (after_temperature_bnll))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

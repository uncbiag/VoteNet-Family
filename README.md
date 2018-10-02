# multi-atlas-segmentation

## 0. Data augmentation and preprocessing

MGH10, CUMC12, IBSR18, LPBA40, OASIS
Currently, data augmentation step is hold on. 

LPBA40 dataset is used for training a U-net both as a segmentation net and as a baseline for segmentation prediction.

OASIS dataset is used to train a fast registration prediction net.  

Preprocessing:

- All brain images are affine transformed, histogram equalized and registrated to a common atlas space.
- LPBA40 labels are reorganized into continuous integer from 0 (background) to 56.
- There is an additional label (taged 116) in s37.nii. Currently set it to 0 (background).
 
## 1. U-Net for segmentation (baseline)
5-fold cross validation is used. 28 images for training, 4 images for validation, 8 images for testing. 5 folds sum up to cover all images in LPBA40 dataset. 

## 2. Registration Net for fast registration prediction

## 3. Atlas Selection (Dice Score)

## 4. Multi-atlas segmentation
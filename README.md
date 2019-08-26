# VoteNet: A Deep Learning Label Fusion Method for Multi-Atlas Segmentation

Key points:
- The registration network solves the computational bottleneck of MAS (Quicksilver currently)
- The VoteNet provides locally binary prediction whether to use trust the voxel or not from each atlas


## 0. Data augmentation and preprocessing

The first version of VoteNet using [LPBA40](https://www.loni.usc.edu/research/atlases) for VoteNet, and [OASIS](https://www.oasis-brains.org) for registration network.
Currently, data augmentation step is hold on. Based on the previous work done by Dr. Xiao Yang, we already can get a good prediction result compared with those of optimization-based registration methods.

- LPBA40 dataset is also used for training a U-net as a baseline for segmentation prediction.

- OASIS dataset is used to train a fast registration prediction net as described [here](https://github.com/uncbiag/quicksilver).


Preprocessing:

- All brain images are affine transformed, histogram equalized and registered to a common atlas space ([ICBM brain template](https://www.ncbi.nlm.nih.gov/pubmed/9343592)).
- LPBA40 labels are reorganized into continuous integer from 0 (background) to 56.
- There is an additional label (taged 116) in s37.nii. Currently set it to 0 (background).

 
## 1. VoteNet
2-fold cross validation is used. 17 images for training, 3 images for validation, 20 images for testing. 2 folds sum up to cover all images in LPBA40 dataset. 

Note:

1. There is a problem with random cropping of the images in dataloader with multiple processors due to np.random generates the same random numbers for each data batch. 
This is because numpy doesn't properly handle RNG states when fork subprocesses. It's numpy's issue with multiprocessing tracked at numpy/numpy#9248). 
Thus, currently, using one worker in dataloader to extract patches.
Update: this issue is fixed by remove the random state generate in random_crop_3d function. Now, multiple workers can be used to load training data.

## 2. Registration Net for fast registration prediction
Please refer to Dr. Xiao Yang's [Quicksilver](https://github.com/uncbiag/quicksilver) project.

 
## 3. Multi-atlas segmentation
First using registration network to fast predict the deformation of each atlas to the target image. Secondly, wraping the atlas image and segmentation using deformation field predicted from the first step.
Then using VoteNet to filter out bad voxels in each warped atlas segmentation. Finally doing plural voting on the filtered warped atlas segmentation to get the final results.
![Pipeline](./img/pineline_2.png)


## 4. Highlight

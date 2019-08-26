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
The architecture of VoteNet is as below.
![Architecture](/img/votenet.png)


## 2. Registration Net for fast registration prediction
Please refer to Dr. Xiao Yang's [Quicksilver](https://github.com/uncbiag/quicksilver) project.

 
## 3. Multi-atlas segmentation
First using registration network to fast predict the deformation of each atlas to the target image. Secondly, wraping the atlas image and segmentation using deformation field predicted from the first step.
Then using VoteNet to filter out bad voxels in each warped atlas segmentation. Finally doing plural voting on the filtered warped atlas segmentation to get the final results.
A visual illustration is as below.
![Pipeline](/img/pipeline_2.png)


## 4. Highlight
In the following image, the left two are ground truth votenet results and predicted. The right two are the recall map of voting.
It is obvious that after removing the bad voxels the probability to get correct label is also increasing. 
![Result](/img/prob_2.png)
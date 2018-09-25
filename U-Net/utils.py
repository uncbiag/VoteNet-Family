import SimpleITK as sitk
import numpy as np

def get_label_intensity(label):
    L = sitk.GetArrayFromImage(label)

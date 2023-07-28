"""
3D查看MRI图像
"""
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D


img_path = "./data/ct_scans/coronacases_org_001.nii"

img = nib.load(img_path)

width, height, queue =img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()

"""
对于医学影像图片进行3D查看
3D 查看 MRI 图像
"""
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D


img_path = './data/imagesTr/la_003.nii.gz'
img = nib.load(img_path)

width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()





































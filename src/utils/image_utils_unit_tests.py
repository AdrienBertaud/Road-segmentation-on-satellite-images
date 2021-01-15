import numpy as np
import numpy.testing as npt
from image_utils import img_crop_, build_from_patches, build_mask_taking_min, build_mask_taking_mean, load_images

im = np.zeros((608,608))
im[399,399]=1
im[607,607]=1
im[607,0]=1
im[0,607]=1

list_patches = img_crop_(im,400,400)

npt.assert_equal(list_patches[0][399,399],1)
npt.assert_equal(list_patches[1][399,399-208],1)
npt.assert_equal(list_patches[2][399-208,399],1)
npt.assert_equal(list_patches[3][399-208,399-208],1)
npt.assert_equal(list_patches[1][0,399],1)
npt.assert_equal(list_patches[2][399,0],1)
npt.assert_equal(list_patches[3][399,399],1)

list_patches[0][10,10] = 1
list_patches[0][300,300] = 1

rebuild_img = build_from_patches(list_patches, 608, 608)

npt.assert_equal(rebuild_img[399,399],1)
npt.assert_equal(rebuild_img[0,607],1)
npt.assert_equal(rebuild_img[607,0],1)
npt.assert_equal(rebuild_img[607,607],1)
npt.assert_equal(rebuild_img[10,10],1)
npt.assert_equal(rebuild_img[300,300],1)
npt.assert_equal(rebuild_img[2,2],0)

rebuild_img = build_mask_taking_min(list_patches, 608, 608)

npt.assert_equal(rebuild_img[399,399],1)
npt.assert_equal(rebuild_img[0,607],1)
npt.assert_equal(rebuild_img[607,0],1)
npt.assert_equal(rebuild_img[607,607],1)
npt.assert_equal(rebuild_img[10,10],1)
#npt.assert_equal(rebuild_img[300,300],0)
npt.assert_equal(rebuild_img[2,2],0)

rebuild_img = build_mask_taking_mean(list_patches, 608, 608)

npt.assert_equal(rebuild_img[399,399],1)
npt.assert_equal(rebuild_img[0,607],1)
npt.assert_equal(rebuild_img[607,0],1)
npt.assert_equal(rebuild_img[607,607],1)
npt.assert_equal(rebuild_img[10,10],1)
npt.assert_equal(rebuild_img[300,300],1/4)
npt.assert_equal(rebuild_img[2,2],0)

train_dir = '../../data/training/'

train_files, imgs = load_images(train_dir + "images/")
gt_files, gt_imgs = load_images(train_dir + "groundtruth/")

# Ensure that train and mask files have the same names
npt.assert_equal(train_files, gt_files)

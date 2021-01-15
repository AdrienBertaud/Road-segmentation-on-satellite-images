import matplotlib.image as mpimg
import numpy.testing as npt
import os
import numpy as np
import matplotlib.image as mpimg
import re
from PIL import Image

# Helper functions
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(im, img_number):
    """Reads a single image and outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *images):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for index, image in enumerate(images[0:]):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image, index+1))

# Extract patches from a given image
def img_crop_(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    
    for i in range(0, imgheight, h):
        i0 = i
        
        for j in range(0, imgwidth, w):
            j0 = j
            
            j_plus_w = j+w-1
            i_plus_h = i+h-1
            
            if i_plus_h >= imgheight:
                i_plus_h = imgheight-1
                i0= imgheight - h
                        
            if j_plus_w >= imgwidth:
                j_plus_w = imgwidth-1
                j0 = imgwidth - w
                            
            if is_2d:
                im_patch = im[i0:i_plus_h+1, j0:j_plus_w+1]
            else:
                im_patch = im[i0:i_plus_h+1, j0:j_plus_w+1, :]
                
            list_patches.append(im_patch)

    return list_patches

def build_from_patches(list_patches, h, w):
    patch_width = list_patches[0].shape[0]
    patch_height = list_patches[0].shape[1]
    is_2d = len(list_patches[0].shape) < 3
    rebuild_img = np.zeros((h, w))
    
    npt.assert_equal(is_2d == True, True)
    npt.assert_equal(len(list_patches) == 4, True)
    
    for i in range(0, patch_height):
        for j in range(0, patch_width):
            rebuild_img[i,j] = list_patches[0][i,j]
            
    width_begin = patch_width-(w%patch_width)
    for i in range(0, patch_height):
        for j in range(width_begin, patch_width):
            rebuild_img[i,patch_width+j-width_begin] = list_patches[1][i,j]
    
    height_begin = patch_height-(h%patch_height)
    for i in range(height_begin, patch_height):
        for j in range(0, patch_width):
            rebuild_img[patch_height+i-height_begin,j] = list_patches[2][i,j]
            
    for i in range(patch_height-(h%patch_height), patch_height):
        for j in range(0, patch_width):
            rebuild_img[patch_height+i-height_begin,patch_width+j-width_begin] = list_patches[3][i,j]
            
    return rebuild_img

def predict(image_input, model):
    patches = img_crop_(image_input, 400, 400)
    patches = np.array(patches)
    patch_predictions = model.predict(patches, verbose=1)
    patch_predictions = np.squeeze(patch_predictions)
    img = build_from_patches(patch_predictions, 608, 608)
    return img
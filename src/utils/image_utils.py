import os
import numpy as np
import numpy.testing as npt
import matplotlib.image as mpimg
from PIL import Image
from natsort import natsorted


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def load_images(image_dir):
    files = natsorted(os.listdir(image_dir))
    n_files = len(files)
    print("Loading " + str(n_files) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n_files)]
    return files, imgs

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

# Extract patches from a given image
def img_crop_(im, h, w):
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

def build_mask_taking_min(list_patches, h, w):
    npt.assert_equal(len(list_patches[0].shape), 2)
    patch_width = list_patches[0].shape[0]
    patch_height = list_patches[0].shape[1]
    max_val = np.max(list_patches)
    rebuild_img = np.full((h, w),max_val)

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[0][i,j]
            rebuild_img[i,j] = min(val, rebuild_img[i,j])

    width_begin = patch_width-(w%patch_width)
    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[1][i,j]
            J=patch_width+j-width_begin
            rebuild_img[i,J] = min(val, rebuild_img[i,J])

    height_begin = patch_height-(h%patch_height)
    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[2][i,j]
            I=patch_height+i-height_begin
            rebuild_img[I,j] = min(val, rebuild_img[I,j])

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[3][i,j]
            I=patch_height+i-height_begin
            J=patch_width+j-width_begin
            rebuild_img[patch_height+i-height_begin,patch_width+j-width_begin] = min(val, rebuild_img[I,J])

    return rebuild_img

def build_mask_taking_mean(list_patches, h, w):
    npt.assert_equal(len(list_patches[0].shape), 2)

    patch_width = list_patches[0].shape[0]
    patch_height = list_patches[0].shape[1]

    rebuild_img = np.zeros((h, w))
    occurences = np.zeros((h, w))

    width_begin = patch_width-(w%patch_width)
    height_begin = patch_height-(h%patch_height)

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            I=i
            J=j
            occurences[I,J] +=1

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            I=i
            J=patch_width+j-width_begin
            occurences[I,J] +=1

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            I=patch_height+i-height_begin
            J=j
            occurences[I,J] +=1

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            I=patch_height+i-height_begin
            J=patch_width+j-width_begin
            occurences[I,J] +=1

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[0][i,j]
            I=i
            J=j
            rebuild_img[I,J] += val / occurences[I,J]

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[1][i,j]
            I=i
            J=patch_width+j-width_begin
            rebuild_img[I,J] += val / occurences[I,J]

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[2][i,j]
            I=patch_height+i-height_begin
            J=j
            rebuild_img[I,J] += val / occurences[I,J]

    for i in range(0, patch_height):
        for j in range(0, patch_width):
            val = list_patches[3][i,j]
            I=patch_height+i-height_begin
            J=patch_width+j-width_begin
            rebuild_img[I,J] += val / occurences[I,J]

    return rebuild_img

def predict(image_input, model, h, w):
    patches = img_crop_(image_input, h, w)
    patches = np.array(patches)
    patch_predictions = model.predict(patches)
    patch_predictions = np.squeeze(patch_predictions)
    img = build_mask_taking_mean(patch_predictions, image_input.shape[0], image_input.shape[1])
    return img



import numpy as np
import numpy.testing as npt
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

def normalize_img(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')

def normalize_data(data, mean_zero = False):
    """normalize the data by (x - mean(x)) / std(x)."""
    mean_data = 0

    if mean_zero == False:
        shape_len = len(data.shape)
        mean_data = np.copy(data)

        for i in range(0,shape_len-1):
            mean_data = np.mean(mean_data, axis=0)

        npt.assert_equal(mean_data.shape,(data.shape[3],))

    std_data = np.std(data)

    data = normalize_data_with_given_mean_and_std(data, mean_data, std_data)
    return data, mean_data, std_data

def normalize_data_with_given_mean_and_std(data, mean_data, std_data):
    """normalize the data by (x - mean(x)) / std(x)."""
    data = data - mean_data
    data = data / std_data / 3 * 125 + 125
    mask = data >= 255
    data[mask] = 255
    return data

def data_augmentation(images, seed, crop_length_):
    data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=20,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     #zoom_range=[0.9, 0.9],
                     horizontal_flip=True,
                     vertical_flip=True)

    image_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(images, augment=True, seed=seed)
    image_generator = image_datagen.flow(images, seed=seed)
    image_generator = crop_generator(image_generator, crop_length=crop_length_)

    return image_generator

def crop_center(img, crop_size):
    # Note: image_data_format is 'channel_last'
    #assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = int((width - dx) / 2)
    y = int((height - dy) / 2)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch = next(batches)
        batch_crops = np.zeros((batch.shape[0], crop_length, crop_length, batch.shape[3]))
        for i in range(batch.shape[0]):
            batch_crops[i] = crop_center(batch[i], (crop_length, crop_length))
        yield batch_crops
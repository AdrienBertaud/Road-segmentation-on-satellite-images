import numpy as np
import numpy.testing as npt

from preprocessing import normalize_data, data_augmentation

M = np.random.rand(5,4,4,3)
M_norm, mean_data, std_data = normalize_data(M)

npt.assert_equal(M_norm.shape, M.shape)
npt.assert_equal(mean_data.shape, (3,))
npt.assert_equal(std_data.shape, ())

norm_test_images = np.random.rand(100,400,400,3)
test_masks = np.random.rand(100,400,400,1)

train_generator = data_augmentation(norm_test_images, test_masks, seed=1)


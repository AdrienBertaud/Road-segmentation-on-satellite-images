import matplotlib.pyplot as plt
import numpy as np
from src.utils.image_utils import make_img_overlay

def plot_results(nrows, originals, predictions):
  fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(10,10))

  for index in range(0,nrows):
    ax[index, 0].imshow(originals[index], cmap='Greys_r')
    new_img = make_img_overlay(originals[index], np.squeeze(predictions[index]))
    ax[index, 0].imshow(new_img);
    ax[index, 0].axis('off');
    ax[index, 1].imshow(predictions[index]);
    ax[index, 1].axis('off');
    
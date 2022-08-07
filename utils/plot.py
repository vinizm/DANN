import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_map(wmap: np.ndarray):
    fig = plt.figure(1, figsize = (7, 7))

    ax = plt.gca()
    im = ax.imshow(wmap, cmap = 'jet')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '5%', pad = 0.1)

    plt.colorbar(im, cax = cax)
    ax.axis('off')
    

def compare_images(original_rgb, original_gray, mask_true, mask_pred, mask_proba, i):
    fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (20, 3))

    ax[0].imshow(original_rgb)
    ax[1].imshow(original_gray, cmap = 'gray')
    ax[2].imshow(mask_true, cmap = 'gray_r')
    ax[3].imshow(mask_pred, cmap = 'gray_r')

    proba_vector = mask_proba.reshape(-1)
    #q1 = np.quantile(proba_vector, 0.25)
    #q2 = np.quantile(proba_vector, 0.75)
    q1 = np.min(proba_vector)
    q2 = np.max(proba_vector)
    levels = np.linspace(q1, q2, 100)

    im = ax[4].contourf(np.flip(mask_proba, axis = 0), cmap = 'viridis', levels = levels)
    #im = ax[4].contourf(np.flip(mask_proba, axis = 0), cmap = 'viridis', levels = levels, extend = 'both')

    # add the bar
    cbar = plt.colorbar(im)
    ticks = np.round(10 ** cbar.get_ticks(), 5)
    cbar.ax.set_yticklabels(ticks)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    
    ax[0].set_title(f'Original RGB (img {i})')
    ax[1].set_title('Original Gray')
    ax[2].set_title('True Mask')
    ax[3].set_title('Predicted Mask')
    ax[4].set_title('Probability Map')

    return fig
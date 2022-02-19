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


import numpy as np
import matplotlib.pyplot as plt


def plot_image_grid(x_data, img_shape, grid_shape):

  one_dim = False
  for i in grid_shape:
    if i == 1:
      one_dim = True

  n_pics = x_data.shape[0]
  n_pics_in_grid = np.prod(grid_shape)

  if n_pics_in_grid < n_pics:
    print("adding extra row to grid")
    grid_shape = (grid_shape[0] + 1, grid_shape[1])

  #sizes grid automatically (in inches)
  fig, ax = plt.subplots(grid_shape[0], grid_shape[1])

  if one_dim:
    for i in np.arange(max(grid_shape)):
      data_idx = i

      if data_idx >= n_pics:
        break

      #ax = plt.subplot(grid_shape[0], grid_shape[1], data_idx+1)

      img_data = x_data[data_idx].reshape(*img_shape)

      ax[i].imshow(img_data, interpolation="nearest", vmin=0, vmax=1)
      ax[i].get_xaxis().set_visible(False)
      ax[i].get_yaxis().set_visible(False)

  else:
    for i in np.arange(grid_shape[0]):
      for j in np.arange(grid_shape[1]):

        data_idx = j + grid_shape[1]*i

        if data_idx >= n_pics:
          break

        #ax = plt.subplot(grid_shape[0], grid_shape[1], data_idx+1)

        img_data = x_data[data_idx].reshape(*img_shape)

        ax[i,j].imshow(img_data, interpolation="nearest", vmin=0, vmax=1)
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
  plt.tight_layout()

  return fig





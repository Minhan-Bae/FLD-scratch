import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def visualize_batch(images_list, landmarks_list, gt_list,
                    imgdim = 256, size = 6, shape = (10, 10), save = None):
    fig = plt.figure(figsize = (size, size))
    grid = ImageGrid(fig, 111, nrows_ncols = shape, axes_pad = 0.08)
    for ax, image, landmarks, gt in zip(grid, images_list, landmarks_list, gt_list):
        image = (image - image.min())/(image.max() - image.min())
        
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks+0.5) * imgdim
        landmarks = landmarks.detach().numpy().tolist()
        landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x and 0 <= y])

        gt = gt.view(-1, 2)
        gt = (gt+0.5) * imgdim
        gt = gt.numpy().tolist()
        gt = np.array([(x, y) for (x, y) in gt if 0 <= x  and 0 <= y])

        ax.imshow(image[0], cmap='gray')
        ax.scatter(landmarks[:, 0], landmarks[:, 1], s = 10, c = 'dodgerblue')
        ax.scatter(gt[:, 0], gt[:, 1], s = 10, c = 'red', marker='*')
        ax.axis('off')

    if save:
        plt.savefig(save)
    plt.show()
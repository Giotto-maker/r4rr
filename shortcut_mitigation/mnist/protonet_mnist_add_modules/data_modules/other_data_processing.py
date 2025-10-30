import torch
from numpy import float32, zeros
from cv2 import (
    INTER_CUBIC,   
    warpAffine, 
    moments, 
    WARP_INVERSE_MAP
)


# * Deskew transformation utility functions
def deskew_image(img):
    img = img.cpu().numpy().reshape((28, 28))
    img_moments = moments(img)
    if abs(img_moments['mu02']) > 1e-2:
        img_skew = (img_moments['mu11'] / img_moments['mu02'])
        m = float32([[1, img_skew, -0.6 * img.shape[0] * img_skew], [0, 1, 0]])
        img_deskew = warpAffine(src=img, M=m, dsize=img.shape, flags=INTER_CUBIC | WARP_INVERSE_MAP)
    else:
        img_deskew = img.copy()
    img_deskew[img_deskew < 1e-2] = 0
    img_deskew = img_deskew.reshape((1, 28, 28))
    return img_deskew

def apply_deskew(imgs):
    imgs_deskewed = zeros(imgs.shape)
    assert imgs_deskewed.shape[1:] == (1, 28, 28), \
        "Image shape should be (1, 28, 28), but got {}".format(imgs_deskewed.shape[1:])

    for i in range(imgs_deskewed.shape[0]):
        imgs_deskewed[i, :] = deskew_image(imgs[i, :])
    imgs_deskewed = torch.tensor(imgs_deskewed, dtype=torch.float32)

    assert imgs_deskewed.shape[1:] == (1, 28, 28), \
        "Image shape should be (1, 28, 28), but got {}".format(imgs_deskewed.shape[1:])
    return imgs_deskewed
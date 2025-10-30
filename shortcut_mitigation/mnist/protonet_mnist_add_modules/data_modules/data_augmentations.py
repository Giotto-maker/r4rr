import cv2
import torch
import numpy as np
from numpy import float32, zeros
from cv2 import (INTER_CUBIC, getRotationMatrix2D, warpAffine)


# * Rotation Transformation
def rotation(img, degree):
    img = img.cpu().numpy().reshape((28, 28))
    rows, cols = img.shape
    center = (cols // 2, rows // 2)
    rotation_matrix = getRotationMatrix2D(center, degree, 1.0)
    img_rotated = warpAffine(img, rotation_matrix, (cols, rows))
    img_rotated[img_rotated < 1e-2] = 0
    img_rotated = img_rotated.reshape((1, 28, 28))
    return img_rotated

def apply_rotation(imgs, degree):
    imgs_rotated = zeros(imgs.shape)
    assert imgs_rotated.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28), but got {}".format(imgs_rotated.shape[1:])

    for i in range(imgs_rotated.shape[0]):
        imgs_rotated[i, :] = rotation(imgs[i, :], degree)
    imgs_rotated = torch.tensor(imgs_rotated, dtype=torch.float32)

    assert imgs_rotated.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28), but got {}".format(imgs_rotated.shape[1:])
    return imgs_rotated


# * Translation Transformation
def translation(img, x, y):
    img = img.cpu().numpy().reshape((28, 28))
    M = np.float32([[1, 0, x], [0, 1, y]]) 
    img_translated = cv2.warpAffine(img, M, (28, 28))
    img_translated[img_translated < 1e-2] = 0
    img_translated = img_translated.reshape((1, 28, 28))
    return img_translated

def apply_translation(imgs, x, y):
    imgs_translated = zeros(imgs.shape)
    assert imgs_translated.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28), but got {}".format(imgs_translated.shape[1:])

    for i in range(imgs_translated.shape[0]): 
        imgs_translated[i, :] = translation(imgs[i, :], x, y)
    imgs_translated = torch.tensor(imgs_translated, dtype=torch.float32)

    assert imgs_translated.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28), but got {}".format(imgs_translated.shape[1:])
    return imgs_translated


# * Scaling transformation
def scaling(img, x):
    img = img.cpu().numpy().reshape((28, 28))
    center = (img.shape[1] // 2, img.shape[0] // 2)
    scale_matrix = cv2.getRotationMatrix2D(center, 0, abs(x))
    img_scaled = cv2.warpAffine(img, scale_matrix, (28, 28), flags=cv2.INTER_CUBIC)
    img_scaled[img_scaled < 1e-2] = 0
    img_scaled = img_scaled.reshape((1, 28, 28))    
    return img_scaled

def apply_scaling(imgs, zoom):
    imgs_scaled = zeros(imgs.shape)
    assert imgs_scaled.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28), but got {}".format(imgs_scaled.shape[1:])
    
    for i in range(imgs_scaled.shape[0]):
        imgs_scaled[i, :] = scaling(imgs[i, :], zoom) 
        
    imgs_scaled = torch.tensor(imgs_scaled, dtype=torch.float32)
    assert imgs_scaled.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28), but got {}".format(imgs_scaled.shape[1:])
    return imgs_scaled


# * Deformation transformation
def deforming(img, x):
    img = img.astype(np.float32)
    dx = np.random.uniform(-1, 1, img.shape) * x
    dy = np.random.uniform(-1, 1, img.shape) * x
    dx = cv2.GaussianBlur(dx, (3, 3), 0)
    dy = cv2.GaussianBlur(dy, (3, 3), 0)
    x_coords, y_coords = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)
    img_deformed = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img_deformed

def apply_deforming(imgs, x):
    imgs_deformed = np.zeros(imgs.shape, dtype=np.float32)
    assert imgs_deformed.shape[1:] == (1, 28, 28), f"Image shape should be (1, 28, 28), but got {imgs.shape[1:]}"
    
    for i in range(imgs.shape[0]): 
        imgs_np = imgs[i, 0].cpu().numpy()  # Get the (28, 28) array for the i-th image
        imgs_deformed[i, 0] = deforming(imgs_np, x)

    imgs_deformed = torch.tensor(imgs_deformed, dtype=torch.float32)
    assert imgs_deformed.shape[1:] == (1, 28, 28), f"Image shape should be (1, 28, 28), but got {imgs_deformed.shape[1:]}"
    return imgs_deformed


# * Noising transformation
def noising(img, x):
    if img.shape[0] == 1 and len(img.shape) == 3:
        img = img[0]  # Remove the channel dimension (1, 28, 28) -> (28, 28)
    noise = np.zeros_like(img, dtype=np.float32)  # Create a 2D noise array
    cv2.randn(noise, 0, x)  # Scalar values for mean and stddev
    img_noisy = img.astype(np.float32) + noise  # Add noise to the image
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)  # Clip to valid range
    img_noisy = img_noisy[np.newaxis, :, :]  # Reshape to (1, 28, 28)
    return img_noisy


def apply_noising(imgs, x):
    imgs_noised = zeros(imgs.shape)
    assert imgs_noised.shape[1:] == (1, 28, 28), f"Image shape should be (1, 28, 28), but got {imgs_noised.shape[1:]}"
    
    for i in range(imgs_noised.shape[0]): 
        imgs_np = imgs[i, 0].cpu().numpy()
        imgs_noised[i, :] = noising(imgs_np, x)
    
    imgs_noised = torch.tensor(imgs_noised, dtype=torch.float32)
    assert imgs_noised.shape[1:] == (1, 28, 28), f"Image shape should be (1, 28, 28), but got {imgs_noised.shape[1:]}"
    return imgs_noised


# * Get batch of rotated images
def get_rotated(images, labels=None, start_degree=5.0, stop_degree=15.0, step_degree=0.5):
    degrees_pos = [i for i in np.arange(start_degree, stop_degree, step_degree)]
    degrees_neg = [-deg for deg in degrees_pos]

    rot = []
    rot_labels = []
    
    for degree in degrees_pos + degrees_neg:
        rotated_images = apply_rotation(images, degree)
        rot.append(rotated_images)
        if labels is not None:
            rot_labels.append(labels)  # Replicate labels for each rotation

    rot = torch.stack(rot, dim=0)
    if labels is not None:
        rot_labels = torch.cat(rot_labels, dim=0)  # Concatenate replicated labels

    return (rot, rot_labels) if labels is not None else rot


def get_no_rotations(start_degree=5.0, stop_degree=15.0, step_degree=0.5):
    degrees_pos = [i for i in np.arange(start_degree, stop_degree, step_degree)]
    degrees_neg = [-deg for deg in degrees_pos]
    return 10 * (len(degrees_pos) + len(degrees_neg))


# * get batch of translated images
def get_translated(images, labels, start_point=1.0, stop_point=4.0, step=0.5):
    
    xs_pos = [i for i in np.arange(start_point, stop_point, step)]
    xs_neg = [-x for x in xs_pos]

    ys_pos = [j for j in np.arange(start_point, stop_point, step)]
    ys_neg = [-y for y in ys_pos]

    trl = []
    lbls = []

    for x in xs_pos:
        for y in ys_pos:
            trl.append(apply_translation(images, x, y))
            lbls.append(labels)

    for x in xs_pos:
        for y in ys_neg:
            trl.append(apply_translation(images, x, y))
            lbls.append(labels)

    for x in xs_neg:
        for y in ys_pos:
            trl.append(apply_translation(images, x, y))
            lbls.append(labels)

    for x in xs_neg:
        for y in ys_neg:
            trl.append(apply_translation(images, x, y))
            lbls.append(labels)

    trl = torch.stack(trl, dim=0)
    lbls = torch.cat(lbls, dim=0)  # Concatenate labels along the batch dimension
    return trl, lbls


def get_no_translations(start_point=1.0, stop_point=4.0, step=0.5):
    xs_pos = [i for i in np.arange(start_point, stop_point, step)]
    xs_neg = [-x for x in xs_pos]

    ys_pos = [j for j in np.arange(start_point, stop_point, step)]
    ys_neg = [-y for y in ys_pos]

    return 10 * (len(xs_pos) * len(ys_pos) + len(xs_pos) * len(ys_neg) + len(xs_neg) * len(ys_pos) + len(xs_neg) * len(ys_neg))


# * get batch of scaled images
def get_scaled(images, labels, start_scale=0.5, stop_scale=1.5, step_scale=0.1):
    scales = [scale for scale in np.arange(start_scale, stop_scale, step_scale)]

    sld = []
    lbls = []

    for scale in scales:
        sld.append(apply_scaling(images, scale))
        lbls.append(labels)  # Duplicate labels for each scaling

    sld = torch.stack(sld, dim=0)  # Stack scaled images
    lbls = torch.cat(lbls, dim=0)  # Concatenate labels
    return sld, lbls


def get_no_scaled(start_scale=0.5, stop_scale=1.5, step_scale=0.1):
    scales = [scale for scale in np.arange(start_scale, stop_scale, step_scale)]
    return 10 * len(scales)


# * get batch of elastic deformed images
def get_elastic_transformations(images, labels, start=0.5, stop=3.0, step=0.1):
    deforms = [deform for deform in np.arange(start, stop, step)]
    eld = []
    lbls = []
    for deform in deforms:
        eld.append(apply_deforming(images, deform))
        lbls.append(labels)  # Duplicate labels for each distortion

    eld = torch.stack(eld, dim=0)  # Stack scaled images
    lbls = torch.cat(lbls, dim=0)  # Concatenate labels
    return eld, lbls


def get_no_elastic_transformations(start=0.5, stop=3.0, step=0.1):
    deforms = [deform for deform in np.arange(start, stop, step)]
    return 10 * len(deforms)


# * get batch of noised images
def get_noisy_transformations(images, labels, start=0.05, stop=0.2, step=0.05):
    noises = [noise for noise in np.arange(start, stop, step)]
    nld = []
    lbls = []
    for noise in noises:
        nld.append(apply_noising(images, noise))
        lbls.append(labels)  # Duplicate labels for each noising

    nld = torch.stack(nld, dim=0)  # Stack scaled images
    lbls = torch.cat(lbls, dim=0)  # Concatenate labels
    return nld, lbls


def get_no_noisy_transformations(start=0.05, stop=0.2, step=0.05):
    noises = [noise for noise in np.arange(start, stop, step)]
    return 10 * len(noises)
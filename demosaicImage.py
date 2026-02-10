# This code is part of:
#
#   CS4501-00:  Computer Vision, Spring 2026
#   University of Virginia
#   Instructor: Zezhou Cheng

import numpy as np


def demosaicImage(image, method):
    """Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    """

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == "nn":
        return demosaicNN(image.copy())  # Implement this
    elif method.lower() == "linear":
        return demosaicLinear(image.copy())  # Implement this
    elif method.lower() == "adagrad":
        return demosaicAdagrad(image.copy())  # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    """Baseline demosaicing.

    Replaces missing values with the mean of each color channel.

    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline
        algorithm.
    """
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[1:image_height:2, 1:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 0] = img[
        1:image_height:2, 1:image_width:2
    ]

    blue_values = img[0:image_height:2, 0:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 2] = img[
        0:image_height:2, 0:image_width:2
    ]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img

# =============================================================================
# Helper functions (provided for you)
# =============================================================================

def rgb_images(img):
    """Splits a mosaiced image into separate R, G, B channel images.
    
    Bayer pattern:
        B G B G ...
        G R G R ...
        B G B G ...
        G R G R ...
    
    Args:
        img: np.array of size NxM (mosaiced image).
    
    Returns:
        red_img, green_img, blue_img: each NxM, with 0 at missing locations.
    """
    image_height, image_width = img.shape
    red_img = np.zeros(img.shape)
    red_img[1::2, 1::2] = img[1::2, 1::2]
    blue_img = np.zeros(img.shape)
    blue_img[0::2, 0::2] = img[0::2, 0::2]
    green_img = np.zeros(img.shape)
    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_img[mask > 0] = img[mask > 0]
    return red_img, green_img, blue_img


def neighbors(h, w, height, width):
    """Returns valid 8-connected neighbor coordinates.
    
    Args:
        h, w: current pixel location.
        height, width: image dimensions.
    
    Returns:
        List of (h, w) tuples for valid neighbors.
    """
    n = [
        (h + 1, w),
        (h - 1, w),
        (h, w + 1),
        (h, w - 1),
        (h + 1, w + 1),
        (h - 1, w + 1),
        (h + 1, w - 1),
        (h - 1, w - 1),
    ]
    return [(h, w) for (h, w) in n if h < height and w < width and h >= 0 and w >= 0]


def in_bounds(p, height, width):
    """Checks if all points in p are within image bounds."""
    return all([h < height and w < width and h >= 0 and w >= 0 for (h, w) in p])


def opposite_neighbors(h, w, height, width):
    """Returns pairs of opposite neighbors for gradient computation.
    
    Args:
        h, w: current pixel location.
        height, width: image dimensions.
    
    Returns:
        List of [(h1, w1), (h2, w2)] pairs that are opposite to each other.
        Only returns pairs where both neighbors are within bounds.
        
    Example: for pixel at (2, 2), returns pairs like:
        [(3, 2), (1, 2)]  - vertical pair
        [(2, 3), (2, 1)]  - horizontal pair
        [(3, 3), (1, 1)]  - diagonal pair
        [(3, 1), (1, 3)]  - anti-diagonal pair
    """
    on = [
        [(h + 1, w), (h - 1, w)],      # vertical
        [(h, w + 1), (h, w - 1)],      # horizontal
        [(h + 1, w + 1), (h - 1, w - 1)],  # diagonal
        [(h + 1, w - 1), (h - 1, w + 1)],  # anti-diagonal
    ]
    return [p for p in on if in_bounds(p, height, width)]


# =============================================================================
# TODO: Complete and implement the following functions
# =============================================================================

def nn(out_channel, channel_img, h, w):
    """Nearest neighbor interpolation for a single pixel.
    
    Args:
        out_channel: output channel image to fill (NxM).
        channel_img: input channel image with 0 at missing locations (NxM).
        h, w: current pixel location.
    """
    height, width = channel_img.shape
    if channel_img[h, w] == 0:
        for x, y in neighbors(h, w, height, width):
            # TODO: if this neighbor has a non-zero value, 
            #       copy it to out_channel[h, w] and return
            if (channel_img[x, y] != 0):
                out_channel[h, w] = channel_img[x, y]
                return
    else:
        out_channel[h, w] = channel_img[h, w]


def demosaicNN(img):
    """Nearest neighbor demosaicing."""
    image_height, image_width = img.shape
    red_img, green_img, blue_img = rgb_images(img)
    out = np.zeros((image_height, image_width, 3))
    for h in range(image_height):
        for w in range(image_width):
            nn(out[:, :, 0], red_img, h, w)
            nn(out[:, :, 1], green_img, h, w)
            nn(out[:, :, 2], blue_img, h, w)
    return out

def linear(out_channel, channel_img, h, w):
    height, width = channel_img.shape
    if channel_img[h, w] == 0:
        average = 0
        neighbor = 0
        for x, y in neighbors(h, w, height, width):
            if (channel_img[x, y] != 0):
                average += channel_img[x, y]
                neighbor += 1
        if (neighbor > 0):
            out_channel[h, w] = average/neighbor
    else:
        out_channel[h, w] = channel_img[h, w]


def demosaicLinear(img):
    """Linear interpolation demosaicing.
    
    Similar to demosaicNN, but instead of taking the first non-zero neighbor,
    compute the average of ALL non-zero neighbors for each missing pixel.
    
    Hint: you may want to write a helper function similar to nn().
    """
    # TODO: Implement this
    image_height, image_width = img.shape
    red_img, green_img, blue_img = rgb_images(img)
    out = np.zeros((image_height, image_width, 3))
    for h in range(image_height):
        for w in range(image_width):
            linear(out[:, :, 0], red_img, h, w)
            linear(out[:, :, 1], green_img, h, w)
            linear(out[:, :, 2], blue_img, h, w)
    return out

def adagrad(out_channel, channel_img, h, w):
    height, width = channel_img.shape
    if channel_img[h, w] == 0:
        smallest = float("inf")
        best_pair = None
        for (y1, x1), (y2, x2) in opposite_neighbors(h, w, height, width):
            if (channel_img[y1, x1] != 0 and channel_img[y2, x2] != 0):
                if (abs(channel_img[y1, x1] - channel_img[y2, x2]) < smallest):
                    smallest = abs(channel_img[y1, x1] - channel_img[y2, x2])
                    best_pair = ((y1, x1), (y2, x2))
        if (best_pair != None):
            y1, x1 = best_pair[0]
            y2, x2 = best_pair[1]
            average = (channel_img[y1, x1]+channel_img[y2, x2])/2
            out_channel[h, w] = average
    else:
        out_channel[h, w] = channel_img[h, w]


def demosaicAdagrad(img):
    """Adaptive gradient demosaicing.
    
    For green channel: instead of averaging all neighbors, find the opposite 
    neighbor pair with the smallest gradient (smallest |a - b|), then average 
    that pair. Use opposite_neighbors() to get pairs.
    
    For red and blue channels: use linear interpolation.
    
    Hint: you may want to write a helper function for the green channel.
    """
    # TODO: Implement this
    image_height, image_width = img.shape
    red_img, green_img, blue_img = rgb_images(img)
    out = np.zeros((image_height, image_width, 3))
    for h in range(image_height):
        for w in range(image_width):
            linear(out[:, :, 0], red_img, h, w)
            adagrad(out[:, :, 1], green_img, h, w)
            linear(out[:, :, 2], blue_img, h, w)
    return out
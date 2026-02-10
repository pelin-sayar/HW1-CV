# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#
import numpy as np


def alignChannels(img, max_shift):
    """Align the color channels of an image.
    
    The first channel (R) is fixed as reference. We compute the alignment 
    of the second (G) and third (B) channels with respect to the first.
    
    Args:
        img: np.array of size HxWx3 (three color channels).
        max_shift: np.array [max_shift_row, max_shift_col], the maximum 
                   shift to search in each direction.
    
    Returns:
        aligned_img: HxWx3 color image with aligned channels.
        pred_shift: 2x2 array where pred_shift[0] is the shift for channel 1 (G)
                    and pred_shift[1] is the shift for channel 2 (B).
                    Each shift is [shift_row, shift_col].
    
    Hints:
        - Use alignChannel() to find the best shift for each channel
        - Use shiftImage() to apply the shift
    """
    pred_shift = np.zeros((2, 2))
    
    aligned_image = img.copy()
    for i in range(1, 3):
        pred_shift[i-1] = alignChannel(img[:,:,0], img[:,:,i], max_shift)
        aligned_image[:,:,i] = shiftImage(img[:,:,i], pred_shift[i-1])
    
    return aligned_image, pred_shift


def alignChannel(ref_img, target_img, max_shift):
    """Find the best shift to align target_img to ref_img.
    
    Search over all possible shifts within [-max_shift, max_shift] range,
    compute a matching metric for each shift, and return the best shift.
    
    Args:
        ref_img: np.array of size HxW, the reference image (fixed).
        target_img: np.array of size HxW, the image to be aligned.
        max_shift: np.array [max_shift_row, max_shift_col].
    
    Returns:
        best_shift: np.array [shift_row, shift_col] that best aligns 
                    target_img to ref_img.
    
    Hints:
        - Use nested for loops to search over all shifts in the range
          [-max_shift[0], max_shift[0]] x [-max_shift[1], max_shift[1]]
        - For each shift, use shiftImage() to shift target_img
        - Compute a matching metric (SSD or Cosine Similarity) between 
          ref_img and the shifted target_img
        - Return the shift with the best score
    """
    best_shift = np.zeros(2)
    
    #searching through image
    best_ssd = float("inf")
    for i in range(-max_shift[0], max_shift[0]+1):
        for j in range(-max_shift[1], max_shift[1]+1):
            shifted_img = shiftImage(target_img, [i,j])
            ssd_value = np.sum((ref_img-shifted_img)**2)
            if (ssd_value < best_ssd):
                best_ssd = ssd_value
                best_shift = [int (i), int (j)]

    
    return best_shift


def shiftImage(img, shift):
    """Shift an image by the given amount.
    
    Args:
        img: np.array of size HxW.
        shift: np.array [shift_row, shift_col], the amount to shift.
               Positive values shift down/right, negative values shift up/left.
    
    Returns:
        shifted_img: HxW image shifted by the specified amount.
    
    Hints:
        - You can use np.roll() for circular shift, which works well for 
          toy examples
        - For real images, consider using np.pad() with 'edge' mode to 
          handle boundaries
    """
    shifted_img = img.copy()
    
    h, w = img.shape
    x, y = int (shift[0]), int (shift[1])
    max_shift = max(abs(x), abs(y))
    padded = np.pad(shifted_img, max_shift, mode = "edge")
    x_start = max_shift + x
    y_start = max_shift + y
    
    
    return padded[x_start:x_start+h, y_start:y_start+w]
# Run-length encoding
import numpy as np


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    flatten_shape = 1
    for dim in range(len(shape)):
        flatten_shape *= shape[dim]
    img = np.zeros(flatten_shape, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def semantic_rle_decode(total_mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    flatten_shape = 1
    for dim in range(len(shape)):
        flatten_shape *= shape[dim]
    img = np.zeros(flatten_shape, dtype=np.uint8)
    
    for label, mask_rle in total_mask_rle.items():
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def semantic_mask2rle(semantic_mask, save_path):
    mask_shape = ' '.join(str(dim) for dim in semantic_mask.shape)
    max_label = np.max(semantic_mask)
    one_hot = (np.arange(semantic_mask.max()) == semantic_mask[...,None]-1).astype(int)
    with open(save_path, 'w+') as f:
        f.write(mask_shape)
        f.write('\n')

    for dim in range(one_hot.shape[-1]):
        binary_mask = one_hot[...,dim]
        rle = mask2rle(binary_mask)
        rle = str(dim+1) + ' ' + rle
        with open(save_path, 'a+') as f:
            f.write(rle)
            f.write('\n')


def semantic_rle2mask(rle_path):
    with open(rle_path, 'r') as f:
        shape = f.readline()
        shape = tuple([int(x) for x in shape.split(' ')])

        semantic_mask = 0
        total_rle_code = {}
        for line in f:
            content_list = line.split(' ')
            semantic_label = int(content_list[0])
            rle_code = ' '.join(content_list[1:])
            total_rle_code[semantic_label] = rle_code
        semantic_mask = semantic_rle_decode(total_rle_code, shape)
        # semantic_mask += mask*semantic_label
    return semantic_mask

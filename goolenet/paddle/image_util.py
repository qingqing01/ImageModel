from PIL import Image
import numpy as np

def resize_image(img, target_size):
    """
    Resize an image so that the shorter edge has length target_size.
    img: the input image to be resized.
    target_size: the target resized image size.
    """
    percent = (target_size/float(min(img.size[0], img.size[1])))
    if percent < 1.0:
        resized_size = int(round(img.size[0] * percent)), int(round(img.size[1] * percent))
        img = img.resize(resized_size, Image.ANTIALIAS)
    return img

def resize_image2(img, target_size):
    """
    Resize an image so that the shorter edge has length target_size.
    img: the input image to be resized.
    target_size: the target resized image size.
    """
    row = img.size[0]
    col = img.size[1]
    scale = target_size/float(min(row, col))
    if row < col:
        row = target_size
        col = int(round(col * scale))
        col = col if col > target_size else target_size
    else:    
        col = target_size
        row = int(round(row * scale))
        row = row if row > target_size else target_size
    resized_size = row, col
    #resized_size = 256, 256
    img = img.resize(resized_size, Image.ANTIALIAS)
    return img

def flip(im):
    """
    Return the flipped image.
    Flip an image along the horizontal direction.
    im: input image.
    """
    if len(im.shape) == 3:
        return im[:, :, ::-1]
    else:
        return im[:, ::-1]

def crop_img(im, inner_size, color=True, test=True):
    """
    Return cropped image.
    The size of the cropped image is inner_size * inner_size.
    inner_size: the cropped image size.
    color: whether it is color image.
    test: whether in test mode.
      If False, does random cropping and flipping.
      If True, crop the center of images.
    """
    if color:
        height, width = max(inner_size, im.shape[1]), max(inner_size, im.shape[2])
        padded_im = np.zeros((3, height, width))
        startY = (height - im.shape[1]) / 2
        startX = (width - im.shape[2]) / 2
        endY, endX = startY + im.shape[1], startX + im.shape[2]
        padded_im[:, startY: endY, startX: endX] = im
    else:
        im = im.astype('float32')
        height, width = max(inner_size, im.shape[0]), max(inner_size, im.shape[1])
        padded_im = np.zeros((height, width))
        startY = (height - im.shape[0]) / 2
        startX = (width - im.shape[1]) / 2
        endY, endX = startY + im.shape[0], startX + im.shape[1]
        padded_im[startY: endY, startX: endX] = im
    if test:
        startY = (height - inner_size) / 2
        startX = (width - inner_size) / 2
    else:
        startY = np.random.randint(0, height - inner_size + 1)
        startX = np.random.randint(0, width - inner_size + 1)
    endY, endX = startY + inner_size, startX + inner_size
    if color:
        pic = padded_im[:, startY: endY, startX: endX]
    else:
        pic = padded_im[startY: endY, startX: endX]
    if (not test) and (np.random.randint(2) == 0):
        pic = flip(pic)
    return pic


def preprocessImg(obj, im):
    """
    Does data augmentation for images.
    If obj.test is true, cropping the center region from the image.
    If obj.test is false, randomly crop a region from the image,
    and randomy does flipping.
    """
    im = im.astype('float32')
    test = not obj.is_train
    pic = crop_img(im, obj.img_size, obj.color, test)
    pic -= obj.img_mean
    return pic.flatten()


def loadMeta(obj):
    """
    Return the loaded meta file.
    Load the meta image, which is the mean of the images in the dataset.
    The mean image is subtracted from every input image so that the expected mean
    of each input image is zero.
    """
    mean = np.load(obj.meta_path)['data_mean']
    border = (obj.mean_img_size - obj.img_size) / 2
    if obj.color:
        assert(obj.mean_img_size * obj.mean_img_size * 3 == mean.shape[0])
        mean = mean.reshape(3, obj.mean_img_size, obj.mean_img_size)
        mean = mean[:, border: border + obj.img_size,
                       border: border + obj.img_size].astype('float32')
    else:
        assert(obj.mean_img_size * obj.mean_img_size == mean.shape[0])
        mean = mean.reshape(obj.mean_img_size, obj.mean_img_size)
        mean = mean[border: border + obj.img_size,
                    border: border + obj.img_size].astype('float32')
    return mean

import image_util
from paddle.utils.image_util import *
from paddle.trainer.PyDataProvider2 import *
import random

from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import StringIO


logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('paddle_data_provider')
logger.setLevel(logging.INFO)

def hook(settings, image_size, crop_size, color, is_train, **kwargs):
    settings.resize_size = image_size
    settings.img_size = crop_size
    settings.crop_size = crop_size
    settings.color = color  # default is color
    settings.is_train = is_train

    if settings.color:
        settings.img_input_size = settings.crop_size * settings.crop_size * 3
    else:
        settings.img_input_size = settings.crop_size * settings.crop_size

    settings.mean_value = kwargs.get('mean_value')
    sz = settings.crop_size * settings.crop_size
    settings.img_mean = np.zeros(sz * 3, dtype=np.single)
    for idx, value in enumerate(settings.mean_value):
        settings.img_mean[idx * sz: (idx + 1) * sz] = value
    settings.img_mean = settings.img_mean.reshape(3, settings.crop_size,
                                                  settings.crop_size)
    settings.img_mean = settings.img_mean.astype('float32')

    settings.input_types = [
        integer_value(1), # labels
        dense_vector(settings.img_input_size)]  # image feature

    settings.logger.info('Image resize size: %s', settings.resize_size)
    settings.logger.info('Crop size: %s', settings.crop_size)
    settings.logger.info('DataProvider Initialization finished')


@provider(init_hook=hook,min_pool_size=0)
def processPIL(settings, file_list):
    with open(file_list, 'r') as fdata:
        lines = [line.strip() for line in fdata]
        random.shuffle(lines)
        for line in lines:
            img_path, lab = line.strip().split('\t')
            img = Image.open(img_path)
            img.load()
            img = image_util.resize_image2(img, settings.resize_size)
            img = np.array(img)
            if len(img.shape) == 3:
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 1, 0)
            img = img[[2,1,0],:,:]
            img_feat= image_util.preprocessImg(settings, img)
            yield int(lab.strip()), img_feat.astype('float32')

@provider(init_hook=hook,min_pool_size=0)
def processScipy(settings, file_list):
    with open(file_list, 'r') as fdata:
        for line in fdata:
            img_path, lab = line.strip().split(' ')
            img = imread(img_path);
            img = imresize(img, [256, 256])
            imsave('temp_resized.jpg', img)
            dat = open('temp_resized.jpg', 'rb').read()
            img = Image.open(StringIO.StringIO(dat))
            img = np.array(img)
            if len(img.shape) == 3:
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 1, 0)
            img = img[[2,1,0],:,:]
            img_feat= image_util.preprocessImg(settings, img)
            yield int(lab.strip()), img_feat.astype('float32')

# Defining the augmentation schemes going to be used
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from collections import namedtuple
import torch
from torchvision import transforms

# Defining custom augmentation class, so that we can inherit this class for trying different augmentation schemes 
class Augmentation:
    def __init__(self, level=0.5, scale = 0.75, method = 0, shear = 0.2, th = 0.5, delta = 0.2, angle = None):    
        self.level = level
        self.scale = scale
        self.method = method
        self.shear = shear
        self.th = th
        self.delta = delta
        self.range1 = [0,1]
        self.range_cutout = [0,0.5]
        self.range_angle = [-45,45]
        self.range_posterize = [1,8]
        self.range_rescale = [0.5,1]
        self.range_shear_translate = [-0.3,0.3]

    def _enhance(self,op, x):
        return op(x).enhance(0.1 + 1.9 * self.level)


    def _imageop(self,x,op):
        return Image.blend(x, op(x), self.level)


    def _filter(self, x, op):
        return Image.blend(x, x.filter(op), self.level)


    def autocontrast(self, x):
        return _imageop(x, ImageOps.autocontrast)


    def blur(self, x):
        return _filter(x, ImageFilter.BLUR)


    def brightness(self, x):
        return _enhance(x, ImageEnhance.Brightness)


    def color(self, x):
        return _enhance(x, ImageEnhance.Color)


    def contrast(self, x):
        return _enhance(x, ImageEnhance.Contrast, self)

    def equalize(self, x):
        return _imageop(x, ImageOps.equalize)


    def invert(self, x):
        return _imageop(x, ImageOps.invert)


    def identity(x):
        return x


    def posterize(self, x):
        level = 1 + int(self.level * 7.999)
        return ImageOps.posterize(x, self.level)


    def rescale(x):
        s = x.size
        scale *= 0.25
        crop = (self.scale, self.scale, s[0] - self.scale, s[1] - self.scale)
        methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
        method = methods[int(self.method * 5.99)]
        return x.crop(crop).resize(x.size, method)


    def rotate(x, angle):
        angle = int(np.round((2 * angle - 1) * 45))
        return x.rotate(angle)


    def sharpness(x, sharpness):
        return _enhance(x, ImageEnhance.Sharpness, sharpness)


    def shear_x(x, shear):
        shear = (2 * shear - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


    def shear_y(x, shear):
        shear = (2 * shear - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


    def smooth(x, level):
        return _filter(x, ImageFilter.SMOOTH, level)


    def solarize(x, th):
        th = int(th * 255.999)
        return ImageOps.solarize(x, th)


    def translate_x(x, delta):
        delta = (2 * delta - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


    def translate_y(x, delta):
        delta = (2 * delta - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))

    
    global_augs_dict_strong = {'translate_x': translate_x, 'translate_y': translate_y, 'solarize': solarize, 'smooth': smooth, 'shear_x': shear_x, 'shear_y': shear_y,
                        'sharpness': sharpness, 'rotate': rotate, 'autocontrast': autocontrast, 'blur': blur, 'brightness': brightness, 'color': color,
                        'contrast': contrast, 'equalize': equalize, 'invert': invert, 'identity': identity, 'posterize': posterize}
    global_augs_dict_weak = {}

    augs = list(global_augs_dict_strong.keys())


augmt = Augmentation()


def process_batch(batch,label = True):
    if label:
        for image in batch:
            aug = random.choice(augs)
            # Need to convert tensor to PIL image and back
            image = transforms.ToPILImage()(image.cpu()).convert("RGB")
            image = global_augs_dict_strong[aug](image)
            image = transforms.ToTensor()(image).cuda()
            return image
    else:
        print("no label")
    return batch

# Defining the augmentation object from the class

print(augmt)

#Rounak Saha
#20CS30043

#Imports
from PIL import Image, ImageFilter
import numpy as np


class BlurImage(object):
    '''
        Applies Gaussian Blur on the image.
    '''

    def __init__(self, radius):
        '''
            Arguments:
            radius (int): radius to blur
        '''
        self.radius = radius

        # Write your code here
        

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL Image)

            Returns:
            image (numpy array or PIL Image)
        '''
        #image is a (3,H,W) numpy array of float 0-1 values
        pil_image = Image.fromarray(np.uint8(image.transpose(1,2,0)*255)) #constructing a PIL image out of numpy array image
        #input to Image.fromarray should be a (H,W,3) numpy array of 0-255 int values
        img_blur = pil_image.filter(ImageFilter.GaussianBlur(self.radius))
        img_blur = np.array(img_blur)
        img_blur = img_blur.transpose(2,0,1)/255.0 #returning back a (3,H,W) numpy array of 0-1 float values

        return img_blur
        # Write your code here
        


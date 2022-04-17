#Rounak Saha
#20CS30043

#Imports
from PIL import Image
import numpy as np


class RotateImage(object):
    '''
        Rotates the image about the centre of the image.
    '''

    def __init__(self, degrees):
        '''
            Arguments:
            degrees: rotation degree.
        '''
        
        # Write your code here
        self.angle = degrees

    def __call__(self, sample):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here

        #image is a (3,H,W) numpy array of float 0-1 values
        pil_image = Image.fromarray(np.uint8(sample.transpose(1,2,0)*255)) #constructing a PIL image out of numpy array image
        #input to Image.fromarray should be a (H,W,3) numpy array of 0-255 int values

        rot_img = pil_image.rotate(self.angle)

        rot_img = np.array(rot_img)
        rot_img = rot_img.transpose(2,0,1)/255.0 #returning back a (3,H,W) numpy array of 0-1 float values

        return rot_img
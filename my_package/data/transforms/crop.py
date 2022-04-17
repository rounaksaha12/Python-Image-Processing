#Rounak Saha
#20CS30043

#Imports
from PIL import Image, ImageFilter
import numpy as np
import random

class CropImage(object):
    '''
        Performs either random cropping or center cropping.
    '''

    def __init__(self, shape, crop_type='center'):
        '''
            Arguments:
            shape: output shape of the crop (h, w)
            crop_type: center crop or random crop. Default: center
        '''
        self.height = shape[0]
        self.width = shape[1]
        self.crop_type = crop_type

        # Write your code here

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''
        #image is a (3,H,W) numpy array of float 0-1 values
        pil_image = Image.fromarray(np.uint8(image.transpose(1,2,0)*255)) #constructing a PIL image out of numpy array image
        #input to Image.fromarray should be a (H,W,3) numpy array of 0-255 int values
        W,H = pil_image.size # W = total width of the image, H = total height of the image

        if(self.crop_type != 'random'): #'center' as well as any stray input except 'random' cause the default center crop
            left = int((W-self.width)/2)
            right = int((W+self.width)/2)
            top = int((H-self.height)/2)
            bottom = int((H+self.height)/2)
            cropped_img = pil_image.crop((left,top,right,bottom))

        else:
            #first determine a random coordinate inside the image that can be the centre of the uncropped residue
            c_x = random.randrange(int(self.width/2),int(W-self.width/2))
            c_y = random.randrange(int(self.height/2),int(H-self.height/2))
            left = int(c_x-self.width/2)
            right = int(c_x+self.width/2)
            top = int(c_y-self.height/2)
            bottom = int(c_y+self.height/2)
            cropped_img = pil_image.crop((left,top,right,bottom))

        cropped_img = np.array(cropped_img)
        cropped_img = cropped_img.transpose(2,0,1)/255.0 #returning back a (3,H,W) numpy array of 0-1 float values

        return cropped_img

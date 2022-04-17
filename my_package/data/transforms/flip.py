#Rounak Saha
#20CS30043

#Imports
from PIL import Image
import numpy as np

class FlipImage(object):
    '''
        Flips the image.
    '''

    def __init__(self, flip_type='horizontal'):
        '''
            Arguments:
            flip_type: 'horizontal' or 'vertical' Default: 'horizontal'
        '''

        # Write your code here
        self.flip_type = flip_type

        
    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here

        #image is a (3,H,W) numpy array of float 0-1 values
        pil_image = Image.fromarray(np.uint8(image.transpose(1,2,0)*255)) #constructing a PIL image out of numpy array image
        #input to Image.fromarray should be a (H,W,3) numpy array of 0-255 int values

        if(self.flip_type != 'vertical'): #'horizontal' as well as any stray imput except 'vertical' causes a default horizontal flip
            flip_img = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        else:
            flip_img = pil_image.transpose(Image.FLIP_TOP_BOTTOM)

        flip_img = np.array(flip_img)
        flip_img = flip_img.transpose(2,0,1)/255.0 #returning back a (3,H,W) numpy array of 0-1 float values

        return flip_img

       
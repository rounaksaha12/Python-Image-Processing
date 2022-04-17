#Rounak Saha
#20CS30043

#Imports
from PIL import Image
import numpy as np


class RescaleImage(object):
    '''
        Rescales the image to a given size.
    '''

    def __init__(self, output_size, is_rel = 0):
        '''
            Arguments:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.

            added 1 extra argument: is_rel
            is_rel == 0 => image needs to be rescaled output_size times
        '''

        # Write your code here
        if(type(output_size) is tuple):
            self.input_type = 1 #input is complete in the sense that we know the height and width of the output image
            self.height = output_size[0]
            self.width = output_size[1]
        
        else:
            self.input_type = 0 #input is incomplete in the sense that we need to know the info of the image for deciding height and width of the rescaled image
            self.output_size = output_size

        self.is_rel = is_rel

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)

            Note: You do not need to resize the bounding boxes. ONLY RESIZE THE IMAGE.
        '''

        # Write your code here
        #image is a (3,H,W) numpy array of float 0-1 values
        pil_image = Image.fromarray(np.uint8(image.transpose(1,2,0)*255)) #constructing a PIL image out of numpy array image
        #input to Image.fromarray should be a (H,W,3) numpy array of 0-255 int values
        W,H = pil_image.size # W = total width of the image, H = total height of the image

        if(self.input_type == 0):
            if self.is_rel == 0:
                if(W<H):
                    self.width = self.output_size
                    self.height = int((H/W)*self.width)
                else:
                    self.height = self.output_size
                    self.width = int((W/H)*self.height)

            else:
                self.width = int(W*self.output_size)
                self.height = int(H*self.output_size)

        res_img = pil_image.resize((self.width,self.height))
        res_img = np.array(res_img)
        res_img = res_img.transpose(2,0,1)/255.0 #returning back a (3,H,W) numpy array of 0-1 float values

        return res_img

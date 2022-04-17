#Rounak Saha
#20CS30043

#Imports
from .transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
from PIL import Image
import numpy as np
import json
import os

class Dataset(object):
    '''
        A class for the dataset that will return data items as per the given index
    '''

    def __init__(self, annotation_file, transforms = None):
        '''
            Arguments:
            annotation_file: path to the annotation file
            transforms: list of transforms (class instances)
                        For instance, [<class 'RandomCrop'>, <class 'Rotate'>]
        '''
        self.annotation_file = annotation_file
        self.transforms = transforms
        

    def __len__(self):
        '''
            return the number of data points in the dataset
        '''
        count = 0;
        with open(self.annotation_file,'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            count = count+1
        
        return count

    def __getitem__(self, idx):
        '''
            return the dataset element for the index: "idx"
            Arguments:
                idx: index of the data element.

            Returns: A dictionary with:
                image: image (in the form of a numpy array) (shape: (3, H, W))
                gt_png_ann: the segmentation annotation image (in the form of a numpy array) (shape: (1, H, W))
                gt_bboxes: N X 5 array where N is the number of bounding boxes, each 
                            consisting of [class, x1, y1, x2, y2]
                            x1 and x2 lie between 0 and width of the image,
                            y1 and y2 lie between 0 and height of the image.

            You need to do the following, 
            1. Extract the correct annotation using the idx provided.
            2. Read the image, png segmentation and convert it into a numpy array (wont be necessary
                with some libraries). The shape of the arrays would be (3, H, W) and (1, H, W), respectively.
            3. Scale the values in the arrays to be with [0, 1].
            4. Perform the desired transformations on the image.
            5. Return the dictionary of the transformed image and annotations as specified.
        '''

        cwd = os.path.dirname(os.path.realpath(__file__))
        #print(cwd)
        parent = os.path.dirname(os.path.dirname(cwd)) # parent is the string representing path of the main directory

        with open(self.annotation_file,'r') as json_file:
            json_list = list(json_file)
        curr_idx = 0
        for json_str in json_list:
            if curr_idx == idx:
                req_str = json_str
                break
            curr_idx = curr_idx+1

        info = json.loads(req_str) #info is the dictionary extracted from the json file

        ret_dict = {} #dictionary to be returned

        img = Image.open(parent+"\data\\"+info["img_fn"])
        img_np_arr = (np.array(img).transpose(2,0,1))/255.0
        for t in self.transforms:
            img_np_arr = t(img_np_arr)
        ret_dict["image"] = img_np_arr #dictionary to be returned loaded with numpy array of image in (3,H,W) scaled to 0-1

        png_anno = Image.open(parent+"\data\\"+info["png_ann_fn"])
        arr = np.array(png_anno)/255.0
        anno_np_arr = np.zeros((1,arr.shape[0],arr.shape[1]))
        for h in range(arr.shape[0]):
            for w in range(arr.shape[1]):
                anno_np_arr[0][h][w] = arr[h][w]
        ret_dict["gt_png_ann"] = anno_np_arr #dictionary to be returned loaded with numpy array of png annotation in (1,H,W) scaled to 0-1

        ret_dict["gt_bboxes"] = []
        #info["bboxes"] is a list of dictionaries, each dictionary represents a detected object in the image
        #we iterate overl all elements(dictionaries/objects) in the list and extract info one at a time
        for list_item in info["bboxes"]:
            #list_item is a dictionary
            new_list = [] #to be filled up in the format [class, x1, y1, x2, y2] and appended to ret_dict["gt_bboxes"]
            new_list.append(list_item["category"])
            coordinates = list_item["bbox"]
            new_list.extend([coordinates[0],coordinates[1],coordinates[0]+coordinates[2],coordinates[1]+coordinates[3]])
            #new_list is fully loaded so append it to ret_dict["gt_bboxes"]
            ret_dict["gt_bboxes"].append(new_list)

        return ret_dict




        
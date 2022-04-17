#Rounak Saha
#20CS30043


#Imports
from my_package.model import InstanceSegmentationModel
from my_package.data import Dataset
from my_package.analysis import plot_visualization
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def experiment(annotation_file, segmentor, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        segmentor: The image segmentor
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''

    #Create the instance of the dataset.
    data = Dataset(annotation_file,[])

    #Iterate over all data items.
    for k in range(len(data)):
        dict = data[k]
        img_np_arr = dict["image"]
        pred_boxes, pred_masks, pred_class, pred_score = segmentor(img_np_arr) #Get the predictions from the segmentor.
        file_path = outputs+r"\\output_without_transform\\raw_masked_n_boxed_"+str(k)+".jpg"
        #Draw the segmentation maps on the image and save them.
        this_img = plot_visualization(img_np_arr, pred_boxes, pred_masks, pred_class, pred_score, file_path, 0)
        plt.subplot(1,2,1)
        plt.imshow(dict["gt_png_ann"].transpose(1,2,0))
        plt.gca().title.set_text("PNG Annotation")
        plt.subplot(1,2,2)
        plt.imshow(this_img)
        plt.gca().title.set_text("Masked Image")
        plt.suptitle("Image "+str(k)+" : Masks & Bounding boxes for top 3 predictions", fontsize=14)
        plt.show()


    #Do the required analysis experiments.
    analysis_dict = data[3] #corresponding to 20CS30043
    title = ["Orininal","Horizontally_flipped","Blurred","Twice_rescaled","Half_rescaled","Right_rot_90_deg","Left_rot_45_deg"]
    k = 0
    for t in transforms:
        analysis_img = analysis_dict["image"]
        if t is not None: analysis_img = t(analysis_img)
        pred_boxes, pred_masks, pred_class, pred_score = segmentor(analysis_img)
        file_path = outputs+r"\\output_after_transform\\"+title[k]+".jpg"
        this_img = plot_visualization(analysis_img, pred_boxes, pred_masks, pred_class, pred_score, file_path, 0)
        plt.subplot(3,3,k+1)
        plt.imshow(this_img)
        plt.gca().title.set_text(title[k])
        k = k+1

    plt.suptitle("Results of segmentation on transformed images", fontsize=14)
    plt.show()

def main():
    segmentor = InstanceSegmentationModel()
    cwd = os.path.dirname(os.path.realpath(__file__))
    experiment(cwd+r'/data/annotations.jsonl', segmentor, [None,FlipImage('horizontal'),BlurImage(5),RescaleImage(2,1),RescaleImage(0.5,1),RotateImage(270),RotateImage(45)], cwd+r'/Outputs')


if __name__ == '__main__':
    main()

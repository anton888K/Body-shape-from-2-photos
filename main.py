import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import numpy as np
from PIL import Image
import glob
#import torch.optim as optim
#import numpy as np
import cv2
from regressCNN import RegressionPCA
from widgets_HSE import getBinaryimage, getSamplePoints, save_obj, repeat_data
from data_loader import RescaleT,ToTensorLab,SalObjDataset,normPRED,save_output
from data_loader import ToTensor
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


def main():
    name = 'test'
    # please input the height  unit:M
    body_height = 1.63

    #"Input image" path
    image_dir = os.path.join(os.getcwd(), 'input')

    #"Output model" path
    outbody_filenames = './output/{}.obj'.format(name)

    #########################################################
    #this code used for image segmentation to remove the background to get Silhouettes
    # --------- 1. get image path and name ---------

    model_name='u2net'#u2net or u2netp

    #set orignal silhouette images path
    prediction_dir1 = os.path.join(os.getcwd(), 'Silhouette' + os.sep)

    #set the path of silhouette images after horizontal flippath
    prediction_dir = os.path.join(os.getcwd(), 'test_data' + os.sep)

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------

    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. silhouette cutting model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir,prediction_dir1)

        del d1,d2,d3,d4,d5,d6,d7


    ###########################################################
    #this code used for reconstruct 3d model:

   #--------5.get the silhouette images --------
    img_filenames = ['./test_data/front.png', './test_data/side.png']

    # img = cv2.imread(img_filenames[1])
    # cv2.flip(img,1)



    # -----------6.load input data---------
    sampling_num = 648
    data = np.zeros([2, 2, sampling_num])
    for i in np.arange(len(img_filenames)):
        img = img_filenames[i]
        im = getBinaryimage(img, 600)  # deal with white-black image simply
        sample_points = getSamplePoints(im, sampling_num, i)
        center_p = np.mean(sample_points, axis=0)
        sample_points = sample_points - center_p
        data[i, :, :] = sample_points.T

    data = repeat_data(data)

    #--------7 load CNN model----reconstruct 3d body shape
    print('==> begining...')
    len_out = 22
    model_name = './Models/model.ckpt'
    ourModel = RegressionPCA(len_out)
    ourModel.load_state_dict(torch.load(model_name))
    ourModel.eval()

    #----------8 output results--------------
    save_obj(outbody_filenames, ourModel, body_height, data)
if __name__ == "__main__":
    main()

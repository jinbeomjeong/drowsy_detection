import platform 
import sys 
import torch 
import cv2 


def system_info(): 
    print('\n', "----- Environment Status -----") 
    print("Operating System Type:", platform.architecture())
    print('Python Version:', sys.version)
    print('Pytorch Version:', torch.__version__)
    print("Open Computer Vision Version:", cv2.__version__)
    print('GPU Available:', torch.cuda.is_available())
    print('GPU Device Name:', torch.cuda.get_device_name(0))
    print('Number of GPU:', torch.cuda.device_count())
    print("Tensor Core TF32 Enable:", torch.backends.cuda.matmul.allow_tf32, '\n')

import time, cv2, torch, threading, argparse, os
import torch.backends.cudnn as cudnn
import numpy as np 
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords 
from utils.plots import Annotator, colors
from utils.torch_utils import select_device 
from utils.accessory_lib import system_info
from PIPNet.lib.networks import Pip_resnet101
from PIPNet.lib.functions import forward_pip, get_meanface
from scipy.spatial import distance


class LoggingFile:
    def __init__(self, logging_header=pd.DataFrame(), file_name='logging_data'):
        self.start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.logging_file_name = file_name + '_' + self.start_time
        self.logging_file_path = './logging_data/' + self.logging_file_name + '.csv'
        logging_header.to_csv(self.logging_file_path, mode='a', header=True)

    def start_logging(self, period=0.1): 
        logging_data = pd.DataFrame({'1': time.strftime('%Y/%m/%d', time.localtime(time.time())),
                                     '2': time.strftime('%H:%M:%S', time.localtime(time.time())),
                                     '3': round(elapsed_time, 2), '4': ref_frame, '5': round(fps, 2)}, index=[0])
        logging_data.to_csv(self.logging_file_path, mode='a', header=False) 
        logging_thread = threading.Timer(period, self.start_logging, (period, )) 
        logging_thread.daemon = True 
        logging_thread.start() 


parser = argparse.ArgumentParser()
parser.add_argument("--fp16", type=int, default=0)
args = parser.parse_args()

net_stride = 32
num_nb = 10
data_name = 'data_300W'
experiment_name = 'pip_32_16_60_r101_l2_l1_10_1_nb10'
num_lms = 68
input_size = 256
det_box_scale = 1.2
eye_det = 0.15

transformations = transforms.Compose([transforms.Resize(448), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('PIPNet/data', data_name,
                                                                                      'meanface.txt'), num_nb)

resnet101 = models.resnet101(weights='ResNet101_Weights.DEFAULT')
landmark_net = Pip_resnet101(resnet101, num_nb=num_nb, num_lms=num_lms, input_size=input_size, net_stride=net_stride)

device = torch.device("cuda")

landmark_net = landmark_net.to(device)
save_dir = os.path.join('PIPNet/snapshots', data_name, experiment_name)
weight_file = os.path.join(save_dir, 'epoch%d.pth' % (60 - 1))
state_dict = torch.load(weight_file, map_location=device)
landmark_net.load_state_dict(state_dict)
landmark_net.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize])

elapsed_time: float = 0.0
fps: float = 0.0
ref_frame: int = 0
prev_time = time.time()
start_time = time.time()


@torch.no_grad()
def main():
    global elapsed_time, fps, ref_frame, prev_time, start_time
    system_info()

    source = "D:\\video\\night_driver_1.mp4"
    weights = "./weights/face_detection_yolov5s.pt"
    img_size = 640
    CONF_THRES = 0.4
    IOU_THRES = 0.45
    half = args.fp16
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False

    # Initialize
    device = select_device('')
    print(f'[1/3] Device Initialized {time.time()-prev_time:.2f}sec')
    prev_time = time.time()
    
    # Load model
    model = attempt_load(weights, device)  # load FP32 model
    model.eval()
    stride = int(model.stride.max())  # model stride
    img_size_chk = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    model(torch.zeros(1, 3, img_size_chk, img_size_chk).to(device).type_as(next(model.parameters())))  # run once
    print(f'[2/3] Yolov5 Detector Model Loaded {time.time()-prev_time:.2f}sec')
    prev_time = time.time()

    # Load image
    video = cv2.VideoCapture(source)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_COMPLEX
    print(f'[3/3] Video Resource Loaded {time.time()-prev_time:.2f}sec')
    start_time = time.time()

    # logging file threading start
    logging_header = pd.DataFrame(columns=['absolute_data', 'absolute_time', 'time(sec)', 'frame', 'throughput(fps)'])
    logging_task = LoggingFile(logging_header, file_name='logging_data')
    logging_task.start_logging(period=0.1)

    normalize_tensor = torch.tensor(255.0).to(device)
    cv2.namedWindow(winname='video', flags=cv2.WINDOW_NORMAL)

    while video.isOpened():
        ret, img0 = video.read()

        if ret:
            ref_frame = ref_frame + 1

            # Padded resize
            img = letterbox(img0, img_size_chk, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = torch.divide(img, normalize_tensor)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)

            # Process detections
            det = pred[0]
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=1, example=str(names))

            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls)))

                    det_xmin = int(xyxy[0])
                    det_ymin = int(xyxy[1])
                    det_xmax = int(xyxy[2])
                    det_ymax = int(xyxy[3])
                    det_width = det_xmax - det_xmin
                    det_height = det_ymax - det_ymin

                    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
                    det_ymin -= int(det_height * (det_box_scale - 1) / 2)
                    det_xmax += int(det_width * (det_box_scale - 1) / 2)
                    det_ymax += int(det_height * (det_box_scale - 1) / 2)

                    det_xmin = max(det_xmin, 0)
                    det_ymin = max(det_ymin, 0)
                    det_xmax = min(det_xmax, frame_width - 1)
                    det_ymax = min(det_ymax, frame_height - 1)

                    det_width = det_xmax - det_xmin + 1
                    det_height = det_ymax - det_ymin + 1
                    det_crop = img0[det_ymin:det_ymax, det_xmin:det_xmax, :]
                    det_crop = cv2.resize(det_crop, (input_size, input_size))

                    inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
                    inputs = preprocess(inputs).unsqueeze(0)
                    inputs = inputs.to(device)
                    lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(landmark_net,
                                                                                                             inputs, preprocess,
                                                                                                             input_size, net_stride, num_nb)

                    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                    tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
                    tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
                    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
                    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)

                    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                    lms_pred = lms_pred.cpu().numpy()
                    lms_pred_merge = lms_pred_merge.cpu().numpy()

                    eye_x = (lms_pred_merge[36 * 2:48 * 2:2] * det_width).astype(np.int32) + det_xmin
                    eye_y = (lms_pred_merge[(36 * 2) + 1:(48 * 2) + 1:2] * det_height).astype(np.int32) + det_ymin

                    left_eye_H_dist = distance.euclidean((eye_x[0], eye_y[0]), (eye_x[3], eye_y[3]))
                    left_eye_V_dist = min(distance.euclidean((eye_x[1], eye_y[1]), (eye_x[5], eye_y[5])),
                                          distance.euclidean((eye_x[2], eye_y[2]), (eye_x[4], eye_y[4])))

                    right_eye_H_dist = distance.euclidean((eye_x[6], eye_y[6]), (eye_x[9], eye_y[9]))
                    right_eye_V_dist = min(distance.euclidean((eye_x[11], eye_y[11]), (eye_x[7], eye_y[7])),
                                           distance.euclidean((eye_x[10], eye_y[10]), (eye_x[8], eye_y[8])))

                    left_eye_ratio = left_eye_V_dist / left_eye_H_dist
                    right_eye_ratio = right_eye_V_dist / right_eye_H_dist

                    left_eye_color = (0, 255, 0) if left_eye_ratio >= eye_det else (0, 0, 255)
                    right_eye_color = (0, 255, 0) if right_eye_ratio >= eye_det else (0, 0, 255)

                    for i in range(len(eye_x)):
                        if i <= 5:
                            cv2.circle(img0, (eye_x[i], eye_y[i]), 1, left_eye_color, 1)
                        else:
                            cv2.circle(img0, (eye_x[i], eye_y[i]), 1, right_eye_color, 1)

            fps = 1/(time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(img0, f'Elapsed Time(sec): {elapsed_time: .2f}', (5, 20), font, 0.5, [0, 0, 255], 1)
            cv2.putText(img0, f'Process Speed(FPS): {fps: .2f}', (5, 40), font, 0.5, [0, 0, 255], 1)
            cv2.putText(img0, f'Frame: {ref_frame}', (5, 60), font, 0.5, [0, 0, 255], 1)

            # Stream results
            cv2.imshow("video", img0)
            elapsed_time = time.time()-start_time

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print('Video Play Done!')


if __name__ == "__main__":
    main()

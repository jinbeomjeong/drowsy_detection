import time, cv2, torch, threading, argparse, os, can
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.accessory_lib import system_info
from PIPNet.networks import Pip_resnet101
from PIPNet.functions import forward_pip, get_meanface
from utils.eye_det_model import EyeDetModel
from scipy.spatial import distance
from utils.ai_hub_eye_data_module import eye_img_pos_expand


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

source = os.path.join('d:\\video\\closed_eye.mp4')

net_stride = 32
num_nb = 10
data_name = 'data_300W'
experiment_name = 'pip_32_16_60_r101_l2_l1_10_1_nb10'
num_lms = 68
face_landmark_input_size = 240
det_box_scale = 1.2
eye_det = 0.12

img_size = 640
CONF_THRES = 0.4
IOU_THRES = 0.45
half = args.fp16
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False

elapsed_time: float = 0.0
fps: float = 0.0
ref_frame: int = 0
det_frame: int = 0
prev_time = time.time()
start_time = time.time()
t0 = time.time()
eye_det_result = [0]*2

device = torch.device("cuda")

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('PIPNet', 'data', data_name,
                                                                                      'meanface.txt'), num_nb)

resnet101 = models.resnet101(weights='ResNet101_Weights.DEFAULT')
face_landmark_net = Pip_resnet101(resnet101, num_nb=num_nb, num_lms=num_lms, input_size=face_landmark_input_size,
                                  net_stride=net_stride)
face_landmark_net = face_landmark_net.to(device)

face_landmark_weight_file_path = os.path.join('PIPNet', 'snapshots', data_name, experiment_name, f'epoch{60-1}.pth')
face_landmark_net.load_state_dict(torch.load(f=face_landmark_weight_file_path, map_location=device))
face_landmark_net.eval()

face_landmark_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize((face_landmark_input_size, face_landmark_input_size)),
                                 transforms.ToTensor(), face_landmark_normalize])

face_detector_weight_file_path = os.path.join('.', 'weights', 'face_detection_yolov5s.pt')
face_det_normalize_tensor = torch.tensor(255.0).to(device)

#eye_img_preprocess = transforms.Compose([transforms.ToTensor(),
                                         #transforms.Resize((224, 224)),
                                         #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#eye_det_model = EyeDetModel.load_from_checkpoint(checkpoint_path="logs/train_drowsy_det/version_4/checkpoints/epoch=49-step=2750.ckpt",
                                                 #map_location=device)
#eye_det_model.eval()

system_info()

# Initialize
print(f'[1/3] Device Initialized {time.time() - prev_time:.2f}sec')
prev_time = time.time()

# Load model
model = attempt_load(face_detector_weight_file_path, device)  # load FP32 model
model.eval()
stride = int(model.stride.max())  # model stride
img_size_chk = check_img_size(img_size, s=stride)  # check img_size

if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

# Run inference
model(torch.zeros(1, 3, img_size_chk, img_size_chk).to(device).type_as(next(model.parameters())))  # run once
print(f'[2/3] Yolov5 Detector Model Loaded {time.time() - prev_time:.2f}sec')
prev_time = time.time()

# Load video resource
video = cv2.VideoCapture(source)
#video.set(cv2.CAP_PROP_FRAME_WIDTH, value=1280)
#video.set(cv2.CAP_PROP_FRAME_HEIGHT, value=720)

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

font = cv2.FONT_HERSHEY_COMPLEX
print(f'[3/3] Video Resource Loaded {time.time() - prev_time:.2f}sec')
start_time = time.time()

# logging file threading start
#logging_header = pd.DataFrame(columns=['absolute_data', 'absolute_time', 'time(sec)', 'frame', 'throughput(fps)'])
#logging_task = LoggingFile(logging_header, file_name='logging_data')
#logging_task.start_logging(period=0.1)

cv2.namedWindow(winname='video', flags=cv2.WINDOW_NORMAL)


@torch.no_grad()
def main():
    global elapsed_time, fps, ref_frame, det_frame, t0

    while video.isOpened():
        elapsed_time = time.time() - start_time
        ret, img0 = video.read()

        if ret:
            img = letterbox(img0, img_size_chk, stride=stride)[0]  # Padded resize

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = torch.divide(img.to(device), face_det_normalize_tensor)
            img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]

            # Apply NMS
            detected_face = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)[0]

            # Process detections
            if len(detected_face):
                # Rescale boxes from img_size to img0 size
                detected_face[:, :4] = scale_coords(img.shape[2:], detected_face[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(detected_face):
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
                    det_crop = cv2.resize(src=det_crop, dsize=(face_landmark_input_size, face_landmark_input_size),
                                          interpolation=cv2.INTER_AREA)
                    #cv2.rectangle(img0, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 1)

                    inputs = preprocess(det_crop).to(device)
                    inputs = inputs.unsqueeze(0)
                    lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(face_landmark_net,
                                                                                                             inputs, preprocess,
                                                                                                             face_landmark_input_size,
                                                                                                             net_stride, num_nb)

                    # lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                    tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
                    tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
                    tmp_x = torch.mean(torch.cat(tensors=(lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
                    tmp_y = torch.mean(torch.cat(tensors=(lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)

                    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                    # lms_pred = lms_pred.cpu().numpy()
                    lms_pred_merge = lms_pred_merge.cpu().numpy()

                    eye_x = (lms_pred_merge[36 * 2:48 * 2:2] * det_width).astype(np.int32) + det_xmin
                    eye_y = (lms_pred_merge[(36 * 2) + 1:(48 * 2) + 1:2] * det_height).astype(np.int32) + det_ymin

                    left_eye_x_pos = eye_x[0:6]
                    left_eye_y_pos = eye_y[0:6]
                    right_eye_x_pos = eye_x[6:12]
                    right_eye_y_pos = eye_y[6:12]

                    left_eye_y_pos_mean = np.mean(left_eye_y_pos)
                    right_eye_y_pos_mean = np.mean(right_eye_y_pos)

                    left_eye_pos = [np.min(left_eye_x_pos), np.min(left_eye_y_pos), np.max(left_eye_x_pos), np.max(left_eye_y_pos)]
                    right_eye_pos = [np.min(right_eye_x_pos), np.min(right_eye_y_pos), np.max(right_eye_x_pos), np.max(right_eye_y_pos)]

                    left_eye_width = left_eye_pos[2]-left_eye_pos[0]
                    left_eye_y_pos_min = int(left_eye_y_pos_mean-(left_eye_pos[2]-left_eye_pos[0])/2)
                    left_eye_y_pos_max = int(left_eye_y_pos_mean+(left_eye_pos[2]-left_eye_pos[0])/2)

                    right_eye_width = right_eye_pos[2]-right_eye_pos[0]
                    right_eye_y_pos_min = int(right_eye_y_pos_mean-(right_eye_pos[2]-right_eye_pos[0])/2)
                    right_eye_y_pos_max = int(right_eye_y_pos_mean+(right_eye_pos[2]-right_eye_pos[0])/2)

                    left_eye_pos_est = [left_eye_pos[0]-int(left_eye_width*0.4), left_eye_y_pos_min-int(left_eye_width*0.2),
                                        left_eye_pos[2]+int(left_eye_width*0.4), left_eye_y_pos_max+int(left_eye_width*0.2)]

                    right_eye_pos_est = [right_eye_pos[0]-int(right_eye_width*0.4), right_eye_y_pos_min-int(right_eye_width*0.3),
                                         right_eye_pos[2]+int(right_eye_width*0.4), right_eye_y_pos_max+int(right_eye_width*0.3)]

                    #cv2.rectangle(img0, (left_eye_pos_est[0], left_eye_pos_est[1]), (left_eye_pos_est[2], left_eye_pos_est[3]), (0, 0, 255), 1)
                    #cv2.rectangle(img0, (right_eye_pos_est[0], right_eye_pos_est[1]), (right_eye_pos_est[2], right_eye_pos_est[3]), (0, 0, 255), 1)

                    left_eye_img = img0[left_eye_pos_est[1]:left_eye_pos_est[3], left_eye_pos_est[0]:left_eye_pos_est[2]]
                    right_eye_img = img0[right_eye_pos_est[1]:right_eye_pos_est[3], right_eye_pos_est[0]:right_eye_pos_est[2]]

                    #left_eye_img_t = eye_img_preprocess(left_eye_img).unsqueeze(dim=0).to(device)
                    #right_eye_img_t = eye_img_preprocess(right_eye_img).unsqueeze(dim=0).to(device)

                    #eye_det_result_t = eye_det_model.forward(torch.concatenate(tensors=(left_eye_img_t, right_eye_img_t), dim=0))
                    #eye_det_result_t = eye_det_result_t.cpu().numpy()

                    #for i in range(len(eye_det_result)):
                        #eye_det_result[i] = (np.argmax(eye_det_result_t[i]))

                    left_eye_color = (0, 255, 0) if eye_det_result[0] else (0, 0, 255)
                    right_eye_color = (0, 255, 0) if eye_det_result[1] else (0, 0, 255)

                    #drowsy_det = left_eye_det_result and right_eye_det_result

                    #if drowsy_det:
                    #    det_frame += 1
                    #print(left_eye_pos)

                    #cv2.rectangle(img0, (left_eye_pos_est[0], left_eye_pos_est[1]), (left_eye_pos_est[2], left_eye_pos_est[3]), left_eye_color, 1)
                    #cv2.rectangle(img0, (right_eye_pos_est[0], right_eye_pos_est[1]), (right_eye_pos_est[2], right_eye_pos_est[3]), right_eye_color, 1)

                    for i in range(len(eye_x)):
                        if i <= 5:
                            cv2.circle(img0, (eye_x[i], eye_y[i]), 1, (0, 255, 0), 1)
                        else:
                            cv2.circle(img0, (eye_x[i], eye_y[i]), 1, (0, 255, 0), 1)

            fps = 1/(time.time() - t0)
            t0 = time.time()

            #cv2.putText(img0, f'Elapsed Time(sec): {elapsed_time: .2f}', (5, 20), font, 0.5, [0, 0, 255], 1)
            #cv2.putText(img0, f'Process Speed(FPS): {fps: .2f}', (5, 40), font, 0.5, [0, 0, 255], 1)
            #cv2.putText(img0, f'Frame: {ref_frame}', (5, 60), font, 0.5, [0, 0, 255], 1)
            #cv2.putText(img0, f'Det Frame: {det_frame}', (5, 80), font, 0.5, [0, 0, 255], 1)

            # Stream results
            cv2.imshow("video", img0)
            ref_frame += 1

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print('Video Play Done!')


if __name__ == "__main__":
    main()

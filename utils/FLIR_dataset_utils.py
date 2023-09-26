import json, shutil, yaml
from glob import glob
from tqdm import tqdm
from models.experimental import attempt_load
from sklearn.model_selection import train_test_split


def flir_data_set_parser():
    # load train image dataset
    root_path = '/home/jinbeom/workspace/object_detection/'
    train_img_list = glob(root_path + 'FLIR_ADAS/train/image/*.jpeg')
    val_img_list = glob(root_path + 'FLIR_ADAS/val/image/*.jpeg')

    print('train number of images:', len(train_img_list))
    print('validation number of images:', len(val_img_list))

    # create train & val img name list file
    with open('./train_img_name_list.txt', 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')

    with open('./val_img_name_list.txt', 'w') as f:
        f.write('\n'.join(val_img_list) + '\n')

    # load annotation data(based coco data format)
    with open('./FLIR_ADAS/thermal_annotations_train.json', 'r') as flir_json:
        train_json = json.load(flir_json)

    with open('./FLIR_ADAS/thermal_annotations_val.json', 'r') as flir_json:
        val_json = json.load(flir_json)

    # create dataset dictionary
    train_img_id_dict = {}; val_img_id_dict = {}

    for i in train_json['images']: train_img_id_dict[i['id']] = i['file_name']

    for i in val_json['images']: val_img_id_dict[i['id']] = i['file_name']

    # create class label
    img_height = 512
    img_width = 640

    for image_id, file_name in tqdm(train_img_id_dict.items(), desc='create train json class label'):
        file_contents = ''
        file_path = file_name.split('/')[1]
        file_path = './FLIR_ADAS/train/label' + file_path.split('.')[0] + '.txt'

        for i in train_json['annotations']:
            if image_id == i['image_id']:
                if i['category_id'] == 1: #search prson class bounding box
                    bbox = i['bbox']
                    bbox = (bbox[0] + (bbox[2] / 2.0)) / img_width, ((bbox[1] + (bbox[3])/ 2.0)) / img_height, bbox[2] / img_width, bbox[3] / img_height
                    file_contents += '0' + ' ' + str(bbox).replace(',', '').replace('(', '').replace(')', '') + '\n'

                elif i['category_id'] == 3: #search car class bounding box
                    bbox = i['bbox']
                    bbox = (bbox[0] + (bbox[2] / 2.0)) / img_width, ((bbox[1] + (bbox[3]) / 2.0)) / img_height, bbox[2] / img_width, bbox[3] / img_height
                    file_contents += '1' + ' ' + str(bbox).replace(',', '').replace('(', '').replace(')', '') + '\n'
                else: continue
            else: continue

        with open(file_path, 'w') as f: f.write(file_contents)

    for image_id, file_name in tqdm(val_img_id_dict.items(), desc='create val json class label'):
        file_contents = ''
        file_path = file_name.split('/')[1]
        file_path = './FLIR_ADAS/val/label' + file_path.split('.')[0] + '.txt'

        for i in val_json['annotations']:
            if image_id == i['image_id']:
                if i['category_id'] == 1: #search prson class bounding box
                    bbox = i['bbox']
                    bbox = (bbox[0] + (bbox[2] / 2.0)) / img_width, ((bbox[1] + (bbox[3])/ 2.0)) / img_height, bbox[2] / img_width, bbox[3] / img_height
                    file_contents += '0' + ' ' + str(bbox).replace(',', '').replace('(', '').replace(')', '') + '\n'

                elif i['category_id'] == 3: #search car class bounding box
                    bbox = i['bbox']
                    bbox = (bbox[0] + (bbox[2] / 2.0)) / img_width, ((bbox[1] + (bbox[3]) / 2.0)) / img_height, bbox[2] / img_width, bbox[3] / img_height
                    file_contents += '1' + ' ' + str(bbox).replace(',', '').replace('(', '').replace(')', '') + '\n'
                else: continue
            else: continue

        with open(file_path, 'w') as f: f.write(file_contents)


def copy_validation_result_files():
    weight = "/home/nx/object_detection/yolov5/FLIR_test.pt"
    source_root_path = "/home/nx/object_detection/"
    destination_root_path = source_root_path + "result/"

    train_label_list = glob(source_root_path + "FLIR_ADAS/train/label/*.txt")
    val_label_list = glob(source_root_path + "FLIR_ADAS/val/label/*.txt")
    ref_label_list = train_label_list + val_label_list
    pred_label_list = glob(source_root_path + "yolov5/Pytorch_fp32/*.txt")

    model = attempt_load(weight, map_location='cpu')  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names

    print("Number of Ground Truth Label Files:", len(ref_label_list))
    print("Number of predicted Label Files:", len(pred_label_list))

    for pred_label in tqdm(pred_label_list, desc="File Copy Processing", ncols=100):
        pred_label_path, pred_label_name = opt.split(pred_label)

        for ref_label in ref_label_list:
            ref_label_path, ref_label_name = opt.split(ref_label)

            if pred_label_name == ref_label_name:
                shutil.copy2(pred_label, destination_root_path + 'detections/' + pred_label_name)
                ref_file = open(destination_root_path + 'groundtruths/' + ref_label_name, 'w')

                file = open(ref_label, 'r')

                while True:
                    file_write_data = []
                    string = file.readline()
                    string_list = string.split()

                    if not string: break
                    file_write_data.append(names[int(string_list[0])])
                    file_write_data.append(string_list[1])
                    file_write_data.append(string_list[2])
                    file_write_data.append(string_list[3])
                    file_write_data.append(string_list[4])

                    ref_file.write(' '.join(file_write_data) + '\n')

def create_flir_data_yaml_format():
    root_path = "/home/jinbeom/workspace/object_detection/FLIR_ADAS/"

    img_list = glob(root_path + "images/*.jpeg")
    print('dataset number of images:', len(img_list))

    train_img_list, val_img_list = train_test_split(img_list, test_size=0.3, random_state=100)
    val_img_list, test_img_list = train_test_split(val_img_list, test_size=0.66, random_state=100)

    print("train number of images:", len(train_img_list))
    print("validation number of images:", len(val_img_list))
    print("Test number of images:", len(test_img_list))

    with open(root_path + "train_img_name_list.txt", 'w') as file: file.write('\n'.join(train_img_list) + '\n')

    with open(root_path + "val_img_name_list.txt", 'w') as file: file.write('\n'.join(val_img_list) + '\n')

    with open(root_path + "test_img_name_list.txt", 'w') as file: file.write('\n'.join(test_img_list) + '\n')

    with open(root_path + "FLIR_Data.yaml", 'r') as f: dataset_label = yaml.load(f, Loader=yaml.FullLoader)

    dataset_label['train'] = root_path + 'train_img_name_list.txt'
    dataset_label['val'] = root_path + 'val_img_name_list.txt'
    dataset_label['test'] = root_path + 'test_img_name_list.txt'

    with open(root_path + "FLIR_Data.yaml", 'w') as f: yaml.dump(dataset_label, f)

    print(dataset_label)

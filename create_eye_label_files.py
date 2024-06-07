import time, os, json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import freeze_support
from tqdm import tqdm
from utils.ai_hub_eye_data_module import create_file_path_list


column_names = ['file_name', 'eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos', 'closed']
start_time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logging_file_name = 'result' + '_' + start_time_str
logging_file_path = 'label_data' + os.sep + logging_file_name + '.csv'

root_path = "d:\\driver_img_set\\val"
root_img_path = root_path + os.sep + "img"
root_label_path = root_path + os.sep + "label"

real_env_label_dir = root_label_path + os.sep + "label_real_env"
semi_constrain_env_label_dir = root_label_path + os.sep + "label_semi_constrain_env"
constrain_env_label_dir = root_label_path + os.sep + "label_constrain_env"


img_file_path_list = []

for (root, dirs, files) in os.walk(root_img_path):
    if len(files) > 0:
        for file_name in files:
            img_file_path_list.append(root + os.sep + file_name)

target_file_path_list = []
target_file_name_list = []

for i, file_path in enumerate(img_file_path_list):
    target_file_path, target_file_name = os.path.split(file_path)
    target_file_path_list.append(target_file_path)
    target_file_name_list.append(target_file_name)

target_file_name_arr = np.array(target_file_name_list)


def load_csv(label_path: str):
    label_data = None

    try:
        label_data = json.load(open(file=label_path, encoding='UTF-8'))
        json_load_error = False
    except:
        json_load_error = True

    if not json_load_error:
        left_eye_label = pd.DataFrame()
        right_eye_label = pd.DataFrame()
        file_name = label_data['FileInfo']['FileName']

        idx = np.argmax(file_name == target_file_name_arr)
        img_file_path = target_file_path_list[idx] + os.sep + target_file_name_arr[idx]

        if label_data['ObjectInfo']['BoundingBox']['Face']['isVisible']:
            if label_data['ObjectInfo']['BoundingBox']['Leye']['isVisible']:
                pos = label_data['ObjectInfo']['BoundingBox']['Leye']['Position']
                label = 1-int(label_data['ObjectInfo']['BoundingBox']['Leye']['Opened'])
                left_eye_label = pd.DataFrame(data=[[img_file_path, pos[0], pos[1], pos[2], pos[3], label]], columns=column_names)

            if label_data['ObjectInfo']['BoundingBox']['Reye']['isVisible']:
                pos = label_data['ObjectInfo']['BoundingBox']['Reye']['Position']
                label = 1-int(label_data['ObjectInfo']['BoundingBox']['Reye']['Opened'])
                right_eye_label = pd.DataFrame(data=[[img_file_path, pos[0], pos[1], pos[2], pos[3], label]], columns=column_names)

        return pd.concat([left_eye_label, right_eye_label], axis=0)


def load_csv_executor(path_list, n_of_workers: int):
    with ThreadPoolExecutor(max_workers=n_of_workers) as executor:
        df_list = list(tqdm(executor.map(load_csv, path_list), desc='dataframe loading...', total=len(path_list)))

    return pd.concat(df_list, axis=0)


if __name__ == '__main__':
    freeze_support()

    real_env_label_path_list = create_file_path_list(real_env_label_dir)
    semi_constrain_env_label_path_list = create_file_path_list(semi_constrain_env_label_dir)
    constrain_env_label_path_list = create_file_path_list(constrain_env_label_dir)
    label_path_list = real_env_label_path_list + semi_constrain_env_label_path_list + constrain_env_label_path_list

    result_df = load_csv_executor(real_env_label_path_list, n_of_workers=12)
    result_df.to_csv(logging_file_path)

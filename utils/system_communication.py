import can, cantools, struct, time, serial
import numpy as np


class AdasCANCommunication:
    def __init__(self, dbc_filename: str):
        self.__t0 = time.time()
        self.__adas_can_db = cantools.db.load_file(filename=dbc_filename)
        self.__vehicle_status_message = self.__adas_can_db.get_message_by_name(name='CLU_info')
        self.__dsm_message = self.__adas_can_db.get_message_by_name(name='DSM')
        self.__fcw_message = self.__adas_can_db.get_message_by_name(name='FCW')
        self.__ldws_message = self.__adas_can_db.get_message_by_name(name='LDWS')
        self.__vehicle_speed = 0.0
        self.__steering_angle = 0.0
        self.__winker_status = np.zeros(2, dtype=np.uint8)  # index 0 is left, index 1 is right

    def __time_elapsed(self) -> float:
        return time.time() - self.__t0

    def get_vehicle_status(self, packet: can.Message):
        if packet.arbitration_id == self.__vehicle_status_message.frame_id:
            data = self.__adas_can_db.decode_message(packet.arbitration_id, packet.data)
            self.__vehicle_speed = data['vehicle_speed_high_precision']
            self.__steering_angle = data['steering_angle']
            self.__winker_status[0] = data['turn_signal_left_indicate']
            self.__winker_status[1] = data['turn_signal_right_indicate']

    def create_dsm_can_msg(self, dsm_state: int = 0) -> can.Message():
        data = self.__dsm_message.encode(data={'state_of_DSM': dsm_state})
        message = can.Message(arbitration_id=self.__dsm_message.frame_id, data=data, is_extended_id=False,
                              timestamp=self.__time_elapsed())
        return message

    def create_fcw_can_msg(self, fcw_state: int = 0, object_type: int = 0, ttc: float = 0.0) -> can.Message():
        data = self.__fcw_message.encode(data={'state_of_FCW': fcw_state, 'object_type': object_type,
                                               'time_to_collision': ttc})
        message = can.Message(arbitration_id=self.__fcw_message.frame_id, data=data, is_extended_id=False,
                              timestamp=self.__time_elapsed())
        return message

    def create_ldws_can_msg(self, ldws_state: int = 0) -> can.Message():
        data = self.__ldws_message.encode(data={'state_of_LDWS': ldws_state})
        message = can.Message(arbitration_id=self.__ldws_message.frame_id, data=data, is_extended_id=False,
                              timestamp=self.__time_elapsed())
        return message

    def read_vehicle_speed(self) -> float:
        return self.__vehicle_speed

    def read_winker_status(self) -> np.ndarray:
        return self.__winker_status

    def read_steering_angle(self) -> float:
        return self.__steering_angle


class ClusterCANCommunication:
    def __init__(self, dbc_filename: str):
        self.__t0 = time.time()
        self.__adas_can_db = cantools.db.load_file(filename=dbc_filename)
        self.__cluster_status_message = self.__adas_can_db.get_message_by_name(name='CLU_VCU_2A1')

    def __time_elapsed(self) -> float:
        return time.time() - self.__t0

    def create_dsm_can_msg(self, dsm_state: int = 0) -> can.Message():
        data = self.__cluster_status_message.encode(data={'dsm': dsm_state})
        message = can.Message(arbitration_id=self.__cluster_status_message.frame_id, data=data, is_extended_id=False,
                              timestamp=self.__time_elapsed())
        return message

    def create_ldws_can_msg(self, ldws_state: int = 0) -> can.Message():
        data = self.__cluster_status_message.encode(data={'LaneDepature': ldws_state})
        message = can.Message(arbitration_id=self.__cluster_status_message.frame_id, data=data, is_extended_id=False,
                              timestamp=self.__time_elapsed())
        return message

    def create_fcw_can_msg(self, fcw_state: int = 0) -> can.Message():
        data = self.__cluster_status_message.encode(data={'fcws': fcw_state})
        message = can.Message(arbitration_id=self.__cluster_status_message.frame_id, data=data, is_extended_id=False,
                              timestamp=self.__time_elapsed())
        return message


class Radar:
    def __init__(self, serial_port='COM1', serial_baudrate=921600):
        self.__com = serial.Serial(port=serial_port, baudrate=serial_baudrate, timeout=None)
        self.__buffer = np.zeros(2000, dtype=np.uint8)
        self.__n_buffer = 0
        self.__header_1 = 0
        self.__header_2 = 0
        self.__n_packets = 0
        self.__n_frames = 0
        self.__n_point_clouds = 0
        self.__point_clouds = np.zeros((200, 4), dtype=np.float32)

        time.sleep(0.05)
        self.__com.setDTR(False)
        self.__com.write('$CM+START'.encode())
        print('Radar Sensor is Initialized')

    def get_point_clouds(self):
        self.__n_buffer = self.__com.in_waiting

        if self.__n_buffer > 1:
            if self.__n_buffer < 1800:
                for i, data in enumerate(self.__com.read(self.__n_buffer)):
                    self.__buffer[i] = data
            else:
                self.__com.readline()  # flush buffer
        else:
            time.sleep(0.04)

        self.__header_1 = int.from_bytes(self.__buffer[0:2].tobytes(), byteorder='little', signed=False)
        self.__header_2 = int.from_bytes(self.__buffer[2:4].tobytes(), byteorder='little', signed=False)
        self.__n_packets = int.from_bytes(self.__buffer[4:8].tobytes(), byteorder='little', signed=False)
        self.__n_frames = int.from_bytes(self.__buffer[8:12].tobytes(), byteorder='little', signed=False)
        self.__n_point_clouds = int.from_bytes(self.__buffer[12:14].tobytes(), byteorder='little', signed=False)

        if 1 < self.__n_point_clouds < 180:
            self.__point_clouds = np.zeros((self.__n_point_clouds, 4), dtype=np.float32)
            point_clouds_raw = self.__buffer[14: 14 + self.__n_point_clouds * 16]
            point_clouds_raw = point_clouds_raw.reshape([self.__n_point_clouds, 16])

            for i, point in enumerate(point_clouds_raw):
                for j in range(4):
                    self.__point_clouds[i, j] = struct.unpack('<f', point[j * 4:(j + 1) * 4].tobytes())[0]

    def read_point_clouds(self):
        return self.__point_clouds

    def read_n_point_clouds(self):
        return self.__n_point_clouds

    def read_n_buffer(self):
        return self.__n_buffer

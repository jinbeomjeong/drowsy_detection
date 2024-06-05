def eye_img_pos_expand(eye_pos, h_offset_scale, v_offset_scale):
    eye_pos[0] = int(eye_pos[0] - (eye_pos[0] * h_offset_scale))
    eye_pos[1] = int(eye_pos[1] - (eye_pos[1] * v_offset_scale))
    eye_pos[2] = int(eye_pos[2] + (eye_pos[2] * h_offset_scale))
    eye_pos[3] = int(eye_pos[3] + (eye_pos[3] * v_offset_scale))

    return eye_pos
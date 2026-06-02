import matplotlib.pyplot as plt
import numpy as np


def decode_obj_trajs(obj_trajs):
    obj_trajs_xy = obj_trajs[..., :2]
    obj_lw = obj_trajs[..., -1, 3:5]
    obj_type_onehot = obj_trajs[..., -1, 6:9]
    obj_type = np.argmax(obj_type_onehot, axis=-1)
    obj_heading_encoding = obj_trajs[..., -1, 33:35]
    return obj_trajs_xy, obj_lw, obj_type, obj_heading_encoding


def interpolate_color_ego(t, total_t):
    # Start is red, end is blue
    return (1 - t / total_t, 0, t / total_t)

def interpolate_color(t, total_t):
    # Start is green, end is blue
    return (0, 1 - t / total_t, t / total_t)


def draw_line_with_mask(point1, point2, color, line_width=4):
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)


def draw_trajectory(trajectory, line_width, ego=False):
    total_t = len(trajectory)
    for t in range(total_t - 1):
        if ego:
            color = interpolate_color_ego(t, total_t)
            if trajectory[t, 0] and trajectory[t + 1, 0]:
                draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
        else:
            color = interpolate_color(t, total_t)
            if trajectory[t, 0] and trajectory[t + 1, 0]:
                draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)


def decode_map(map):
    map_xy = map[..., :2]
    map_type = map[..., 0, 9:29]
    map_type = np.argmax(map_type, axis=-1)
    return map_xy, map_type

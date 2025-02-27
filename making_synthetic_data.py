import os
import json
import numpy as np
import mitsuba as mi
from renderer import render_scene

np.random.seed(0)

def sample_uniform_sphere_surface(r: float, y_min: float, num_samples: int) -> np.ndarray:
    assert r != 0
    min_theta = np.arcsin(y_min / r)
    max_theta = np.radians(90)
    samples = []
    step_size = 2 * np.pi / num_samples
    i = 0
    while len(samples) < num_samples:
        phi = i * step_size
        theta = (max_theta + min_theta) * np.random.uniform(0, 1) + min_theta

        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta)
        z = -r * np.cos(theta) * np.sin(phi)

        if y_min <= y:
            samples.append([x, y, z])
            i += 1
    
    return np.array(samples)

def get_view_matrix(pos: np.ndarray, up: np.ndarray, to: np.ndarray) -> np.ndarray:
    zaxis = (pos - to) / np.linalg.norm(pos - to)
    xaxis = np.cross(up, zaxis) / np.linalg.norm(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis) / np.linalg.norm(np.cross(zaxis, xaxis))

    tx = -np.dot(xaxis, pos)
    ty = -np.dot(yaxis, pos)
    tz = -np.dot(zaxis, pos)

    ret = np.array([
        [xaxis[0], xaxis[1], xaxis[2], tx], 
        [yaxis[0], yaxis[1], yaxis[2], ty], 
        [zaxis[0], zaxis[1], zaxis[2], tz], 
        [0.0, 0.0, 0.0, 1.0]
    ])

    return ret

"""
Parameters
"""
output_dir = "output/"
is_constant_emitter = True
width = 1920
height = 1080
spp = 16 # Please edit here to adjust the number of sampling!
fov = 35.0
r = 30.0
y_min = -5.0
num_train_samples = 60
num_test_samples = 10

os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, 'train'))
os.makedirs(os.path.join(output_dir, 'test'))
train_samples = sample_uniform_sphere_surface(r, y_min, num_train_samples)
test_samples = sample_uniform_sphere_surface(r, y_min, num_test_samples)
cam_target = [0, 0, 0]
cam_up = [0, 1, 0]

"""
Making train data
"""
train_transform_json = {}
train_transform_json['camera_angle_x'] = np.radians(fov)
train_frame_list = []
for i, pos in enumerate(train_samples):
    fname = f"image_{i}.png"
    print(f"Rendering {i}th image for train.")
    render_scene(
        width = width, 
        height = height, 
        spp = spp, 
        cam_pos = pos, 
        cam_target = cam_target, 
        cam_up = cam_up, 
        fov = fov, 
        output_path= os.path.join(output_dir, 'train', fname), 
        is_constant_emitter=is_constant_emitter
    )
    # Generating c2w matrix with my logic.
    # Because the look_at method in Mitsuba generate c2w matrix that a camera see positive z-axis direction.
    w2c = get_view_matrix(pos, np.array(cam_up), np.array(cam_target))
    c2w = np.linalg.inv(w2c)
    c2w = [list(row) for row in c2w]
    train_frame_list.append({
        'file_path': os.path.join('./train/', fname.split('.')[0]), 
        'rotation': i * 2 * np.pi / num_train_samples, 
        'transform_matrix': c2w
    })

train_transform_json['frames'] = train_frame_list
with open(os.path.join(output_dir, 'transforms_train.json'), 'w') as f:
    json.dump(train_transform_json, f, indent=2)

"""
Making test data
"""
test_transform_json = {}
test_transform_json['camera_angle_x'] = np.radians(fov)
test_frame_list = []
for i, pos in enumerate(test_samples):
    fname = f"image_{i}.png"
    print(f"Rendering {i}th image for test.")
    render_scene(
        width = width, 
        height = height, 
        spp = spp, 
        cam_pos = pos, 
        cam_target = cam_target, 
        cam_up = cam_up, 
        fov = fov, 
        output_path= os.path.join(output_dir, 'test', fname.split('.')[0]), 
        is_constant_emitter=is_constant_emitter
    )
    # Generating c2w matrix with my logic.
    # Because the look_at method in Mitsuba generate c2w matrix that a camera see positive z-axis direction.
    w2c = get_view_matrix(pos, np.array(cam_up), np.array(cam_target))
    c2w = np.linalg.inv(w2c)
    c2w = [list(row) for row in c2w]
    test_frame_list.append({
        'file_path': os.path.join('./test/', fname), 
        'rotation': i * 2 * np.pi / num_test_samples, 
        'transform_matrix': c2w
    })

test_transform_json['frames'] = test_frame_list
with open(os.path.join(output_dir, 'transforms_test.json'), 'w') as f:
    json.dump(test_transform_json, f, indent=2)

import os
import math
import mitsuba as mi
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 


print("mi.__version__: ", mi.__version__)
mi.set_variant('scalar_rgb')
# mi.set_variant('cuda_ad_rgb')
# mi.set_variant('cuda_ad_spectral_polarized')

# Please edit here and set some shapes!
shape_dict = {

}

# Emitter settings
point_emitter_dict = {}
theta_deg_list = [50, 0]
num_splits = 10
r = 40
for i, theta_deg in enumerate(theta_deg_list):
    theta = np.radians(theta_deg)
    for j in range(num_splits):
        phi = 2 * np.pi * j / num_splits
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta)
        z = -r * np.cos(theta) * np.sin(phi)
        index = i * num_splits + j
        point_emitter_dict.update({
            f'emitter_{index}': {
                'type': 'point', 
                'id': f'elm__{index}', 
                # 'name': 'elm__1', 
                'position': [x, y, z], 
                'intensity': {
                    'type': 'spectrum', 
                    'value': 1000.0, 
                }
            }
        })

constant_emitter_dict = {
    'emmiter_0': {
        'type': 'constant', 
        'radiance': {
            'type': 'rgb', 
            'value': 1.0
        }
    }
}

scene_dict = {
    'type': 'scene', 
    'integrator': {
        'type': 'path', 
        'max_depth': 2
    }
}

def get_scene(is_constant_emitter: bool):
    scene_dict.update(shape_dict)
    if is_constant_emitter:
        scene_dict.update(constant_emitter_dict)
    else:
        scene_dict.update(point_emitter_dict)
    scene = mi.load_dict(scene_dict)
    return scene

def render_scene(
    width: int, 
    height: int, 
    spp: int, 
    cam_pos: list, 
    cam_target: list, 
    cam_up: list, 
    fov: float, 
    output_path: str, 
    is_constant_emitter: bool, 
):
    scene = get_scene(
        is_constant_emitter=is_constant_emitter, 
        background_color=[0.0, 1.0, 0.0]
    )
    tile_size = 256
    rendered_img = np.array([[[0.0] * 3 for _ in range(width)] for _ in range(height)])
    for y in range(math.ceil(height / tile_size)):
        for x in range(math.ceil(width / tile_size)):
            crop_offset_x = x * tile_size
            crop_offset_y = y * tile_size
            crop_width = min(tile_size, width - crop_offset_x)
            crop_height = min(tile_size, height - crop_offset_y)
            sensor = mi.load_dict({
                'type': "perspective", 
                'fov': fov, 
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=cam_pos, 
                    target=cam_target, 
                    up=cam_up
                ), 
                'sampler': {
                    'type': 'independent', 
                    'sample_count': spp
                }, 
                'film': {
                    'type': 'hdrfilm', 
                    'width': width, 
                    'height': height, 
                    'crop_offset_x': crop_offset_x, 
                    'crop_offset_y': crop_offset_y, 
                    'crop_width': crop_width, 
                    'crop_height': crop_height, 
                    'rfilter': {
                        'type': 'tent', 
                    }, 
                    'pixel_format': 'rgb', 
                }, 
            })
            image = mi.render(scene, sensor=sensor)
            image = np.array(image.array, dtype=np.float32)
            image = image.reshape([crop_height, crop_width, 3])
            rendered_img[
                crop_offset_y: crop_offset_y + crop_height, 
                crop_offset_x: crop_offset_x + crop_width, 
                :
            ] = image

    rendered_img = mi.TensorXf(rendered_img)
    mi.util.write_bitmap(output_path, rendered_img)
    print(f"Rendering completed: {output_path}")


if __name__ == "__main__":
    render_scene(
        width = 1920, 
        height = 1080, 
        spp = 32, 
        cam_pos = [35, 0, 0], 
        cam_target = [0, 0, 0], 
        cam_up = [0, 1, 0], 
        fov=35.0, 
        output_path= "output.png", 
        is_constant_emitter=True
    )

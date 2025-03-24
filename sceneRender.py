import bpy
import sys
import mathutils
from mathutils import Vector, Matrix, Euler
import argparse
import numpy as np
import math
import os
# import time
# import pickle
# from PIL import Image
import random

import pandas as pd
import torch
import json
import time

class UIDSampler:
    def __init__(self, file_path):
        with open(file_path) as f:
            self.data = json.load(f)
        # print(len(self.data["walking stick, walkingstick, stick insect"]))
        for key in list(self.data.keys()):
            self.data[key.lower().split(',')[0]] = self.data.pop(key)
    def get_obj_uid(self, cate):
        random.seed(time.time())  # 使用系统时间作为随机数种子
        # random.seed(5678)
        uid = random.choice(self.data[cate.lower().replace('_', ' ')])
        return uid

def get_file_size_in_mb(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

## solve the division problem
from decimal import Decimal, getcontext
getcontext().prec = 28  # Set the precision for the decimal calculations.

parser = argparse.ArgumentParser()
# parser.add_argument('--object_path_pkl', type = str, required = True)
# parser.add_argument("--parent_dir", type = str, default='./example_material')
parser.add_argument("--save_root", type = str, default='./output_blender/')
parser.add_argument("--worker", type = int, default=0)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

worker = int(args.worker)

# save_root = '../../../mnt/sdd/SceneRender/2025_fix_camera_distance_7/'
save_root = '../../../mnt/sdd/SceneRender/20250322/'

DATA_ROOT = '/home/wangzehan/Objaverse/clitest/glbs/'

meta_path = '/home2/LayoutGen/gen_output/gpt-4o_layout_in1k.json'
# meta = pd.read_csv(meta_path)
with open(meta_path) as f:
    layouts = json.load(f)
    layouts = layouts['layouts']
    print(len(layouts))


def create_chessboard_floor(size=64, scale=(1, 1, 1)):
    """创建带有棋盘格材质的可渲染地板"""
    # 添加平面
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "ChessboardFloor"

    # 创建材质
    mat = bpy.data.materials.new(name="ChessboardMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 清空默认节点
    for node in nodes:
        nodes.remove(node)

    # 添加节点
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    tex_node = nodes.new(type='ShaderNodeTexChecker')
    mapping_node = nodes.new(type='ShaderNodeMapping')
    coord_node = nodes.new(type='ShaderNodeTexCoord')

    # 连接节点
    links.new(coord_node.outputs['Object'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # 调整缩放
    mapping_node.inputs['Scale'].default_value = scale

    # 设置 Checker Texture 的颜色
    tex_node.inputs['Color1'].default_value = (1, 1, 1, 1)  # 白色
    tex_node.inputs['Color2'].default_value = (0.2, 0.2, 0.2, 1)  # 黑色

    # 赋予材质
    floor.data.materials.append(mat)

# white_output = None
white_output = None
def set_default_scene():
    global white_output
    # bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    # # small samples for fast rendering
    # bpy.context.scene.cycles.samples = 4
    # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    # bpy.context.scene.cycles.device = 'GPU'

    # # 检查是否有可用的 OptiX GPU
    # if bpy.context.scene.cycles.use_adaptive_sampling:
    #     bpy.context.scene.cycles.use_optix = True
    #     print("using OptiX")
    # else:
    #     print("No OptiX")

    # 可选：设置其他与 OptiX 相关的参数
    # bpy.context.scene.cycles.optix.use_denoising = True  # 启用 OptiX 降噪
    # bpy.context.scene.cycles.optix.denoising_radius = 2.0  # 设置降噪半径
    
    # for scene in bpy.data.scenes:
    #     scene.cycles.device = 'GPU'
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
        
    bpy.context.scene.eevee.taa_render_samples = 16
    bpy.context.scene.eevee.use_soft_shadows = True

    # get_devices() to let Blender detects GPU device
    # GPU_RANK = [0,1,2,3]
    # GPU_COUNT = 0
    # bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    #     if 'NVIDIA' in d['name']:
    #     # if d.type == 'OPTIX':
    #         d["use"] = True # Using all devices, include GPU and CPU
    #         print('Using', d["name"], d["use"])
    #         # if GPU_COUNT not in GPU_RANK:
    #         #     d["use"] = 1 # Using all devices, include GPU and CPU
    #         #     print(d["name"], d["use"])
    #         # else:
    #         #     d["use"] = 0 # Using all devices, include GPU and CPU
    #         #     print(d["name"], d["use"])
    #         # GPU_COUNT += 1
    #     else:
    #         d["use"] = False # Using all devices, include GPU and CPU
    #         print('Not Using', d["name"], d["use"])

    
    # render_prefs = bpy.context.preferences.addons['cycles'].preferences
    # render_device_type = render_prefs.compute_device_type
    # compute_device_type = render_prefs.devices[0].type if len(render_prefs.devices) > 0 else None
    # print(render_device_type, compute_device_type)
    # Check if the compute device type is GPU
    # if render_device_type == 'CUDA' and compute_device_type == 'CUDA':
    #     # GPU is being used for rendering
    #     print("Using GPU for rendering")
    # else:
    #     # GPU is not being used for rendering
    #     print("Not using GPU for rendering")
    
    # prepare the scene
    quality = 100
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.quality = quality
    # bpy.context.scene.render.use_file_extension = False
    
    # Create lights
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    three_point_lighting(10)
    
    bpy.context.view_layer.use_pass_z = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # 创建渲染层节点
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    map_value = tree.nodes.new(type="CompositorNodeMapValue")
    map_value.size = [0.2]  # 可调整的缩放参数，根据场景深度范围来设置
    map_value.use_min = True
    map_value.min = [0.0]
    map_value.use_max = True
    map_value.max = [1.0]

    # 添加一个混合节点
    alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
    alpha_over.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)  # 纯白背景

    # 文件输出节点
    # dark_output  = tree.nodes.new('CompositorNodeOutputFile')
    white_output = tree.nodes.new('CompositorNodeOutputFile')

    # 连接节点
    tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
    tree.links.new(alpha_over.outputs[0], white_output.inputs[0])
    # tree.links.new(render_layers.outputs['Image'], dark_output.inputs[0])

    # bpy.context.scene.use_nodes = True
    # tree = bpy.context.scene.node_tree
    # links = tree.links

device = 'cuda:0'

def create_light(name, light_type, energy, location, rotation):
    bpy.ops.object.light_add(type=light_type, align='WORLD', location=location, scale=(1, 1, 1))
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.rotation_euler = rotation
    return light

def three_point_lighting(distance):
    
    # Key ligh
    key_light = create_light(
        name="KeyLight",
        light_type='AREA',
        energy=1000,
        location=(4* distance, -4* distance, 4* distance),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    key_light.data.size = 2

    # Fill light
    fill_light = create_light(
        name="FillLight",
        light_type='AREA',
        energy=300,
        location=(-4* distance, -4* distance, 2* distance),
        rotation=(math.radians(45), 0, math.radians(135))
    )
    fill_light.data.size = 2

    # Rim/Back light
    rim_light = create_light(
        name="RimLight",
        light_type='AREA',
        energy=600,
        location=(0* distance, 4* distance, 0* distance),
        rotation=(math.radians(45), 0, math.radians(225))
    )
    rim_light.data.size = 2

def set_camera(camera, location, look_at, up):
    camera.location = location
    direction = (look_at - location).normalized()
    right = up.cross(direction).normalized()
    
    # 计算新的上轴（通过重新计算正交矩阵的 up 确保它与方向正交）
    corrected_up = direction.cross(right).normalized()
    
    rot_matrix = mathutils.Matrix((
        right,
        corrected_up,
        -direction  # 相机的-Z 轴应指向 look_at
    )).transposed()  # Blender 中的矩阵是列主序的，需要转置

    # 设置相机的旋转（从旋转矩阵转换为欧拉角）
    camera.rotation_euler = rot_matrix.to_euler('XYZ')

# uid_metadata = '/home2/LayoutGen/cocoimnt3d_filter80k_uid_map.json'
uid_metadata = '/home2/LayoutGen/in1k_filter80k_uid_map.json'
uid_sampler = UIDSampler(uid_metadata)

def create_scene_meta(layout, save_path=None):
    for obj in layout:
        obj["x_center"] = obj["left"] + obj["width"] / 2
        obj["y_center"] = obj["top"] + obj["height"] / 2
        obj['uid']   = uid_sampler.get_obj_uid(obj['category'])
        obj['scale'] = obj['width']
    
    x_min = min(obj["left"] for obj in layout)
    x_max = max(obj["left"] + obj["width"] for obj in layout)
    y_min = min(obj["top"] for obj in layout)
    y_max = max(obj["top"] + obj["height"] for obj in layout)
    x_center_scene = (x_max + x_min) / 2
    y_center_scene = (y_max + y_min) / 2

    for obj in layout:
        obj["x_norm"] = (obj["x_center"] - x_center_scene)
        obj["y_norm"] = (obj["y_center"] - y_center_scene)
    
    if save_path != None:
        with open(os.path.join(save_path, 'scene_meta.json'), 'w') as file:
            json.dump({'scene_meta':layout}, file)
    
    return layout, max(x_max - x_min, y_max - y_min)

def get_mesh_bbox(objects):
    all_coords = [mathutils.Vector(obj.matrix_world @ mathutils.Vector(corner)) for obj in objects for corner in obj.bound_box]
    
    min_coords = [min(coord[i] for coord in all_coords) for i in range(3)]
    max_coords = [max(coord[i] for coord in all_coords) for i in range(3)]
    return np.array([min_coords, max_coords])

# 归一化所有Mesh对象
def get_max_dim_center_minz(objects):
    """计算集合对象的最大尺寸、包围盒中心位置和 Z 方向的最小值"""
    all_coords = [mathutils.Vector(obj.matrix_world @ mathutils.Vector(corner)) for obj in objects for corner in obj.bound_box]
    
    min_coords = [min(coord[i] for coord in all_coords) for i in range(3)]
    max_coords = [max(coord[i] for coord in all_coords) for i in range(3)]

    max_dim = max(max_coords[i] - min_coords[i] for i in range(3))
    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
    min_z = min_coords[2]
    
    print(f"Max dimension: {max_dim}, Center: {center}, Min Z: {min_z}")
    
    return max_dim, center, min_z
    # for obj in objects:
    #     obj.scale = (1 / max_dim, 1 / max_dim, 1 / max_dim)
        # bpy.context.view_layer.objects.active = obj
        # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        # obj.location = (0, 0, 0)

def import_and_transform_glb(file_path, location, scale, rotation_degree=None, save_matrix_path=None):
    """导入 GLB 文件并对其根节点进行归一化、定位、旋转、缩放。"""
    # 导入 GLB 文件
    bpy.ops.import_scene.gltf(filepath=file_path)
    # 假设最后导入的集合是我们需要的
    imported_objects = bpy.context.selected_objects
    root_obj = imported_objects[0].parent if imported_objects[0].parent else imported_objects[0]
    
    # 归一化根节点
    max_dim, center, min_z = get_max_dim_center_minz(imported_objects)
    # for bvalue in root_obj.bound_box:
    #     print(tuple(bvalue))
    # 上面的输出都是0
    
    root_obj.scale = (root_obj.scale[0]*scale / max_dim, root_obj.scale[1]*scale / max_dim, root_obj.scale[2]*scale / max_dim)
    # root_obj.location = location
    root_obj.location = (location[0], location[1], location[2] - min_z*root_obj.scale[2])
    # root_obj.location = (-center[0],-center[1], -center[2])
    # root_obj.location = (0,0,0)
    # # 设置旋转（如果提供），绕 Z 轴从上往下看为顺时针旋转
    if rotation_degree is not None:
        root_obj.rotation_mode = 'XYZ'
        root_obj.rotation_euler[2] = math.radians(-rotation_degree)
    bpy.context.view_layer.update()
    bbox = get_mesh_bbox(imported_objects)
    
    if save_matrix_path:
        matrix_bbox = {
            'matrix': np.array(root_obj.matrix_world),
            'bbox'  : bbox,
        }
        np.save(save_matrix_path, matrix_bbox)
        print(f"Pose matrix and bbox saved to {save_matrix_path}")


RANDOM_START = 0
RANDOM_END   = 5
def render_single_scene_index(index):
    print('begin*************')
    layout = layouts[index]['layout']
    save_path = os.path.join(save_root, f'{index//100}/{index}')
    os.makedirs(os.path.join(save_root, f'{index//100}/{index}'), exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    print(save_path)
    
    random_path    = os.path.join(save_path, f'random_white{RANDOM_END-1}.jpg')
    random_exist   = True if os.path.exists(random_path) else False
    if random_exist:
        return 0
    
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    # create floor
    create_chessboard_floor()
    
    scene_meta, norm_distance = create_scene_meta(layout=layout, save_path=save_path)
    norm_distance = norm_distance*1.2
    print('set object*************')
    for obj_detail in scene_meta[:2]:
        uid = obj_detail['uid']
        obj_path = os.path.join(DATA_ROOT, '{0}.glb'.format(uid))
        save_matrix_path = os.path.join(save_path, '{0}.npy'.format(uid.replace('/','_')))
        import_and_transform_glb(obj_path, 
                                 location=(obj_detail['x_norm'], 
                                           obj_detail['y_norm'], 
                                           0.), 
                                 scale=obj_detail['scale'], 
                                 rotation_degree=obj_detail['orientation'],
                                 save_matrix_path = save_matrix_path)
    
    print('normalize and center obj over*************')

    # ['Image', 'Alpha', 'Depth', 'Noisy Image', 'Normal', 
    # 'UV', 'Vector', 'Position', 'Deprecated', 'Deprecated', 
    # 'Shadow', 'AO', 'Deprecated', 'Deprecated', 'Deprecated', 
    # 'IndexOB', 'IndexMA', 'Mist', 'Emit', 'Env', 
    # 'DiffDir', 'DiffInd', 'DiffCol', 
    # 'GlossDir', 'GlossInd', 'GlossCol', 
    # 'TransDir', 'TransInd', 'TransCol', 
    # 'SubsurfaceDir', 'SubsurfaceInd']
    
    camera = bpy.context.scene.camera

    # change to white background to render the final 4 views
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bpy.context.scene.render.film_transparent = True

    for camera_opt in range(5):
        # use transparent background to adjust camera distance
        # camera.location = Vector((norm_distance, - norm_distance, distance * elevation_factor * ratio))
        if camera_opt == 0:
            location = Vector((norm_distance, 0., 0.))
            up       = Vector((0., 0., 1.))
        elif camera_opt == 1:
            location = Vector((-1 * norm_distance, 0., 0.))
            up       = Vector((0., 0., 1.))
        elif camera_opt == 2:
            location = Vector((0., norm_distance, 0.))
            up       = Vector((0., 0., 1.))
        elif camera_opt == 3:
            location = Vector((0., -1 * norm_distance, 0.))
            up       = Vector((0., 0., 1.))
        elif camera_opt == 4:
            location = Vector((0., 0., norm_distance))
            up       = Vector((1., 0., 0.))
        look_at = Vector((0., 0., 0.))
        # set_camera(camera=camera, location=location, look_at=look_at, up=up)
        
        # Make the camera point at the bounding box center
        camera.location = location
        direction = (look_at - camera.location).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()

        camera.data.clip_start = 0.1
        camera.data.clip_end = max(1000, norm_distance)
        
        print('distance: ', norm_distance)
        print('camera.location: ', camera.location)

        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024
        
        for output_node in [white_output]:
            output_node.base_path = ''
        dark_path = os.path.join(save_path, f'cano_dark{camera_opt}.jpg')
        bpy.context.scene.render.filepath = dark_path
        # dark_path = os.path.join(save_root, obj_id, f'dark#')
        # dark_output.file_slots[0].path = dark_path
        
        white_path = os.path.join(save_path, f'cano_white{camera_opt}_#.jpg')
        # white_path = os.path.join(save_root, obj_id, f'white#')
        white_output.file_slots[0].path = white_path
        
        if os.path.exists(dark_path):
            return 0
        
        bpy.ops.render.render(write_still=True)
        print('RENAME:', white_path[:-6] + '_1.jpg', white_path[:-6] + '.jpg')
        os.rename(white_path[:-6] + '_1.jpg', white_path[:-6] + '.jpg')
        def get_3x4_RT_matrix_from_blender(cam):
            # Use matrix_world instead to account for all constraints
            location, rotation = cam.matrix_world.decompose()[0:2]
            R_world2bcam = rotation.to_matrix().transposed()

            # Use location from matrix_world to account for constraints:     
            T_world2bcam = -1*R_world2bcam @ location

            # put into 3x4 matrix
            RT = Matrix((
                R_world2bcam[0][:] + (T_world2bcam[0],),
                R_world2bcam[1][:] + (T_world2bcam[1],),
                R_world2bcam[2][:] + (T_world2bcam[2],)
                ))
            return RT

        if camera_opt>=0:
            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_path = os.path.join(save_path, f'cano_rt{camera_opt}.npy')
            # RT_path = os.path.join(save_root, 'Cap3D_imgs', 'Cap3D_imgs_view%d_CamMatrix'%camera_opt, '%s_%d.npy'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            if os.path.exists(RT_path):
                return
            np.save(RT_path, RT)

    # return 0
    for camera_opt in range(RANDOM_START, RANDOM_END):
        # 相机和原点连线与 xz 平面的夹角（俯仰角）在 -30° 到 30° 之间
        if camera_opt < 40:
            # pitch_angle = math.radians(random.uniform(-30, 30))
            pitch_angle = math.radians(random.uniform(5, 30))
        else:
            # pitch_angle = math.radians(
            #     random.choice([
            #         random.uniform(-89, -30),
            #         random.uniform(30, 89)
            #     ])
            # )
            pitch_angle = math.radians(random.uniform(30, 85))
        
        # 相机在 xz 平面的单位圆上随机选取一个角度（方位角）
        azimuth_angle = random.uniform(0, 2 * math.pi)
        # 单位圆的半径，稍微偏离 1（单位圆附近）
        radius = random.uniform(0.9, 1.2)  * norm_distance
        # 根据方位角和半径计算相机的 x, z 坐标
        cam_x = radius * math.cos(azimuth_angle) * math.cos(pitch_angle)
        cam_y = radius * math.sin(azimuth_angle) * math.cos(pitch_angle)
        # y 轴高度根据俯仰角计算
        cam_z = radius * math.sin(pitch_angle)

        location = Vector((cam_x, cam_y, cam_z))
        up       = Vector((0., 0., 1.))
        look_at  = Vector((0., 0., 0.))
        # set_camera(camera=camera, location=location, look_at=look_at, up=up)
        camera.location = location
        # # Make the camera point at the bounding box center
        direction = (look_at - camera.location).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()

        camera.data.clip_start = 0.1
        camera.data.clip_end = max(1000, norm_distance)

        print('distance: ', norm_distance)
        print('camera.location: ', camera.location)

        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024
        # print('resolution: 512*512')
        for output_node in [white_output]:
            output_node.base_path = ''
        # dark_path = os.path.join(save_root, '{0}/{1}'.format(meta['part'][index], meta['uid'][index]), f'random_dark{camera_opt}.jpg')
        dark_path = os.path.join(save_path, f'random_dark{camera_opt}.jpg')
        bpy.context.scene.render.filepath = dark_path
        # dark_path = os.path.join(save_root, obj_id, f'dark#')
        # dark_output.file_slots[0].path = dark_path
        # print(f'dark_path: {dark_path}')
        
        # white_path = os.path.join(save_root, '{0}/{1}'.format(meta['part'][index], meta['uid'][index]), f'random_white{camera_opt}_#.jpg')
        white_path = os.path.join(save_path, f'random_white{camera_opt}_#.jpg')
        # white_path = os.path.join(save_root, obj_id, f'white#')
        white_output.file_slots[0].path = white_path
        # print(f'white_path: {white_path}')
        if os.path.exists(dark_path):
            return 0
        
        bpy.ops.render.render(write_still=True)
        print('RENAME:', white_path[:-6] + '_1.jpg', white_path[:-6] + '.jpg')
        os.rename(white_path[:-6] + '_1.jpg', white_path[:-6] + '.jpg')
        def get_3x4_RT_matrix_from_blender(cam):
            # Use matrix_world instead to account for all constraints
            location, rotation = cam.matrix_world.decompose()[0:2]
            R_world2bcam = rotation.to_matrix().transposed()

            # Use location from matrix_world to account for constraints:     
            T_world2bcam = -1*R_world2bcam @ location

            # put into 3x4 matrix
            RT = Matrix((
                R_world2bcam[0][:] + (T_world2bcam[0],),
                R_world2bcam[1][:] + (T_world2bcam[1],),
                R_world2bcam[2][:] + (T_world2bcam[2],)
                ))
            return RT

        if camera_opt>=0:
            RT = get_3x4_RT_matrix_from_blender(camera)
            # RT_path = os.path.join(save_root, '{0}/{1}'.format(meta['part'][index], meta['uid'][index]), f'random_rt{camera_opt}.npy')
            RT_path = os.path.join(save_path, f'random_rt{camera_opt}.npy')
            # RT_path = os.path.join(save_root, 'Cap3D_imgs', 'Cap3D_imgs_view%d_CamMatrix'%camera_opt, '%s_%d.npy'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            if os.path.exists(RT_path):
                return
            np.save(RT_path, RT)
    
    return 0

render_failed = []
failed_reason = []
# for i in tqdm(range(len(meta))):
# for i in tqdm(range(10)):
    # obj_id = meta['id'][i]
    # try:
    #     render_single_obj(obj_id=obj_id)
        # render_single_obj('00781a2bfbcc426b8db901d37409461b')
    # except:
    #     print(f'failed:{obj_id}')
    #     render_failed.append(obj_id)
# pd.DataFrame(render_failed).to_json(os.path.join(save_root, 'failed.json'))
print(layouts[:3])

start = len(layouts) * worker//8
end   = len(layouts) * (worker+1)//8
# start = 0
# end   = 1

print(start, end)
set_default_scene()
for index in range(start, end):
# for index in range(10):
    # uid = '0a1b1bd42cf24f2085ab9e0642f50293'
    if index % 10 == 0:
        bpy.ops.wm.read_homefile(use_empty=False)
        set_default_scene()
    # try:
    ret = render_single_scene_index(index)
    # except:
    #     print('failed:{0}/{1}'.format(meta['part'][index], meta['uid'][index]))
    #     render_failed.append('{0}/{1}'.format(meta['part'][index], meta['uid'][index]))
    #     failed_reason.append('render error')
    #     continue
    # if ret != 0:
    #     print('failed:{0}/{1}'.format(meta['part'][index], meta['uid'][index]))
    #     render_failed.append('{0}/{1}'.format(meta['part'][index], meta['uid'][index]))
    #     if ret == 1:
    #         failed_reason.append('too big')
    #     elif ret == 2:
    #         failed_reason.append('glb not find')
    #     else:
    #         failed_reason.append('other error')
    
pd.DataFrame({'uid':render_failed, 'failed_reason':failed_reason}).to_csv(os.path.join(save_root, f'failed_{worker}.csv'), index=False)
print(os.path.join(save_root, f'random_failed_{worker}.csv'))
bpy.ops.wm.quit_blender()


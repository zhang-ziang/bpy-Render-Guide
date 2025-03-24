# bpy-Render-Guide

## EEVEE渲染：
EEVEE渲染是blender中一个较为基础的渲染模式，它需要指定渲染的屏幕，而一般的Linux服务器中并没有物理的屏幕输出，因此需要开启虚拟屏幕进行渲染。为了充分利用GPU资源，给每一个GPU配置一个虚拟屏幕。多个渲染可以指定在同一个虚拟屏幕上。

### 虚拟屏幕配置

找到GPU的busID总线ID：
```shell
nvidia-xconfig --query-gpu-info
```
大概会看到如下输出：
```shell
GPU #0:
  Name      : NVIDIA GeForce RTX 2080 Ti
  UUID      : GPU-a1da6f54-6190-83dc-8f94-09b39a9dad86
  PCI BusID : PCI:59:0:0

  Number of Display Devices: 0
```

虚拟屏幕设置`xorg.conf`,修改：

```yaml
# nvidia-xconfig: X configuration file generated by nvidia-xconfig
# nvidia-xconfig:  version 535.183.01

Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0" 0 0
    # Screen      1  "Screen1" 0 1
    # Screen      2  "Screen2" 0 2
    # Screen      3  "Screen3" 0 3
    InputDevice    "Keyboard0" "CoreKeyboard"
    InputDevice    "Mouse0" "CorePointer"
EndSection

Section "Files"
EndSection

Section "ServerFlags"
    Option "AutoAddGPU" "false"
EndSection

Section "InputDevice"
    Identifier     "Mouse0"
    Driver         "mouse"
    Option         "Protocol" "auto"
    Option         "Device" "/dev/psaux"
    Option         "Emulate3Buttons" "no"
    Option         "ZAxisMapping" "4 5"
EndSection

Section "InputDevice"
    Identifier     "Keyboard0"
    Driver         "kbd"
EndSection

Section "Monitor"
    Identifier     "Monitor0"
    VendorName     "Unknown"
    ModelName      "Unknown"
    Option         "DPMS"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
# 修改总线ID
    BusID          "PCI:59:0:0"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    Monitor        "Monitor0"
    DefaultDepth   24
    Option         "UseDisplayDevice" "none"
    Option         "MultiGPU" "false"
    Option         "ProbeAllGpus" "false"
    SubSection     "Display"
        Virtual     1280 1024
        Depth       24
    EndSubSection
EndSection
```
使用xserver开启虚拟屏幕
```shell
# 开四个虚拟屏幕，从conf中指定到四个屏幕编号上
X -config xorg_confs/xorg0.conf :0 &X -config xorg_confs/xorg1.conf :1 & X -config xorg_confs/xorg2.conf :2 & X -config xorg_confs/xorg3.conf :3
```

### EEVEE设置

需要初始化的配置如下：
```python

def set_default_scene():
    global white_output
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # 高版本的Blender需要指定NEXT版本的EEVEE
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
        
    bpy.context.scene.eevee.taa_render_samples = 16
    bpy.context.scene.eevee.use_soft_shadows = True

    # 设置输出格式和图像质量
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

    # 以下节点操作为了同时输出黑色背景和白色背景
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


```


渲染前将3D物体进行归一化，将3D包围盒放缩到单位立方体，并将物体中心移动至坐标轴原点。
需要指定导入的物体的上一层节点select，进行变换。而3D包围盒需要从下层的物体中通过世界坐标系变换矩阵变换到世界坐标系中进行计算。

```python
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
    # 原点居中但是对齐底部到z=0平面
    root_obj.location = (location[0], location[1], location[2] - min_z*root_obj.scale[2])
    # 原点居中设置
    # root_obj.location = (-center[0]*scale / max_dim,-center[1]*scale / max_dim, -center[2]*scale / max_dim)
    
    # 设置旋转（如果提供），绕 Z 轴从上往下看为顺时针旋转
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

```

启动渲染：
```shell
# 激活python环境，替换掉blender默认的python环境
conda activate render

# 指定屏幕和python脚本
# ./blender 是blender程序
DISPLAY=:0 ./blender -b -P sceneRender.py -- --worker 0

```


## CYCLES渲染
cycles是blender中的默认光线追踪渲染管线，不需要指定虚拟屏幕可以使用CUDA进行渲染加速。
需要初始化的配置如下：
```python
def set_default_scene():
    global white_output
    bpy.context.scene.render.engine = 'CYCLES'
    # small samples for fast rendering
    bpy.context.scene.cycles.samples = 4
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()


    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if 'NVIDIA' in d['name']:
            d["use"] = True # Using all devices, include GPU and CPU
            print('Using', d["name"], d["use"])

        else:
            d["use"] = False # Using all devices, include GPU and CPU
            print('Not Using', d["name"], d["use"])

    render_prefs = bpy.context.preferences.addons['cycles'].preferences
    render_device_type = render_prefs.compute_device_type
    compute_device_type = render_prefs.devices[0].type if len(render_prefs.devices) > 0 else None
    print(render_device_type, compute_device_type)
    Check if the compute device type is GPU
    if render_device_type == 'CUDA' and compute_device_type == 'CUDA':
        # GPU is being used for rendering
        print("Using GPU for rendering")
    else:
        # GPU is not being used for rendering
        print("Not using GPU for rendering")
    
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


```
启动渲染：
```shell
# 激活python环境，替换掉blender默认的python环境
conda activate render

# CYCLES无须指定屏幕
# ./blender 是blender程序
./blender -b -P sceneRender.py -- --worker 0

```
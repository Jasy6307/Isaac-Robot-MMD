"""Render an evaluated Isaac mesh with the existing MMD Tools Append Toon shader."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

ROOT = PROJECT_ROOT
sys.path.insert(0, str(SOURCE_DIR))
from convert_pmx import read_pmx

OUT = ROOT/'output/blender-toon-reference'
meta = json.loads((OUT/'matched_pose.reference.json').read_text())
data = np.load(OUT/'matched_pose.reference.npz')
model = read_pmx(Path(meta['pmx']))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
library = OUT/'blender_mmd_tools_append/mmd_tools_append/blends/MMDAppend_Materials.blend'
with bpy.data.libraries.load(str(library), link=False) as (source, target):
    target.node_groups = ['Toon Shader Texture']
toon = bpy.data.node_groups['Toon Shader Texture']
# Tune the bundled shader's light threshold to this scene's illumination.
toon.nodes['Math'].inputs[1].default_value = .45

mesh = bpy.data.meshes.new('Exact Isaac evaluated mesh frame 60')
mesh.from_pydata(data['points'].tolist(), [], data['faces'].tolist())
mesh.update()
obj = bpy.data.objects.new('Anon matched pose', mesh)
bpy.context.collection.objects.link(obj)
uv = mesh.uv_layers.new(name='UVMap')
for loop in mesh.loops:
    uv.data[loop.index].uv = data['uv'][loop.vertex_index]
for poly in mesh.polygons:
    poly.use_smooth = True
mesh.normals_split_custom_set_from_vertices(data['normals'].tolist())

offset = 0
for item in model['materials']:
    mat = bpy.data.materials.new(item['name'])
    mat.use_nodes = True
    mat.surface_render_method = 'DITHERED'
    mat.use_backface_culling = False
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    group = nodes.new('ShaderNodeGroup')
    group.node_tree = toon
    group.inputs['Base Color'].default_value = item['diffuse']
    group.inputs['Roughness'].default_value = 1.
    group.inputs['Highlight Color'].default_value = (.02, .02, .02, 1.)
    group.inputs['Shadow Color'].default_value = (.62, .65, .72, 1.)
    group.inputs['Rim Light Strength'].default_value = .1
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = .55
    links.new(group.outputs['Color'], emission.inputs['Color'])
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = item['diffuse'][3]
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])
    idx = item['texture']
    if 0 <= idx < len(model['textures']):
        path = Path(meta['pmx']).parent/model['textures'][idx].replace('\\', '/')
        if path.is_file():
            tex = nodes.new('ShaderNodeTexImage')
            tex.image = bpy.data.images.load(str(path), check_existing=True)
            tex.image.colorspace_settings.name = 'sRGB'
            links.new(tex.outputs['Color'], group.inputs['Base Color'])
            links.new(tex.outputs['Alpha'], mix.inputs[0])
    obj.data.materials.append(mat)
    count = item['count']//3
    for poly in mesh.polygons[offset:offset+count]:
        poly.material_index = len(obj.data.materials)-1
    offset += count

scene = bpy.context.scene
# The stock Toon Shader Texture uses Shader to RGB and therefore requires EEVEE.
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x, scene.render.resolution_y = meta['width'], meta['height']
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.96,.98,1.,1.)
scene.world.node_tree.nodes['Background'].inputs[1].default_value = .55

def sun(name, power, color, angles, angle):
    import math
    light = bpy.data.lights.new(name, 'SUN')
    light.energy, light.color, light.angle = power, color, math.radians(angle)
    inst = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(inst)
    inst.rotation_euler = [math.radians(a) for a in angles]
sun('Key', .7, (1.,.94,.89), (25,-25,-35), 25)
sun('Fill', .18, (.94,.97,1.), (55,25,140), 25)
light = bpy.data.lights.new('Face softbox', 'AREA')
light.energy, light.shape, light.size = 40, 'DISK', 2.8
light.color = (1.,.97,.94)
inst = bpy.data.objects.new('Face softbox', light)
bpy.context.collection.objects.link(inst)
inst.location = (meta['target'][0], meta['target'][1]-2.5, 1.7)
inst.rotation_euler = (Vector((meta['target'][0],meta['target'][1],1.4))-inst.location).to_track_quat('-Z','Y').to_euler()

# Recreate the original ground layout; its shader is native to Blender.
bpy.ops.mesh.primitive_plane_add(size=24, location=(0,0,-.025))
ground = bpy.context.object
ground.name = 'Original visible floor layout'
floor_mat = bpy.data.materials.new('Original floor neutral PBR')
floor_mat.use_nodes = True
bsdf = floor_mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (.18,.21,.26,1.)
bsdf.inputs['Roughness'].default_value = .5
ground.data.materials.append(floor_mat)

camera = bpy.data.cameras.new('Matched Isaac Camera')
camera.lens = meta['focal_length']
camera.sensor_fit = 'HORIZONTAL'
camera.sensor_width = meta['horizontal_aperture']
camera.clip_start = .01
inst = bpy.data.objects.new('Matched Isaac Camera', camera)
bpy.context.collection.objects.link(inst)
inst.location = meta['eye']
inst.rotation_euler = (Vector(meta['target'])-inst.location).to_track_quat('-Z','Y').to_euler()
scene.camera = inst
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(OUT/'blender_mmd_toon.png')
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'anon_matched_toon.blend'))
bpy.ops.render.render(write_still=True)
(OUT/'blender_reference.json').write_text(json.dumps(dict(
    shader='MMD Tools Append: Toon Shader Texture; highlight .02, shadow (.62,.65,.72), rim .1',
    source='https://github.com/MMD-Blender/blender_mmd_tools_append',
    frame=meta['frame'], matched_geometry=True, matched_camera=True,
    engine='EEVEE', notes='Lighting powers adapted between renderers; native ground shading, no Isaac grid overlay.'
    ,shader_light_threshold=.45,shader_output_strength=.55,outline=False
),indent=2),encoding='utf-8')
print('BLENDER_TOON_REFERENCE_PASS',flush=True)

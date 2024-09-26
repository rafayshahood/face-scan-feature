


# import bpy

# # Path to your GLTF model and JPG file
# # Replace with the absolute path to your files
# gltf_path = "/Users/rafayshahood/Desktop/face_scan/HEADS/FEMALE HEAD/Responce.gltf"
# jpg_texture_path = "/Users/rafayshahood/Desktop/face_scan/HEADS//FEMALE HEAD/Responce.jpg"


# # Clear existing scene data
# bpy.ops.wm.read_factory_settings(use_empty=True)

# # Import the GLTF file
# bpy.ops.import_scene.gltf(filepath=gltf_path)

# # Select the imported object (assuming it's the first object in the scene)
# obj = bpy.context.selected_objects[0]
# bpy.context.view_layer.objects.active = obj

# # Create a new material
# material = bpy.data.materials.new(name="ImportedMaterial")
# material.use_nodes = True

# # Get the material node tree
# nodes = material.node_tree.nodes

# # Remove the default node
# nodes.clear()

# # Add the Principled BSDF shader node
# bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

# # Add the material output node
# material_output = nodes.new(type="ShaderNodeOutputMaterial")

# # Link BSDF to Material Output
# material.node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

# # Add the Image Texture node
# tex_image = nodes.new(type="ShaderNodeTexImage")

# # Load the image texture
# image = bpy.data.images.load(jpg_texture_path)
# tex_image.image = image

# # Link the Image Texture node to the Base Color input of the BSDF shader
# material.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])

# # Assign the material to the object
# if obj.data.materials:
#     # If the object already has materials, replace the first one
#     obj.data.materials[0] = material
# else:
#     # Otherwise, append a new material
#     obj.data.materials.append(material)

# # Update the scene
# bpy.context.view_layer.update()

# print("GLTF model imported and texture applied successfully.")

import trimesh
import pyrender
import numpy as np
from PIL import Image

# Load the GLTF model using trimesh
gltf_path = "./HEADS/FEMALE HEAD/Responce.gltf"
jpg_texture_path = "./HEADS/FEMALE HEAD/Responce.jpg"
scene = trimesh.load(gltf_path)

# Load texture image using PIL
texture_image = Image.open(jpg_texture_path)
texture_data = np.array(texture_image).astype(np.float32) / 255.0

# Create a pyrender scene
render_scene = pyrender.Scene()

# Iterate over all geometries in the trimesh scene
for name, mesh in scene.geometry.items():
    # Ensure the mesh has UV mapping
    if hasattr(mesh.visual, 'uv'):
        # Manually create a material and apply the texture
        material = pyrender.MetallicRoughnessMaterial(baseColorTexture=pyrender.texture.Texture(source=texture_data))
        
        # Convert the trimesh geometry to pyrender Mesh and apply the material
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        render_scene.add(pyrender_mesh)
    else:
        # Add the mesh without a texture if no UV mapping is present
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        render_scene.add(pyrender_mesh)

# Set up the camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
render_scene.add_node(camera_node)

# Set up lighting
light = pyrender.PointLight(color=np.ones(3), intensity=3.0)
light_node = pyrender.Node(light=light, translation=np.array([0, -1, 2]))
render_scene.add_node(light_node)

# Create the viewer and display the model
pyrender.Viewer(render_scene, use_raymond_lighting=True)
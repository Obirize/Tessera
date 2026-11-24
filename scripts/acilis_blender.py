"""Render a cat-symbol stone layout (from iterasyon_kedi_tasarim.py) in Blender / Cycles.

Still:      blender -b -P acilis_blender.py -- still <kedi_n.json> <out.png> [res] [samples]
Layers:     blender -b -P acilis_blender.py -- katman <kedi_n.json> <out_dir> [res] [samples]
Every stone is its own bevelled prism with a small tilt and height jitter, sitting on an ink base.
"""
import json
import math
import random
import sys

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MODE, SRC, OUT = argv[0], argv[1], argv[2]
RES = int(argv[3]) if len(argv) > 3 else 1024
SAMPLES = int(argv[4]) if len(argv) > 4 else 128
STYLE = argv[5] if len(argv) > 5 else "drop"
FPS = 30

data = json.load(open(SRC, encoding="utf-8"))
PAL = data["palette"]
METAL = set(data["metal"])
H0 = 0.062          # stone height
PLATE_T = 0.07
random.seed(7)

# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "CYCLES"
cy = scn.cycles
cy.samples = SAMPLES
cy.use_denoising = True
try:
    cy.denoiser = "OPENIMAGEDENOISE"
except TypeError:
    pass
cy.max_bounces = 8
cy.glossy_bounces = 4
cy.transparent_max_bounces = 8
prefs = bpy.context.preferences.addons["cycles"].preferences
dev = "CPU"
for kind in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = kind
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            dev = kind
            break
    except Exception:
        pass
cy.device = "GPU" if dev != "CPU" else "CPU"
print("DEVICE", dev)

scn.render.resolution_x = scn.render.resolution_y = RES
scn.render.film_transparent = True
scn.view_settings.view_transform = "Standard"
scn.view_settings.look = "None"
scn.view_settings.exposure = -1.0
scn.render.image_settings.file_format = "PNG"
scn.render.image_settings.color_mode = "RGBA"
scn.render.image_settings.color_depth = "8"

world = bpy.data.worlds.new("W")
scn.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
bg = wn["Background"]
grad = wn.new("ShaderNodeTexGradient")
grad.gradient_type = "LINEAR"
tc = wn.new("ShaderNodeTexCoord")
mp = wn.new("ShaderNodeMapping")
mp.inputs["Rotation"].default_value = (0, math.radians(-90), 0)   # gradient along Z: bright sky
ramp = wn.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.10, 0.09, 0.08, 1)
ramp.color_ramp.elements[1].color = (0.95, 0.90, 0.82, 1)
wl.new(tc.outputs["Generated"], mp.inputs["Vector"])
wl.new(mp.outputs["Vector"], grad.inputs["Vector"])
wl.new(grad.outputs["Fac"], ramp.inputs["Fac"])
wl.new(ramp.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 0.25


def lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


_mats = {}


def material(key, jitter=0.0, fade=False):
    """Per-stone material; jitter shifts value/hue slightly so neighbours never look identical."""
    rgb = PAL[key]
    j = round(jitter, 3)
    name = f"{key}_{j}_{int(fade)}"
    if name in _mats:
        return _mats[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    col = [min(1.0, max(0.0, lin(v) * (1 + j))) for v in rgb]
    p.inputs["Base Color"].default_value = (*col, 1)
    if key in METAL:
        p.inputs["Metallic"].default_value = 1.0
        p.inputs["Roughness"].default_value = 0.26 + random.random() * 0.08
    elif key in ("base",):
        p.inputs["Roughness"].default_value = 0.85
    else:
        p.inputs["Roughness"].default_value = 0.38
        p.inputs["Coat Weight"].default_value = 0.18
        p.inputs["Coat Roughness"].default_value = 0.12
        if key in ("blue", "amber", "blue2", "amber2"):
            p.inputs["Roughness"].default_value = 0.2
            p.inputs["Coat Weight"].default_value = 0.8
            p.inputs["Coat Roughness"].default_value = 0.04
    # very subtle hand-made surface
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 45.0
    tex.inputs["Detail"].default_value = 3.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025 if key not in METAL else 0.10
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    if key == "base":
        bump.inputs["Strength"].default_value = 0.25
    if fade:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "OBJECT"
        attr.attribute_name = "fade"
        nt.links.new(attr.outputs["Fac"], p.inputs["Alpha"])
        m.blend_method = "HASHED" if hasattr(m, "blend_method") else None
    _mats[name] = m
    return m


def prism(name, pts, z0, h, bevel, mat):
    bm = bmesh.new()
    vs_b = [bm.verts.new((x, y, z0)) for x, y in pts]
    vs_t = [bm.verts.new((x, y, z0 + h)) for x, y in pts]
    bm.faces.new(vs_t)
    bm.faces.new(vs_b[::-1])
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((vs_b[i], vs_b[j], vs_t[j], vs_t[i]))
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    scn.collection.objects.link(ob)
    me.materials.append(mat)
    if bevel > 0:
        bv = ob.modifiers.new("bevel", "BEVEL")
        bv.width = bevel
        bv.segments = 4
        bv.profile = 0.55
        bv.limit_method = "ANGLE"
        bv.angle_limit = math.radians(55)
        bv.use_clamp_overlap = True
        bv.harden_normals = True
    return ob


def inradius(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(max(xs) - min(xs), max(ys) - min(ys)) / 2


# ------------------------------------------------------------------ paper ground (animation only)
if MODE == "katman":
    scn.render.film_transparent = False
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -PLATE_T - 0.001))
    gp = bpy.context.object
    gm = bpy.data.materials.new("paper_ground")
    gm.use_nodes = True
    gpb = gm.node_tree.nodes["Principled BSDF"]
    gpb.inputs["Base Color"].default_value = (*[lin(v) for v in (226, 216, 198)], 1)
    gpb.inputs["Roughness"].default_value = 0.95
    gp.data.materials.append(gm)

# ------------------------------------------------------------------ plate + stones
plate = prism("plate", data["plate"], -PLATE_T, PLATE_T, 0.014, material(data["plate_col"]))

stones = []
for i, s in enumerate(data["stones"]):
    pts = s["pts"]
    cx, cy_ = s["cx"], s["cy"]
    loc = [(x - cx, y - cy_) for x, y in pts]
    r = inradius(pts)
    h = H0 * s["h"] * (1 + random.uniform(-0.10, 0.10))
    bev = max(0.004, min(0.024, r * 0.4))
    jit = random.uniform(-0.05, 0.05)
    ob = prism(f"s{i}", loc, 0.0, h, bev, material(s["col"], jit, fade=False))
    tilt = math.radians(2.4 if s["col"] not in METAL else 4.0)
    ob.location = (cx, cy_, 0.0)
    ob.rotation_euler = (random.uniform(-tilt, tilt), random.uniform(-tilt, tilt), random.uniform(-0.012, 0.012))
    ob["fade"] = 1.0
    stones.append((ob, s))

# ------------------------------------------------------------------ lights
def area(name, loc, size, energy, color=(1, 1, 1), shape="DISK"):
    L = bpy.data.lights.new(name, "AREA")
    L.shape = shape
    L.size = size
    L.energy = energy
    L.color = color
    o = bpy.data.objects.new(name, L)
    o.location = loc
    scn.collection.objects.link(o)
    d = Vector((0, 0, 0)) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return o


area("key", (-3.2, 2.6, 3.0), 1.6, 620, (1.0, 0.95, 0.88))
area("fill", (3.0, -2.2, 3.2), 5.0, 150, (0.88, 0.93, 1.0))
area("rim", (2.0, 3.6, 1.4), 1.5, 90, (1.0, 0.90, 0.80))
area("top", (0.0, -0.8, 5.0), 6.0, 160, (1.0, 0.98, 0.95))
area("bounce", (0.0, -3.2, 7.2), 9.0, 90, (1.0, 0.93, 0.80))   # broad soft source the gold can mirror

# ------------------------------------------------------------------ camera: frontal, slight angle
cam = bpy.data.cameras.new("cam")
cam.lens = 105
cam.type = "ORTHO"
cam.ortho_scale = 2.55
cam.sensor_width = 36
co = bpy.data.objects.new("cam", cam)
scn.collection.objects.link(co)
scn.camera = co
PITCH = 0.0
YAW = 0.0
DIST = 6.6
ROT = Euler((PITCH, YAW, 0.0))
dirv = ROT.to_matrix() @ Vector((0, 0, 1))
target = Vector((0, -0.01, -0.02))
co.location = target + dirv * DIST
co.rotation_euler = ROT


if MODE == "still":
    scn.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    print("STILL DONE", len(stones))
    sys.exit(0)

# ------------------------------------------------------------------ layers for the live opening card
# katman <kedi.json> <out_dir> [res] [samples]: the finished picture, the empty plate and where every stone is.
# tessera/acilis.py cuts the stones out of the finished picture and animates them live.
scn.cycles.samples = SAMPLES
for ob, _ in stones:
    ob["fade"] = 1.0
bpy.context.view_layer.update()
out = {"scale": cam.ortho_scale, "cx": target.x, "cy": target.y, "stones": []}
for ob, st in stones:
    n = len(st["pts"])
    ring = [ob.matrix_world @ v.co for v in list(ob.data.vertices)[n:2 * n]]
    out["stones"].append({"region": st["region"], "col": st["col"], "cx": st["cx"], "cy": st["cy"],
                          "pts": [[round(v.x, 4), round(v.y, 4)] for v in ring]})
json.dump(out, open(OUT.rstrip("/\\") + "/katman.json", "w"))
scn.render.filepath = OUT.rstrip("/\\") + "/tam.png"
bpy.ops.render.render(write_still=True)
for ob, _ in stones:
    ob.hide_render = True
scn.render.filepath = OUT.rstrip("/\\") + "/levha.png"
bpy.ops.render.render(write_still=True)
print("LAYERS DONE", len(stones))

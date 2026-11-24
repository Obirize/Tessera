"""Render a cat-symbol stone layout (from iterasyon_kedi_tasarim.py) in Blender / Cycles.

Still:      blender -b -P iterasyon_kedi_blender.py -- still <kedi_n.json> <out.png> [res] [samples]
Animation:  blender -b -P iterasyon_kedi_blender.py -- anim  <kedi_n.json> <out_dir> [res] [samples] [style]
            writes <out_dir>/f_0001.png ... ; style = drop | slide
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
if MODE == "anim":
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
    ob = prism(f"s{i}", loc, 0.0, h, bev, material(s["col"], jit, fade=(MODE == "anim")))
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


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_out_back(t, s=1.25):
    t -= 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def ease_in_out(t):
    return t * t * (3 - 2 * t)


if MODE == "still":
    scn.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    print("STILL DONE", len(stones))
    sys.exit(0)

# ------------------------------------------------------------------ animation
N_FR = int(round(2.2 * FPS))
HOLD = 8
FLIGHT = 13
scn.render.fps = FPS
scn.frame_start, scn.frame_end = 1, N_FR
scn.render.use_motion_blur = True
scn.render.motion_blur_shutter = 0.45
scn.render.filepath = OUT.rstrip("/\\") + "/f_"

last_land = N_FR - HOLD
start_window = last_land - FLIGHT     # stones start between frame 1 and this
# choreography: which regions arrive when (0 first .. 1 last); eyes always close the sequence
SEQ = {
    "3": {"bg": 0.0, "ruff": 0.30, "ear": 0.36, "ear_in": 0.42, "face": 0.50, "nose": 0.74,
          "eye": 0.90, "pupil": 0.92},
    "4": {"bust": 0.0, "ruff": 0.06, "ear": 0.10, "ear_in": 0.13, "face": 0.12, "nose": 0.28,
          "ring": 0.78, "eye": 0.92, "pupil": 0.94},
}.get(data["key"], {})


def region_order(s):
    r = s["region"]
    if r.startswith("gold") and r[4:].isdigit():
        return 0.30 + 0.075 * int(r[4:])
    return SEQ.get(r, s["order"])


RANK = {"pupil": 0, "eye": 0, "nose": 1, "face": 2, "ear_in": 3, "ear": 4, "ruff": 5, "bg": 6}
keyed = sorted(range(len(stones)), key=lambda i: (RANK.get(stones[i][1]["region"], 5),
               stones[i][1]["cy"] < -9, round(math.atan2(stones[i][1]["cx"], stones[i][1]["cy"]), 3)))
n_eye = sum(RANK.get(s["region"], 5) == 0 for _, s in stones)
orders = [0.0] * len(stones)
for pos, i in enumerate(keyed):        # the eyes land first and alone, the rest follows one by one
    orders[i] = pos * 0.35 if pos < n_eye else n_eye * 0.35 + 5.0 + (pos - n_eye)
lo, hi = min(orders), max(orders)
print("ORDER", [stones[i][1]["region"] for i in keyed])

for (ob, s), o in zip(stones, orders):
    t0 = 1 + (o - lo) / max(hi - lo, 1e-6) * (start_window - 1)
    base_loc = Vector(ob.location)
    base_rot = Euler(ob.rotation_euler)
    rr = math.hypot(s["cx"], s["cy"]) + 1e-6
    out = Vector((s["cx"] / rr, s["cy"] / rr, 0))
    spin = random.uniform(-0.45, 0.45)
    for f in range(1, N_FR + 1):
        t = (f - t0) / FLIGHT
        t = min(1.0, max(0.0, t))
        if s["col"] in METAL:   # gold tesserae flip over as they land and catch the light
            e = ease_out_cubic(t)
            ob.location = base_loc + Vector((0, 0, (1 - e) * 0.6)) + out * (1 - e) * 0.06
            tang = math.atan2(out.y, out.x)
            flip = (1 - e) * 2.2
            ob.rotation_euler = (base_rot.x + flip * -math.sin(tang), base_rot.y + flip * math.cos(tang),
                                 base_rot.z + (1 - e) * spin * 0.4)
            sc = 0.9 + 0.1 * e
        elif STYLE == "drop":
            e = ease_out_cubic(t)
            lift = (1 - e) * 0.9
            ob.location = base_loc + Vector((0, 0, lift)) + out * (1 - e) * 0.10
            ob.rotation_euler = (base_rot.x + (1 - e) * 0.35 * math.copysign(1, spin),
                                 base_rot.y + (1 - e) * spin * 0.6, base_rot.z + (1 - e) * spin)
            sc = 1.0 + 0.55 * (1 - e)
        else:  # slide: in from outside, rotating into place
            e = ease_out_back(t, 1.1) if t < 1 else 1.0
            ob.location = base_loc + out * (1 - e) * 0.55 + Vector((0, 0, (1 - ease_out_cubic(t)) * 0.25))
            ob.rotation_euler = (base_rot.x, base_rot.y, base_rot.z + (1 - e) * spin * 2.2)
            sc = 0.8 + 0.2 * ease_out_cubic(t)
        ob.scale = (sc, sc, sc)
        ob["fade"] = min(1.0, t / 0.28)
        ob.keyframe_insert("location", frame=f)
        ob.keyframe_insert("rotation_euler", frame=f)
        ob.keyframe_insert("scale", frame=f)
        ob.keyframe_insert('["fade"]', frame=f)
        ob.hide_render = t <= 0.0                 # not there yet: no stray shadow on the plate
        ob.keyframe_insert("hide_render", frame=f)
    for fc in ob.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

bpy.ops.render.render(animation=True)
print("ANIM DONE", len(stones))

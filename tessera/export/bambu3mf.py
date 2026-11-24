"""Bambu Studio project 3MF writer (multi-plate, multi-filament, multi-volume objects).

Layout follows what Bambu Studio 2.x itself writes (checked against files exported by its CLI):
  3D/3dmodel.model            build items + component objects (production extension)
  3D/Objects/object_<k>.model  meshes; one sub-object per volume
  Metadata/model_settings.config  objects, parts (volumes, extruder), plates (plater_id + instances)
  Metadata/project_settings.config  full printer/process/filament settings as JSON

Plates are laid out in world space the way Bambu does (PartPlateList::compute_origin):
stride = plate * 1.2, columns = ceil-ish(sqrt(n)), row r at y = -r * stride.
"""
from __future__ import annotations

import io
import json
import math
import os
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

PLATE_GAP_FACTOR = 1.2   # LOGICAL_PART_PLATE_GAP = 1/5
MAX_PLATES = 36
_APP = "BambuStudio-02.09.01.70"
_TEMPLATE = os.path.join(os.path.dirname(__file__), "data", "bambu_x1c_project.json")


@dataclass
class Volume:
    name: str
    vertices: np.ndarray        # (n, 3) float, object-local mm
    triangles: np.ndarray       # (m, 3) int
    extruder: int = 1           # 1-based filament slot


@dataclass
class Obj:
    name: str
    volumes: list[Volume]
    plate: int                  # 0-based plate index
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)   # plate-local translation (mm)


@dataclass
class FilamentSpec:
    name: str
    hex: str                    # "#RRGGBB"
    finish: str = "matte"


def column_count(n: int) -> int:
    """Bambu's PartPlateList::compute_colum_count."""
    v = math.sqrt(n)
    r = round(v)
    return int(r + 1 if v > r else r)


def plate_origin(i: int, n: int, plate_mm: float = 256.0) -> tuple[float, float]:
    cols = column_count(max(n, 1))
    stride = plate_mm * PLATE_GAP_FACTOR
    return (i % cols) * stride, -(i // cols) * stride


def _uuid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def project_settings(filaments: Sequence[FilamentSpec], wipe_tower_xy: Optional[list] = None,
                     n_plates: int = 1, prime_tower: Optional[bool] = None) -> dict:
    tpl = json.load(open(_TEMPLATE, encoding="utf-8"))
    s = dict(tpl["settings"])
    n = len(filaments)
    for k in tpl["per_filament_keys"]:
        vals = s[k]
        s[k] = [vals[1 if f.finish == "matte" else 0] for f in filaments]
    s["filament_colour"] = [f.hex.upper() for f in filaments]
    s["filament_self_index"] = [str(i + 1) for i in range(n)]
    s["filament_map"] = ["1"] * n
    s["filament_nozzle_map"] = ["1"] * n
    s["filament_volume_map"] = ["0"] * n
    s["filament_diameter_map"] = ["0.4"] * n
    s["flush_volumes_matrix"] = [("0" if i == j else "280") for i in range(n) for j in range(n)]
    s["flush_volumes_vector"] = ["140"] * (2 * n)
    s["different_settings_to_system"] = [""] * (n + 2)
    s["inherits_group"] = [""] * (n + 2)
    if prime_tower is not None:
        s["enable_prime_tower"] = "1" if prime_tower else "0"
    if wipe_tower_xy:
        s["wipe_tower_x"] = ["%.3f" % p[0] for p in wipe_tower_xy]
        s["wipe_tower_y"] = ["%.3f" % p[1] for p in wipe_tower_xy]
    else:
        s["wipe_tower_x"] = [s["wipe_tower_x"][0]] * n_plates
        s["wipe_tower_y"] = [s["wipe_tower_y"][0]] * n_plates
    return s


def _mesh_xml(vid: int, v: np.ndarray, t: np.ndarray, uid: str) -> str:
    vs = "\n".join('     <vertex x="%.4f" y="%.4f" z="%.4f"/>' % (a, b, c) for a, b, c in v.tolist())
    ts = "\n".join('     <triangle v1="%d" v2="%d" v3="%d"/>' % (a, b, c) for a, b, c in t.tolist())
    return (f'  <object id="{vid}" p:UUID="{uid}" type="model">\n   <mesh>\n    <vertices>\n{vs}\n'
            f'    </vertices>\n    <triangles>\n{ts}\n    </triangles>\n   </mesh>\n  </object>\n')


_NS = ('xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
       'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
       'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"')


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def write_bambu_3mf(path: str, objects: list[Obj], n_plates: int, filaments: Sequence[FilamentSpec],
                    plate_names: Optional[list[str]] = None, thumbnails: Optional[list[bytes]] = None,
                    wipe_tower_xy: Optional[list] = None, prime_tower: Optional[bool] = None,
                    plate_mm: float = 256.0, title: str = "Tessera") -> str:
    if n_plates > MAX_PLATES:
        raise ValueError(f"Bir 3MF dosyasında en fazla {MAX_PLATES} tabla olabilir")
    model_items, comp_objs, cfg_objs = [], [], []
    plate_inst: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_plates)}
    files: dict[str, str] = {}
    obj_files = []
    next_id = 1
    identify = 1000
    for k, ob in enumerate(objects):
        sub_ids = []
        parts_xml = []
        mesh_chunks = []
        face_total = 0
        for vol in ob.volumes:
            sid = next_id
            next_id += 1
            sub_ids.append(sid)
            mesh_chunks.append(_mesh_xml(sid, vol.vertices, vol.triangles, _uuid(f"{path}:{k}:{sid}")))
            nf = int(len(vol.triangles))
            face_total += nf
            parts_xml.append(
                f'    <part id="{sid}" subtype="normal_part">\n'
                f'      <metadata key="name" value="{_esc(vol.name)}"/>\n'
                f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                f'      <metadata key="extruder" value="{vol.extruder}"/>\n'
                f'      <mesh_stat face_count="{nf}" edges_fixed="0" degenerate_facets="0" facets_removed="0" '
                f'facets_reversed="0" backwards_edges="0"/>\n    </part>\n')
        oid = next_id
        next_id += 1
        fname = f"3D/Objects/object_{k + 1}.model"
        obj_files.append(fname)
        files[fname] = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                        f'<model unit="millimeter" xml:lang="en-US" {_NS}>\n'
                        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n'
                        + "".join(mesh_chunks) + ' </resources>\n <build/>\n</model>\n')
        comps = "".join(
            f'    <component p:path="/{fname}" objectid="{sid}" p:UUID="{_uuid(f"{path}:c:{sid}")}" '
            f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n' for sid in sub_ids)
        comp_objs.append(f'  <object id="{oid}" p:UUID="{_uuid(f"{path}:o:{oid}")}" type="model">\n'
                         f'   <components>\n{comps}   </components>\n  </object>\n')
        ox, oy = plate_origin(ob.plate, n_plates, plate_mm)
        x, y, z = ob.offset
        model_items.append(f'  <item objectid="{oid}" p:UUID="{_uuid(f"{path}:i:{oid}")}" '
                           f'transform="1 0 0 0 1 0 0 0 1 {x + ox:.4f} {y + oy:.4f} {z:.4f}" printable="1"/>\n')
        first_ex = ob.volumes[0].extruder if ob.volumes else 1
        cfg_objs.append(f'  <object id="{oid}">\n    <metadata key="name" value="{_esc(ob.name)}"/>\n'
                        f'    <metadata key="extruder" value="{first_ex}"/>\n'
                        f'    <metadata face_count="{face_total}"/>\n' + "".join(parts_xml) + '  </object>\n')
        plate_inst[ob.plate].append((oid, identify))
        identify += 1

    head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<model unit="millimeter" xml:lang="en-US" {_NS}>\n'
            f' <metadata name="Application">{_APP}</metadata>\n'
            ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            f' <metadata name="Title">{_esc(title)}</metadata>\n'
            ' <metadata name="Designer">Tessera</metadata>\n')
    files["3D/3dmodel.model"] = (head + ' <resources>\n' + "".join(comp_objs) + ' </resources>\n'
                                 f' <build p:UUID="{_uuid(path + ":build")}">\n' + "".join(model_items)
                                 + ' </build>\n</model>\n')
    files["3D/_rels/3dmodel.model.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "".join(f' <Relationship Target="/{f}" Id="rel-{i + 1}" '
                  'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
                  for i, f in enumerate(obj_files)) + '</Relationships>\n')
    plates_xml = []
    nf = len(filaments)
    for i in range(n_plates):
        name = _esc(plate_names[i]) if plate_names and i < len(plate_names) else ""
        inst = "".join(f'    <model_instance>\n      <metadata key="object_id" value="{oid}"/>\n'
                       f'      <metadata key="instance_id" value="0"/>\n'
                       f'      <metadata key="identify_id" value="{ident}"/>\n    </model_instance>\n'
                       for oid, ident in plate_inst[i])
        thumb = (f'    <metadata key="thumbnail_file" value="Metadata/plate_{i + 1}.png"/>\n'
                 if thumbnails and i < len(thumbnails) and thumbnails[i] else "")
        plates_xml.append(
            f'  <plate>\n    <metadata key="plater_id" value="{i + 1}"/>\n'
            f'    <metadata key="plater_name" value="{name}"/>\n    <metadata key="locked" value="false"/>\n'
            f'    <metadata key="filament_map_mode" value="Auto For Flush"/>\n'
            f'    <metadata key="filament_maps" value="{" ".join(["1"] * nf)}"/>\n'
            f'    <metadata key="filament_volume_maps" value="{" ".join(["0"] * nf)}"/>\n'
            + thumb + inst + '  </plate>\n')
    files["Metadata/model_settings.config"] = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
                                               + "".join(cfg_objs) + "".join(plates_xml)
                                               + '  <assemble>\n  </assemble>\n</config>\n')
    files["Metadata/project_settings.config"] = json.dumps(
        project_settings(filaments, wipe_tower_xy, n_plates, prime_tower), indent=4, ensure_ascii=False)
    files["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        ' <Default Extension="png" ContentType="image/png"/>\n'
        ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n</Types>\n')
    rels = ['<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n']
    if thumbnails and thumbnails[0]:
        rels.append(' <Relationship Target="/Metadata/plate_1.png" Id="rel-2" '
                    'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>\n')
    files["_rels/.rels"] = "".join(rels) + '</Relationships>\n'

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for name in ("[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "3D/_rels/3dmodel.model.rels"):
            z.writestr(name, files.pop(name))
        for name, data in files.items():
            z.writestr(name, data)
        if thumbnails:
            for i, png in enumerate(thumbnails):
                if png:
                    z.writestr(f"Metadata/plate_{i + 1}.png", png)
    return path


def manifold_to_volume(man, name: str, extruder: int, offset=(0.0, 0.0, 0.0)) -> Volume:
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    if any(offset):
        v = v + np.asarray(offset, dtype=np.float64)
    return Volume(name, v, np.asarray(mesh.tri_verts, dtype=np.int64), extruder)

"""3D team: Mosaic -> panels, parts, backing and meshes.

layout.build_layout(m)      panels + parts (2D, shapely, mm, y up)
mesh.build_part / build_backing   manifold3d solids (watertight)
packing.pack_outlines       MaxRects plate packing
"""
from .layout import GeoParams, Layout, Panel, Part, build_layout, colour_letters  # noqa: F401

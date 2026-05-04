"""
heat_flux.py
============
Computes the solar radiative heat flux distribution over a 3D building mesh.

Physical model
--------------
For each triangular face i of the mesh, the absorbed heat flux is:

    q_i = α_i · [ DNI · max(0, n̂_i · ŝ) · shadow_i
                + DHI · (1 + cos θ_i) / 2 ]          [W/m²]

Where:
    α_i      : absorptivity of face material (0–1)
    n̂_i      : outward unit normal of face i  (in Blender world space)
    ŝ        : unit vector pointing FROM scene TOWARD the sun
    shadow_i : 1 if face centroid has line-of-sight to sun, 0 if occluded
    θ_i      : tilt angle of face from horizontal (for sky view factor)
    DNI      : Direct Normal Irradiance  [W/m²]  from pvlib Ineichen model
    DHI      : Diffuse Horizontal Irradiance [W/m²]  from pvlib

Coordinate system
-----------------
House.obj uses OBJ convention: Y-up, Z-forward.
Blender (and pvlib→Blender conversion) uses: Z-up, Y-forward.
The standard OBJ→Blender remapping applied here is:
    (x, y, z)_obj  →  (x, -z, y)_blender

The sun direction vector ŝ is computed in Blender world space using the same
pvlib_to_blender_azimuth() convention as generate_try.py, so shadow geometry
and solar position are always consistent.

Output
------
- heat_flux_vectors : np.ndarray, shape (N_faces, 3)
      Per-face heat flux vector = q_i · n̂_i  [W/m²] in Blender world space
- heat_flux_scalars : np.ndarray, shape (N_faces,)
      Per-face scalar heat flux magnitude q_i  [W/m²]
- Coloured PLY file  : one colour per face, blue→yellow→red gradient
- PNG renders        : one per camera pose, projecting flux onto the image plane

Usage
-----
    python heat_flux.py \\
        --obj-path bproc_generator/data/example/House.obj \\
        --latitude 46.5 --longitude 6.6 --altitude 400 \\
        --date-time "2024-06-21 10:00:00" \\
        --output-path outputs/heatflux \\
        --camera-poses outputs/generated/10h00/meta_data.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pvlib
import pandas as pd
from PIL import Image
import tyro


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CIVIL_TWILIGHT_ZENITH: float = 96.0
DEFAULT_TURBIDITY:     float = 3.0
NORTH_OFFSET_DEG:      float = 180.0

# Default absorptivity per OBJ object name.
# α = 1.0 means all incident radiation is absorbed (conservative upper bound).
# Override via --absorptivity or the absorptivity_map parameter.
DEFAULT_ABSORPTIVITY: dict[str, float] = {
    "Front":  0.75,   # brick wall
    "Back":   0.75,
    "East":   0.75,
    "West":   0.75,
    "Roof":   0.90,   # dark slate tiles
    "Window": 0.10,   # glass — mostly transmits/reflects
    "Door":   0.80,   # wood
    "Others": 0.75,   # misc building parts
}

# Colourmap: blue (0 W/m²) → yellow → red (max W/m²)
# Sampled at 256 levels, stored as uint8 RGB tuples.
def _build_colormap() -> np.ndarray:
    """Returns a (256, 3) uint8 array: blue→cyan→yellow→red."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            s = t / 0.25
            cmap[i] = [0, int(s * 255), 255]               # blue → cyan
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            cmap[i] = [0, 255, int((1 - s) * 255)]         # cyan → green
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            cmap[i] = [int(s * 255), 255, 0]               # green → yellow
        else:
            s = (t - 0.75) / 0.25
            cmap[i] = [255, int((1 - s) * 255), 0]         # yellow → red
    return cmap

COLORMAP = _build_colormap()


# ---------------------------------------------------------------------------
# OBJ parser
# ---------------------------------------------------------------------------

@dataclass
class Face:
    """One triangular face of the mesh."""
    obj_name:    str           # OBJ object name (e.g. "Roof")
    vertices:    np.ndarray    # (3, 3) float — vertex positions in Blender space
    normal:      np.ndarray    # (3,)   float — outward unit normal in Blender space
    centroid:    np.ndarray    # (3,)   float — face centroid in Blender space
    area:        float         # face area [m² if OBJ units are metres]
    absorptivity:float = 1.0   # material absorptivity α


def _obj_to_blender(v: np.ndarray) -> np.ndarray:
    """
    Converts a point or vector from OBJ space to Blender world space.
    OBJ: Y-up, Z-forward  →  Blender: Z-up, Y-forward
    Mapping: (x, y, z)_obj → (x, -z, y)_blender
    """
    return np.array([v[0], -v[2], v[1]], dtype=np.float64)


def parse_obj(obj_path: Path,
              absorptivity_map: dict[str, float] | None = None) -> list[Face]:
    """
    Parses an OBJ file and returns a list of triangulated Face objects.

    Faces with more than 3 vertices are fan-triangulated.
    All positions and normals are converted to Blender world space.

    Parameters
    ----------
    obj_path : Path
        Path to the .obj file.
    absorptivity_map : dict, optional
        Maps OBJ object names to absorptivity values.
        Defaults to DEFAULT_ABSORPTIVITY; missing keys default to 0.75.

    Returns
    -------
    list[Face]
        One Face per triangle in the mesh.
    """
    if absorptivity_map is None:
        absorptivity_map = DEFAULT_ABSORPTIVITY

    vertices: list[np.ndarray] = []   # raw OBJ vertices (Y-up)
    normals:  list[np.ndarray] = []   # raw OBJ normals  (Y-up)
    faces:    list[Face]       = []

    current_obj = "Unknown"

    with open(obj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("o "):
                current_obj = line[2:].strip()

            elif line.startswith("v ") and not line.startswith("vn") \
                                       and not line.startswith("vt"):
                parts = line.split()
                vertices.append(np.array([float(parts[1]),
                                          float(parts[2]),
                                          float(parts[3])], dtype=np.float64))

            elif line.startswith("vn "):
                parts = line.split()
                normals.append(np.array([float(parts[1]),
                                         float(parts[2]),
                                         float(parts[3])], dtype=np.float64))

            elif line.startswith("f "):
                parts = line.split()[1:]
                # Parse indices — format is v/vt/vn or v//vn or v
                v_idx, vn_idx = [], []
                for p in parts:
                    tokens = p.split("/")
                    v_idx.append(int(tokens[0]) - 1)
                    vn_idx.append(int(tokens[2]) - 1 if len(tokens) > 2
                                  and tokens[2] else None)

                # Fan-triangulate: (0,1,2), (0,2,3), (0,3,4), ...
                absorptivity = absorptivity_map.get(current_obj, 0.75)
                for i in range(1, len(v_idx) - 1):
                    tri_v = [v_idx[0], v_idx[i], v_idx[i + 1]]
                    tri_n = [vn_idx[0], vn_idx[i], vn_idx[i + 1]]

                    # Vertex positions in Blender space
                    verts_bl = np.array([_obj_to_blender(vertices[j])
                                         for j in tri_v])

                    # Face normal: average of vertex normals, converted to Blender
                    if all(n is not None for n in tri_n):
                        raw_n = np.mean([normals[j] for j in tri_n], axis=0)
                        n_bl  = _obj_to_blender(raw_n)
                    else:
                        # Fallback: compute from cross product
                        e1 = verts_bl[1] - verts_bl[0]
                        e2 = verts_bl[2] - verts_bl[0]
                        n_bl = np.cross(e1, e2)

                    norm = np.linalg.norm(n_bl)
                    if norm < 1e-10:
                        continue   # degenerate face — skip
                    n_bl /= norm

                    centroid = verts_bl.mean(axis=0)

                    # Face area (half cross-product magnitude)
                    e1 = verts_bl[1] - verts_bl[0]
                    e2 = verts_bl[2] - verts_bl[0]
                    area = 0.5 * np.linalg.norm(np.cross(e1, e2))

                    faces.append(Face(
                        obj_name=current_obj,
                        vertices=verts_bl,
                        normal=n_bl,
                        centroid=centroid,
                        area=area,
                        absorptivity=absorptivity,
                    ))

    return faces


# ---------------------------------------------------------------------------
# Solar position & irradiance
# ---------------------------------------------------------------------------

def get_solar_irradiance(latitude: float, longitude: float, altitude: float,
                         date_time: str,
                         turbidity: float = DEFAULT_TURBIDITY) -> dict:
    """
    Returns solar position and clear-sky irradiance using pvlib Ineichen.

    Returns
    -------
    dict with keys: dni, dhi, ghi, zenith, azimuth, elevation,
                    sun_above_horizon, in_civil_twilight
    """
    dt_utc = pd.Timestamp(date_time, tz="UTC")
    times  = pd.DatetimeIndex([dt_utc])

    pos     = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    zenith  = float(pos["apparent_zenith"].iloc[0])
    azimuth = float(pos["azimuth"].iloc[0])
    elev    = float(pos["apparent_elevation"].iloc[0])

    sun_above  = zenith < 90.0
    in_twilight = 90.0 <= zenith <= CIVIL_TWILIGHT_ZENITH

    if sun_above or in_twilight:
        am_rel = pvlib.atmosphere.get_relative_airmass(zenith)
        am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel)
        cs = pvlib.clearsky.ineichen(
            apparent_zenith=pd.Series([zenith], index=times),
            airmass_absolute=pd.Series([am_abs], index=times),
            linke_turbidity=turbidity,
        )
        dni = float(cs["dni"].iloc[0]) if sun_above else 0.0
        dhi = float(cs["dhi"].iloc[0])
        ghi = float(cs["ghi"].iloc[0])
    else:
        dni = dhi = ghi = 0.0

    return {
        "dni": dni, "dhi": dhi, "ghi": ghi,
        "zenith": zenith, "azimuth": azimuth, "elevation": elev,
        "sun_above_horizon": sun_above,
        "in_civil_twilight": in_twilight,
    }


def sun_direction_vector(azimuth_pvlib_deg: float,
                         elevation_deg: float,
                         north_offset_deg: float = NORTH_OFFSET_DEG) -> np.ndarray:
    """
    Returns unit vector pointing FROM scene TOWARD the sun, in Blender world space.

    Uses the same coordinate convention as generate_try.py so that heat flux
    shadows are consistent with rendered shadows.

    Parameters
    ----------
    azimuth_pvlib_deg : float
        Solar azimuth from pvlib (N=0, E=90, S=180, W=270, clockwise).
    elevation_deg : float
        Solar elevation above horizon [°].
    north_offset_deg : float
        Azimuth correction for building orientation (180° for House.obj).

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector (x, y, z) in Blender world space pointing toward the sun.
    """
    az_bl  = math.radians(-azimuth_pvlib_deg + north_offset_deg)
    el_rad = math.radians(elevation_deg)

    # Sun position vector (toward sun, opposite of light travel direction)
    sx = math.sin(az_bl) * math.cos(el_rad)
    sy = math.cos(az_bl) * math.cos(el_rad)
    sz = math.sin(el_rad)
    return np.array([sx, sy, sz], dtype=np.float64)


# ---------------------------------------------------------------------------
# Shadow testing via BVH
# ---------------------------------------------------------------------------

class BVHNode:
    """Minimal axis-aligned bounding box BVH for ray–triangle intersection."""

    def __init__(self, faces: list[Face], depth: int = 0):
        self.faces    = faces
        self.left     = None
        self.right    = None
        self.is_leaf  = False

        # Compute AABB
        all_verts = np.vstack([f.vertices for f in faces])
        self.aabb_min = all_verts.min(axis=0) - 1e-4
        self.aabb_max = all_verts.max(axis=0) + 1e-4

        if len(faces) <= 8 or depth > 20:
            self.is_leaf = True
            return

        # Split along longest axis
        extent = self.aabb_max - self.aabb_min
        axis   = int(np.argmax(extent))
        mid    = np.median([f.centroid[axis] for f in faces])

        left_faces  = [f for f in faces if f.centroid[axis] <= mid]
        right_faces = [f for f in faces if f.centroid[axis] >  mid]

        if not left_faces or not right_faces:
            self.is_leaf = True
            return

        self.left  = BVHNode(left_faces,  depth + 1)
        self.right = BVHNode(right_faces, depth + 1)

    def _ray_aabb(self, origin: np.ndarray, inv_dir: np.ndarray) -> bool:
        t1 = (self.aabb_min - origin) * inv_dir
        t2 = (self.aabb_max - origin) * inv_dir
        tmin = np.maximum(t1, t2).min()
        tmax = np.minimum(t1, t2).max()
        return tmax >= max(tmin, 1e-4)

    def ray_intersects(self, origin: np.ndarray, direction: np.ndarray,
                       max_dist: float = 1e9) -> bool:
        """
        Returns True if the ray from origin in direction hits any face
        within max_dist (Möller–Trumbore algorithm).
        """
        inv_dir = np.where(np.abs(direction) > 1e-12,
                           1.0 / direction, np.sign(direction) * 1e12)

        if not self._ray_aabb(origin, inv_dir):
            return False

        if self.is_leaf:
            for face in self.faces:
                v0, v1, v2 = face.vertices
                e1 = v1 - v0
                e2 = v2 - v0
                h  = np.cross(direction, e2)
                a  = e1 @ h
                if abs(a) < 1e-8:
                    continue
                f  = 1.0 / a
                s  = origin - v0
                u  = f * (s @ h)
                if u < 0 or u > 1:
                    continue
                q  = np.cross(s, e1)
                v  = f * (direction @ q)
                if v < 0 or u + v > 1:
                    continue
                t  = f * (e2 @ q)
                if 1e-4 < t < max_dist:
                    return True
            return False

        return (self.left  is not None and
                self.left.ray_intersects(origin, direction, max_dist)) or \
               (self.right is not None and
                self.right.ray_intersects(origin, direction, max_dist))


# ---------------------------------------------------------------------------
# Heat flux computation
# ---------------------------------------------------------------------------

def compute_heat_flux(
    faces:      list[Face],
    sun_vec:    np.ndarray,    # unit vector toward sun, Blender space
    dni:        float,         # W/m²
    dhi:        float,         # W/m²
    bvh:        BVHNode,
    offset:     float = 0.005, # ray origin offset along normal to avoid self-intersection
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes per-face solar heat flux using Lambert's cosine law +
    sky view factor + BVH shadow testing.

    Parameters
    ----------
    faces   : list of Face objects (from parse_obj)
    sun_vec : unit vector pointing toward sun (Blender world space)
    dni     : Direct Normal Irradiance [W/m²]
    dhi     : Diffuse Horizontal Irradiance [W/m²]
    bvh     : BVH tree built from the same face list
    offset  : small offset to avoid self-intersection in shadow rays

    Returns
    -------
    scalars : np.ndarray (N_faces,)  — flux magnitude q_i [W/m²]
    vectors : np.ndarray (N_faces, 3) — flux vector q_i · n̂_i [W/m²]
    """
    N = len(faces)
    scalars = np.zeros(N, dtype=np.float64)
    vectors = np.zeros((N, 3), dtype=np.float64)

    for i, face in enumerate(faces):
        n = face.normal
        α = face.absorptivity

        # ── Direct component (DNI · Lambert cosine law) ──────────────────
        cos_theta = float(np.dot(n, sun_vec))
        if cos_theta > 0 and dni > 0:
            # Cast shadow ray from face centroid toward sun
            origin = face.centroid + offset * n
            in_shadow = bvh.ray_intersects(origin, sun_vec)
            direct = α * dni * cos_theta * (0.0 if in_shadow else 1.0)
        else:
            direct = 0.0   # face pointing away from sun

        # ── Diffuse component (DHI · sky view factor) ────────────────────
        # Sky view factor for a tilted surface:
        #   F_sky = (1 + cos θ_tilt) / 2
        # where θ_tilt is the angle of the face normal from vertical (Z-up).
        cos_tilt = float(n[2])   # n · ẑ, where ẑ = (0,0,1) in Blender
        sky_vf   = (1.0 + cos_tilt) / 2.0   # 1.0 for horizontal, 0.5 for vertical
        diffuse  = α * dhi * max(0.0, sky_vf)

        q = direct + diffuse
        scalars[i]   = q
        vectors[i]   = q * n

    return scalars, vectors


# ---------------------------------------------------------------------------
# Coloured PLY export
# ---------------------------------------------------------------------------

def export_ply(faces: list[Face], scalars: np.ndarray,
               output_path: Path,
               q_max_override: float | None = None) -> None:
    """
    Exports the mesh as a PLY file with per-face colours mapped from
    heat flux magnitude (blue = 0 W/m², red = max W/m²).

    Parameters
    ----------
    faces          : list of Face objects
    scalars        : (N_faces,) heat flux magnitudes [W/m²]
    output_path    : path for the output .ply file
    q_max_override : fixed colormap ceiling [W/m²]. If None, uses scene max.
    """
    q_max = q_max_override if q_max_override is not None \
            else (scalars.max() if scalars.max() > 0 else 1.0)
    n_faces = len(faces)
    n_verts = n_faces * 3   # un-indexed — one set of 3 verts per face

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_verts}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {n_faces}",
        "property list uchar int vertex_indices",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    vert_lines = []
    face_lines = []

    for i, (face, q) in enumerate(zip(faces, scalars)):
        base = i * 3
        for v in face.vertices:
            vert_lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        # Map q to colormap index
        idx = int(min(255, (q / q_max) * 255))
        r, g, b = COLORMAP[idx]
        face_lines.append(f"3 {base} {base+1} {base+2} {r} {g} {b}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines + vert_lines + face_lines) + "\n")

    print(f"  PLY exported → {output_path}")
    print(f"  Flux range   : 0 – {q_max:.1f} W/m²")


# ---------------------------------------------------------------------------
# PNG heat flux render (project onto camera)
# ---------------------------------------------------------------------------

def render_flux_png(faces: list[Face], scalars: np.ndarray,
                    camera_to_world: np.ndarray,
                    intrinsics: np.ndarray,
                    resolution: tuple[int, int],
                    output_path: Path,
                    q_max_override: float | None = None) -> None:
    """
    Projects the per-face heat flux onto a camera image plane and saves as PNG.

    BlenderProc camera convention (OpenCV):
      - camera_to_world[:3, 3] = camera position in world space
      - camera looks along +Z in camera space
      - Y axis points downward in image space
    World-to-camera = inv(camera_to_world)

    Parameters
    ----------
    faces          : list of Face objects
    scalars        : (N_faces,) heat flux magnitudes [W/m²]
    camera_to_world: (4,4) BlenderProc camera pose matrix
    intrinsics     : (3,3) camera intrinsic matrix K
    resolution     : (width, height) in pixels
    output_path    : path for the output PNG
    """
    W, H  = resolution
    img   = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf  = np.full((H, W), np.inf, dtype=np.float64)
    q_max = q_max_override if q_max_override is not None \
            else (scalars.max() if scalars.max() > 0 else 1.0)

    # World-to-camera: invert the camera_to_world pose
    c2w = np.array(camera_to_world, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    R   = w2c[:3, :3]
    t   = w2c[:3,  3]

    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    def world_to_pixel(pt: np.ndarray):
        """
        Projects a world-space point to pixel (u, v) and returns depth.
        BlenderProc cameras look along -Z, Y points up in camera space.
        Returns None if point is behind the camera.
        """
        pc    = R @ pt + t
        depth = -pc[2]        # BlenderProc: forward = -Z, so depth = -z
        if depth <= 0.01:
            return None
        u = int(round(fx *   pc[0]  / depth + cx))
        v = int(round(fy * (-pc[1]) / depth + cy))  # Y flipped: image Y down
        return u, v, depth

    def draw_triangle(p0, p1, p2, color, depth):
        """Rasterizes a filled triangle with z-buffer test."""
        xs = [p0[0], p1[0], p2[0]]
        ys = [p0[1], p1[1], p2[1]]
        xmin = max(0,     min(xs))
        xmax = min(W - 1, max(xs))
        ymin = max(0,     min(ys))
        ymax = min(H - 1, max(ys))

        if xmin > xmax or ymin > ymax:
            return

        ax, ay = p1[0] - p0[0], p1[1] - p0[1]
        bx, by = p2[0] - p0[0], p2[1] - p0[1]
        denom  = ax * by - ay * bx
        if abs(denom) < 1e-8:
            return

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                px, py = x - p0[0], y - p0[1]
                u = (px * by - py * bx) / denom
                v = (ax * py - ay * px) / denom
                if u >= 0 and v >= 0 and (u + v) <= 1:
                    if depth < zbuf[y, x]:
                        zbuf[y, x]   = depth
                        img[y, x, :] = color

    # Sort faces back-to-front (painter's algorithm)
    # BlenderProc: depth = -z in camera space, sort descending depth
    def face_depth(i):
        pts = [R @ v + t for v in faces[i].vertices]
        return np.mean([p[2] for p in pts])   # more positive z = further away

    order = sorted(range(len(faces)), key=face_depth)

    n_drawn = 0
    for i in order:
        face  = faces[i]
        q     = scalars[i]
        cidx  = int(min(255, (q / q_max) * 255))
        color = COLORMAP[cidx]

        projs = [world_to_pixel(v) for v in face.vertices]
        if any(p is None for p in projs):
            continue

        depth = float(np.mean([p[2] for p in projs]))
        p0 = (projs[0][0], projs[0][1])
        p1 = (projs[1][0], projs[1][1])
        p2 = (projs[2][0], projs[2][1])

        # Skip if all vertices outside frame
        in_frame = any(
            0 <= p[0] < W and 0 <= p[1] < H
            for p in [p0, p1, p2]
        )
        if not in_frame:
            continue

        draw_triangle(p0, p1, p2, color, depth)
        n_drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)
    print(f"    Drew {n_drawn}/{len(faces)} faces → {output_path.name}")


# ---------------------------------------------------------------------------
# Colorbar legend
# ---------------------------------------------------------------------------

def save_colorbar(q_max: float, output_path: Path,
                  width: int = 40, height: int = 256) -> None:
    """Saves a vertical colorbar PNG with flux range annotation."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        idx = int((1.0 - row / (height - 1)) * 255)
        bar[row, :] = COLORMAP[idx]

    img = Image.fromarray(bar)
    img.save(output_path)
    print(f"  Colorbar saved → {output_path}  (0 – {q_max:.1f} W/m²)")


def compute_solstice_q_max(latitude: float, longitude: float,
                            altitude: float, year: int,
                            turbidity: float = DEFAULT_TURBIDITY) -> float:
    """
    Computes the maximum possible solar heat flux on the summer solstice
    (June 21) for the given location and year.

    Uses a fast analytical estimate: at each 10-min solar position sample,
    the theoretical maximum per-face flux is α_max × DNI + α_max × DHI
    (face perfectly perpendicular to the sun, α_max = 0.9 for roof tiles).
    This avoids running the full BVH for the calibration pass.

    Returns the peak value with a 5% margin — used as the fixed colormap
    ceiling so colours are physically comparable across all times of day
    and all dates within the same year and location.
    """
    solstice_date = f"{year}-06-21"
    print(f"  Computing solstice q_max for {solstice_date} "
          f"at {latitude}°N {longitude}°E {altitude}m...")

    times = pd.date_range(f"{solstice_date} 00:00",
                          f"{solstice_date} 23:59",
                          freq="10min", tz="UTC")
    solar = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    q_max = 0.0
    for ts, row in solar.iterrows():
        if float(row["apparent_zenith"]) >= 90.0:
            continue
        sol = get_solar_irradiance(
            latitude, longitude, altitude,
            ts.strftime("%Y-%m-%d %H:%M:%S"), turbidity
        )
        # Best-case face: perfectly perpendicular to sun (cos θ = 1)
        # using α_roof = 0.9 (highest absorptivity in DEFAULT_ABSORPTIVITY)
        q_face = 0.9 * sol["dni"] + 0.9 * sol["dhi"]
        if q_face > q_max:
            q_max = q_face

    q_max_margin = q_max * 1.05
    print(f"  Solstice q_max  : {q_max:.1f} W/m²  "
          f"→ +5% margin = {q_max_margin:.1f} W/m²")
    return q_max_margin


def get_day_timesteps(date: str, latitude: float, longitude: float) -> list[str]:
    """
    Returns renderable UTC timesteps at :00 and :30 bounded by civil twilight.
    Identical to the helper in generate_try.py.
    """
    times = pd.date_range(f"{date} 00:00", f"{date} 23:59",
                          freq="10min", tz="UTC")
    solar = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    lit   = solar[solar["apparent_zenith"] <= CIVIL_TWILIGHT_ZENITH]
    if lit.empty:
        return []

    def _round_up(dt):
        if dt.minute < 30:
            return dt.replace(minute=30, second=0, microsecond=0)
        return (dt + pd.Timedelta(hours=1)).replace(minute=0, second=0,
                                                     microsecond=0)
    def _round_down(dt):
        if dt.minute >= 30:
            return dt.replace(minute=30, second=0, microsecond=0)
        return dt.replace(minute=0, second=0, microsecond=0)

    steps = pd.date_range(_round_up(lit.index[0]), _round_down(lit.index[-1]),
                          freq="30min")
    return [s.strftime("%Y-%m-%d %H:%M:%S") for s in steps]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@dataclass
class HeatFluxParams:
    """Parameters for solar heat flux computation."""

    obj_path:    Path = Path("bproc_generator/data/example/House.obj")
    """Path to the building OBJ file."""
    latitude:    float = tyro.MISSING
    """Building latitude [decimal degrees]."""
    longitude:   float = tyro.MISSING
    """Building longitude [decimal degrees]."""
    altitude:    float = 400.0
    """Building altitude above sea level [m]."""
    date:        str = tyro.MISSING
    """
    UTC date for the full-day loop: 'YYYY-MM-DD'.
    Runs every 30-min step between civil twilight bounds automatically.
    The year is extracted from this date to compute the solstice q_max.
    """
    turbidity:   float = DEFAULT_TURBIDITY
    """Linke turbidity factor."""
    north_offset_deg: float = NORTH_OFFSET_DEG
    """Azimuth correction for building orientation [°]."""
    output_path: Path = Path("outputs/generated")
    """
    Root output folder — same root as generate_try.py.
    Heat flux PNGs go into output_path/10h00/heatflux/ etc.,
    alongside the existing images/, depths/, normals/ subfolders.
    """
    resolution:  tuple[int, int] = (512, 512)
    """Image resolution for PNG renders."""
    q_max:       float = 0.0
    """
    Fixed colormap ceiling [W/m²].
    If 0 (default), automatically computed as the summer solstice maximum
    for the given location and year — physically correct and location-aware.
    Override with a positive value to use a custom fixed scale.
    """


@dataclass
class HeatFluxRunner(HeatFluxParams):

    def run(self):
        # ── Step 1: Resolve colormap ceiling ─────────────────────────────
        # If q_max == 0 (default), compute it dynamically from the summer
        # solstice of the input year and location — fully automatic and
        # physically correct for any location worldwide.
        q_max = self.q_max
        if q_max <= 0:
            year  = int(self.date[:4])
            q_max = compute_solstice_q_max(
                self.latitude, self.longitude, self.altitude,
                year, self.turbidity,
            )

        print(f"\n{'═' * 55}")
        print(f"  Solar Heat Flux — daily loop")
        print(f"  Date     : {self.date}")
        print(f"  Location : {self.latitude}°N  {self.longitude}°E  "
              f"{self.altitude}m")
        print(f"  Colormap : 0 – {q_max:.1f} W/m²  "
              f"({'auto solstice' if self.q_max <= 0 else 'user override'})")
        print(f"{'═' * 55}\n")

        # ── Step 2: Parse OBJ and build BVH once ─────────────────────────
        print(f"  Parsing OBJ : {self.obj_path}")
        faces = parse_obj(self.obj_path)
        print(f"  Faces       : {len(faces)}")
        print(f"  Building BVH tree...")
        bvh = BVHNode(faces)

        # ── Step 2: Get timesteps ─────────────────────────────────────────
        timesteps = get_day_timesteps(self.date, self.latitude, self.longitude)
        if not timesteps:
            print("  ⚠  No renderable timesteps.")
            return
        print(f"  Timesteps   : {len(timesteps)}  "
              f"({timesteps[0][11:16]} → {timesteps[-1][11:16]} UTC)\n")

        # ── Step 3: Shared colorbar ───────────────────────────────────────
        self.output_path.mkdir(parents=True, exist_ok=True)
        save_colorbar(q_max,
                      self.output_path / "heatflux_colorbar.png")

        day_summary = []

        # ── Step 4: Loop over timesteps ───────────────────────────────────
        for date_time in timesteps:
            hhmm = date_time[11:16].replace(":", "h")
            print(f"\n  ── Timestep {hhmm} {'─' * 38}")

            sol = get_solar_irradiance(
                self.latitude, self.longitude, self.altitude,
                date_time, self.turbidity,
            )
            print(f"  el={sol['elevation']:+.1f}°  "
                  f"DNI={sol['dni']:.0f}  DHI={sol['dhi']:.0f} W/m²")

            if sol["sun_above_horizon"] or sol["in_civil_twilight"]:
                sun_vec = sun_direction_vector(
                    sol["azimuth"], sol["elevation"], self.north_offset_deg
                )
                scalars, vectors = compute_heat_flux(
                    faces, sun_vec, sol["dni"], sol["dhi"], bvh
                )
            else:
                scalars = np.zeros(len(faces))
                vectors = np.zeros((len(faces), 3))

            print(f"  flux: {scalars.min():.0f}–{scalars.max():.0f} W/m²  "
                  f"(scale 0–{q_max:.0f})")

            # Output goes into output_path/10h00/heatflux/
            # sitting next to images/, depths/, normals/ etc.
            step_path    = self.output_path / hhmm
            heatflux_dir = step_path / "heatflux"
            heatflux_dir.mkdir(parents=True, exist_ok=True)

            # Arrays
            np.save(heatflux_dir / "scalars.npy", scalars)
            np.save(heatflux_dir / "vectors.npy", vectors)

            # Coloured PLY mesh with fixed scale
            export_ply(faces, scalars,
                       heatflux_dir / "heat_flux.ply",
                       q_max_override=q_max)

            # Per-timestep metadata
            total_power = float(sum(
                scalars[i] * faces[i].area for i in range(len(faces))
            ))
            with open(heatflux_dir / "meta_data.json", "w") as f:
                json.dump({
                    "date_time":        date_time,
                    "solar_elevation":  sol["elevation"],
                    "solar_azimuth":    sol["azimuth"],
                    "DNI_Wm2":          sol["dni"],
                    "DHI_Wm2":          sol["dhi"],
                    "flux_min_Wm2":     float(scalars.min()),
                    "flux_max_Wm2":     float(scalars.max()),
                    "flux_mean_Wm2":    float(scalars.mean()),
                    "total_power_W":    total_power,
                    "colormap_ceiling": q_max,
                }, f, indent=4)

            # PNG renders — read camera poses from sibling meta_data.json
            meta_json = step_path / "meta_data.json"
            if meta_json.exists():
                with open(meta_json) as f:
                    render_meta = json.load(f)
                K = np.array(render_meta["frames"][0]["intrinsics"])
                for frame in render_meta["frames"]:
                    c2w   = np.array(frame["camera_to_world"])
                    idx   = frame["rgb_path"].replace(".png", "")
                    out_p = heatflux_dir / f"{idx}_flux.png"
                    render_flux_png(faces, scalars, c2w, K,
                                    self.resolution, out_p,
                                    q_max_override=q_max)
            else:
                print(f"  ⚠  {meta_json} not found — no PNG renders")

            day_summary.append({
                "time_utc":      date_time,
                "hhmm":          hhmm,
                "elevation":     sol["elevation"],
                "DNI_Wm2":       sol["dni"],
                "DHI_Wm2":       sol["dhi"],
                "flux_max_Wm2":  float(scalars.max()),
                "flux_mean_Wm2": float(scalars.mean()),
                "total_power_W": total_power,
            })

        # ── Step 5: Day summary ───────────────────────────────────────────
        with open(self.output_path / "heatflux_day_summary.json", "w") as f:
            json.dump({
                "date":             self.date,
                "latitude":         self.latitude,
                "longitude":        self.longitude,
                "colormap_ceiling": q_max,
                "timesteps":        day_summary,
            }, f, indent=4)

        print(f"\n{'═' * 55}")
        print(f"  Done. {len(timesteps)} timesteps processed.")
        print(f"  Summary → {self.output_path / 'heatflux_day_summary.json'}")
        print(f"{'═' * 55}\n")


def main():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(HeatFluxRunner).run()


if __name__ == "__main__":
    main()
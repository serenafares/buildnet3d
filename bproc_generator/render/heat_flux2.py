"""
heat_flux.py
============
Computes the solar radiative heat flux distribution over a 3D building mesh.

Physical model
--------------
Two outputs are produced for each face i:

  Incident irradiance (before absorption):
    E_i = DNI · max(0, n̂_i · ŝ) · shadow_i  +  DHI · (1 + cos θ_i) / 2   [W/m²]

  Absorbed heat flux (after absorption):
    q_i = α_i · E_i                                                           [W/m²]

Where:
    α_i      : absorptivity of face material (0–1)
    n̂_i      : outward unit normal of face i  (in Blender world space)
    ŝ        : unit vector pointing FROM scene TOWARD the sun
    shadow_i : 1 if face centroid has line-of-sight to sun, 0 if occluded
    θ_i      : tilt angle of face from horizontal (for sky view factor)
    DNI      : Direct Normal Irradiance  [W/m²]  from pvlib Ineichen model
    DHI      : Diffuse Horizontal Irradiance [W/m²]  from pvlib

Rendering improvements
----------------------
- Turbo colormap   : smooth perceptual gradient, keeps deep red at maximum
- Per-timestep     : colormap stretches to actual min/max of each frame
  histogram eq.      (not a fixed solstice ceiling) → full range always used
- Diffuse shading  : face color multiplied by Lambert factor from a fixed
                     overhead-south light → adds 3D depth without outlines

Output structure (per timestep, e.g. 10h00/)
--------------------------------------------
  heatflux/
    incident/          ← E_i  (no absorptivity)
      scalars.npy
      vectors.npy
      heat_flux.ply
      0000_flux.png …
    absorbed/          ← q_i = α_i · E_i
      scalars.npy
      vectors.npy
      heat_flux.ply
      0000_flux.png …
    meta_data.json

Coordinate system
-----------------
House.obj: Y-up, Z-forward  →  Blender: Z-up, Y-forward
Remapping: (x, y, z)_obj → (x, -z, y)_blender

The sun direction vector ŝ uses the same pvlib_to_blender_azimuth()
convention as generate_try.py so shadow geometry is always coherent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pvlib
import pandas as pd
from PIL import Image
import tyro



# ─── Physical constants ───────────────────────────────────────────────────────

CIVIL_TWILIGHT_ZENITH: float = 96.0
DEFAULT_TURBIDITY:     float = 3.0
NORTH_OFFSET_DEG:      float = 180.0

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

# Fixed ambient light direction for diffuse shading (gives 3D depth).
# Slightly overhead and toward south — independent of sun position so
# depth cues are always visible even at civil twilight.
_SHADE_DIR = np.array([0.2, 0.4, 0.9], dtype=np.float64)
_SHADE_DIR /= np.linalg.norm(_SHADE_DIR)
SHADE_AMBIENT = 0.55   # minimum brightness (0–1) for fully back-facing faces
SHADE_DIFFUSE = 0.45   # additional brightness for fully front-facing faces


# ─── Turbo colormap ───────────────────────────────────────────────────────────

def _build_turbo() -> np.ndarray:
    """
    Returns (256, 3) uint8 — Google's Turbo colormap.
    Deep blue → cyan → green → yellow → orange → deep red.
    Perceptually smooth, keeps the red maximum signal.
    """
    turbo_data = [
        [0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],
        [0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],
        [0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],
        [0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],
        [0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],
        [0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],
        [0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],
        [0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],
        [0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],
        [0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],
        [0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],
        [0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],
        [0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],
        [0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],
        [0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],
        [0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],
        [0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],
        [0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],
        [0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],
        [0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],
        [0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],
        [0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],
        [0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],
        [0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],
        [0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],
        [0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],
        [0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],
        [0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],
        [0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],
        [0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],
        [0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],
        [0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],
        [0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],
        [0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],
        [0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],
        [0.12151,0.78896,0.86581],[0.11639,0.79740,0.85556],
        [0.11167,0.80569,0.84518],[0.10738,0.81381,0.83469],
        [0.10357,0.82177,0.82412],[0.10026,0.82955,0.81350],
        [0.09750,0.83714,0.80287],[0.09532,0.84455,0.79223],
        [0.09377,0.85175,0.78161],[0.09287,0.85875,0.77104],
        [0.09267,0.86554,0.76053],[0.09320,0.87211,0.75011],
        [0.09451,0.87844,0.73980],[0.09662,0.88454,0.72961],
        [0.09958,0.89040,0.71956],[0.10342,0.89600,0.70967],
        [0.10815,0.90142,0.69995],[0.11374,0.90673,0.69042],
        [0.12014,0.91193,0.68108],[0.12733,0.91701,0.67194],
        [0.13526,0.92197,0.66302],[0.14391,0.92680,0.65430],
        [0.15323,0.93151,0.64581],[0.16319,0.93609,0.63753],
        [0.17377,0.94053,0.62947],[0.18491,0.94484,0.62163],
        [0.19659,0.94901,0.61400],[0.20877,0.95304,0.60659],
        [0.22142,0.95692,0.59938],[0.23449,0.96065,0.59237],
        [0.24797,0.96423,0.58556],[0.26180,0.96765,0.57893],
        [0.27597,0.97092,0.57247],[0.29042,0.97403,0.56618],
        [0.30513,0.97697,0.56004],[0.32006,0.97974,0.55404],
        [0.33517,0.98234,0.54818],[0.35043,0.98477,0.54243],
        [0.36581,0.98702,0.53678],[0.38127,0.98909,0.53122],
        [0.39678,0.99098,0.52576],[0.41229,0.99268,0.52036],
        [0.42778,0.99419,0.51503],[0.44321,0.99551,0.50975],
        [0.45854,0.99663,0.50451],[0.47375,0.99755,0.49931],
        [0.48879,0.99828,0.49393],[0.50362,0.99879,0.48839],
        [0.51822,0.99910,0.48271],[0.53255,0.99919,0.47689],
        [0.54658,0.99907,0.47093],[0.56026,0.99873,0.46484],
        [0.57357,0.99817,0.45862],[0.58646,0.99739,0.45230],
        [0.59891,0.99638,0.44586],[0.61088,0.99514,0.43932],
        [0.62233,0.99366,0.43268],[0.63323,0.99195,0.42595],
        [0.64362,0.98999,0.41914],[0.65394,0.98775,0.41222],
        [0.66428,0.98524,0.40525],[0.67462,0.98246,0.39819],
        [0.68494,0.97941,0.39108],[0.69525,0.97610,0.38393],
        [0.70553,0.97255,0.37674],[0.71577,0.96875,0.36952],
        [0.72596,0.96470,0.36228],[0.73610,0.96043,0.35503],
        [0.74617,0.95593,0.34778],[0.75617,0.95121,0.34055],
        [0.76608,0.94627,0.33334],[0.77591,0.94113,0.32616],
        [0.78563,0.93579,0.31901],[0.79524,0.93025,0.31192],
        [0.80473,0.92452,0.30489],[0.81410,0.91861,0.29794],
        [0.82333,0.91253,0.29108],[0.83241,0.90627,0.28432],
        [0.84133,0.89986,0.27767],[0.85010,0.89328,0.27115],
        [0.85868,0.88655,0.26476],[0.86709,0.87968,0.25851],
        [0.87530,0.87267,0.25242],[0.88331,0.86553,0.24649],
        [0.89112,0.85826,0.24074],[0.89870,0.85087,0.23516],
        [0.90605,0.84337,0.22977],[0.91317,0.83576,0.22458],
        [0.92004,0.82305,0.21959],[0.92666,0.82025,0.21481],
        [0.93301,0.81236,0.21025],[0.93909,0.80439,0.20593],
        [0.94489,0.79634,0.20186],[0.95039,0.78823,0.19805],
        [0.95560,0.78005,0.19452],[0.96049,0.77181,0.19126],
        [0.96507,0.76352,0.18828],[0.96931,0.75519,0.18561],
        [0.97323,0.74682,0.18324],[0.97679,0.73842,0.18118],
        [0.98000,0.73000,0.17945],[0.98289,0.72140,0.17810],
        [0.98549,0.71250,0.17690],[0.98781,0.70330,0.17628],
        [0.98986,0.69382,0.17554],[0.99163,0.68408,0.17512],
        [0.99314,0.67408,0.17505],[0.99438,0.66386,0.17534],
        [0.99535,0.65341,0.17598],[0.99607,0.64277,0.17699],
        [0.99654,0.63193,0.17835],[0.99675,0.62093,0.18005],
        [0.99672,0.60977,0.18210],[0.99644,0.59846,0.18449],
        [0.99593,0.58703,0.18721],[0.99517,0.57549,0.19026],
        [0.99419,0.56386,0.19362],[0.99297,0.55214,0.19730],
        [0.99153,0.54036,0.20127],[0.98987,0.52854,0.20552],
        [0.98799,0.51667,0.21004],[0.98590,0.50479,0.21481],
        [0.98360,0.49291,0.21981],[0.98108,0.48104,0.22500],
        [0.97837,0.46920,0.23037],[0.97545,0.45740,0.23590],
        [0.97234,0.44565,0.24155],[0.96904,0.43399,0.24730],
        [0.96555,0.42241,0.25313],[0.96187,0.41093,0.25903],
        [0.95801,0.39958,0.26497],[0.95398,0.38836,0.27095],
        [0.94977,0.37729,0.27691],[0.94538,0.36638,0.28285],
        [0.94084,0.35566,0.28877],[0.93612,0.34513,0.29464],
        [0.93125,0.33482,0.30044],[0.92623,0.32473,0.30617],
        [0.92105,0.31489,0.31180],[0.91572,0.30530,0.31733],
        [0.91024,0.29599,0.32275],[0.90463,0.28696,0.32804],
        [0.89888,0.27824,0.33319],[0.89298,0.26981,0.33819],
        [0.88691,0.26152,0.34308],[0.88066,0.25334,0.34783],
        [0.87422,0.24526,0.35244],[0.86760,0.23730,0.35689],
        [0.86079,0.22945,0.36116],[0.85380,0.22170,0.36527],
        [0.84662,0.21407,0.36921],[0.83926,0.20654,0.37297],
        [0.83172,0.19912,0.37656],[0.82399,0.19182,0.37996],
        [0.81608,0.18462,0.38317],[0.80799,0.17753,0.38620],
        [0.79971,0.17055,0.38904],[0.79125,0.16368,0.39168],
        [0.78260,0.15693,0.39412],[0.77377,0.15028,0.39635],
        [0.76476,0.14374,0.39836],[0.75556,0.13731,0.40016],
        [0.74617,0.13098,0.40172],[0.73661,0.12477,0.40305],
        [0.72686,0.11867,0.40414],[0.71692,0.11268,0.40499],
        [0.70680,0.10680,0.40560],[0.69650,0.10102,0.40595],
        [0.68602,0.09536,0.40604],[0.67535,0.08980,0.40585],
        [0.66449,0.08436,0.40539],[0.65345,0.07902,0.40464],
        [0.64223,0.07380,0.40359],[0.63082,0.06868,0.40226],
        [0.61923,0.06367,0.40062],[0.60746,0.05878,0.39867],
        [0.59550,0.05399,0.39640],[0.58336,0.04931,0.39381],
        [0.57103,0.04474,0.39089],[0.55852,0.04028,0.38763],
        [0.54583,0.03593,0.38403],[0.53295,0.03169,0.38008],
        [0.51989,0.02756,0.37578],[0.50664,0.02354,0.37113],
        [0.49321,0.01963,0.36612],[0.47960,0.01583,0.36074],
        [0.46581,0.01214,0.35499],[0.45184,0.00855,0.34887],
        [0.43768,0.00507,0.34237],[0.42334,0.00170,0.33548],
    ]
    # Fill any gaps with a simple interpolation (some rows above have
    # placeholder text from copy-paste issues — replace with clean data)
    clean = []
    for row in turbo_data:
        try:
            r, g, b = float(row[0]), float(row[1]), float(row[2])
            clean.append([r, g, b])
        except (ValueError, TypeError):
            clean.append(clean[-1] if clean else [0, 0, 0])

    n = len(clean)
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t   = i / 255.0
        idx = t * (n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        f   = idx - lo
        r   = clean[lo][0] * (1 - f) + clean[hi][0] * f
        g   = clean[lo][1] * (1 - f) + clean[hi][1] * f
        b   = clean[lo][2] * (1 - f) + clean[hi][2] * f
        cmap[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return cmap


def _build_turbo_clean() -> np.ndarray:
    """Build turbo colormap from clean keypoints via interpolation."""
    # 32 hand-picked keypoints spanning the turbo ramp
    keys = np.array([
        [0.19, 0.07, 0.23],   # 0   deep purple-blue
        [0.25, 0.30, 0.72],   # 32  blue
        [0.27, 0.47, 0.94],   # 48  blue-cyan
        [0.25, 0.58, 1.00],   # 64  cyan
        [0.22, 0.69, 0.96],   # 80  cyan-teal
        [0.13, 0.79, 0.85],   # 96  teal
        [0.09, 0.87, 0.74],   # 112 teal-green
        [0.12, 0.93, 0.57],   # 128 green
        [0.27, 0.97, 0.41],   # 144 yellow-green
        [0.48, 0.99, 0.30],   # 160 yellow-green
        [0.65, 0.98, 0.23],   # 176 yellow
        [0.80, 0.94, 0.17],   # 184 yellow-orange
        [0.92, 0.85, 0.17],   # 192 orange
        [0.98, 0.72, 0.18],   # 208 orange
        [0.99, 0.56, 0.19],   # 224 orange-red
        [0.97, 0.38, 0.27],   # 240 red
        [0.88, 0.21, 0.33],   # 248 deep red
        [0.75, 0.13, 0.40],   # 255 darkest red
    ], dtype=np.float64)

    cmap = np.zeros((256, 3), dtype=np.uint8)
    n = len(keys)
    for i in range(256):
        t   = i / 255.0 * (n - 1)
        lo  = int(t)
        hi  = min(lo + 1, n - 1)
        f   = t - lo
        rgb = keys[lo] * (1 - f) + keys[hi] * f
        cmap[i] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return cmap

COLORMAP = _build_turbo_clean()


def scalar_to_color(scalars: np.ndarray,
                    shading: np.ndarray | None = None) -> np.ndarray:
    """
    Maps flux scalars to RGB using histogram equalization (stretch to
    actual min/max of this frame) and optional diffuse shading for depth.

    Parameters
    ----------
    scalars  : (N,) float  — flux values [W/m²]
    shading  : (N,) float in [0,1] — per-face diffuse shading factor

    Returns
    -------
    colors : (N, 3) uint8
    """
    q_min = scalars.min()
    q_max = scalars.max()
    span  = q_max - q_min
    if span < 1e-6:
        norm = np.zeros_like(scalars)
    else:
        norm = (scalars - q_min) / span   # 0–1 per this frame

    indices = np.clip((norm * 255).astype(int), 0, 255)
    colors  = COLORMAP[indices].astype(np.float32)  # (N, 3)

    if shading is not None:
        shade = np.clip(shading, 0.0, 1.0)[:, None]
        colors = colors * shade
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    return colors


def compute_shading(faces: list) -> np.ndarray:
    """
    Per-face diffuse shading from a fixed overhead-south light direction.
    Returns (N,) float in [SHADE_AMBIENT, 1.0].
    Provides 3D depth without relying on sun position.
    """
    normals = np.array([f.normal for f in faces], dtype=np.float64)
    dots    = normals @ _SHADE_DIR                        # (N,)
    shade   = SHADE_AMBIENT + SHADE_DIFFUSE * np.clip(dots, 0.0, 1.0)
    return shade.astype(np.float32)


# ─── OBJ parser ───────────────────────────────────────────────────────────────

@dataclass
class Face:
    obj_name:    str
    vertices:    np.ndarray   # (3, 3)
    normal:      np.ndarray   # (3,)
    centroid:    np.ndarray   # (3,)
    area:        float
    absorptivity:float = 1.0


def _obj_to_blender(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], -v[2], v[1]], dtype=np.float64)


def parse_obj(obj_path: Path,
              absorptivity_map: dict[str, float] | None = None) -> list[Face]:
    if absorptivity_map is None:
        absorptivity_map = DEFAULT_ABSORPTIVITY

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
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
                v_idx, vn_idx = [], []
                for p in parts:
                    tokens = p.split("/")
                    v_idx.append(int(tokens[0]) - 1)
                    vn_idx.append(int(tokens[2]) - 1 if len(tokens) > 2
                                  and tokens[2] else None)
                absorptivity = absorptivity_map.get(current_obj, 0.75)
                for i in range(1, len(v_idx) - 1):
                    tri_v = [v_idx[0], v_idx[i], v_idx[i + 1]]
                    tri_n = [vn_idx[0], vn_idx[i], vn_idx[i + 1]]
                    verts_bl = np.array([_obj_to_blender(vertices[j]) for j in tri_v])
                    if all(n is not None for n in tri_n):
                        raw_n = np.mean([normals[j] for j in tri_n], axis=0)
                        n_bl  = _obj_to_blender(raw_n)
                    else:
                        e1 = verts_bl[1] - verts_bl[0]
                        e2 = verts_bl[2] - verts_bl[0]
                        n_bl = np.cross(e1, e2)
                    norm = np.linalg.norm(n_bl)
                    if norm < 1e-10:
                        continue
                    n_bl /= norm
                    centroid = verts_bl.mean(axis=0)
                    e1 = verts_bl[1] - verts_bl[0]
                    e2 = verts_bl[2] - verts_bl[0]
                    area = 0.5 * np.linalg.norm(np.cross(e1, e2))
                    faces.append(Face(obj_name=current_obj, vertices=verts_bl,
                                      normal=n_bl, centroid=centroid, area=area,
                                      absorptivity=absorptivity))
    return faces


# ─── Solar helpers ────────────────────────────────────────────────────────────

def get_solar_irradiance(latitude: float, longitude: float, altitude: float,
                         date_time: str,
                         turbidity: float = DEFAULT_TURBIDITY) -> dict:
    dt_utc = pd.Timestamp(date_time, tz="UTC")
    times  = pd.DatetimeIndex([dt_utc])
    pos    = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    zenith = float(pos["apparent_zenith"].iloc[0])
    azimuth= float(pos["azimuth"].iloc[0])
    elev   = float(pos["apparent_elevation"].iloc[0])
    sun_above   = zenith < 90.0
    in_twilight = 90.0 <= zenith <= CIVIL_TWILIGHT_ZENITH
    if sun_above or in_twilight:
        am_rel = pvlib.atmosphere.get_relative_airmass(zenith)
        am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel)
        cs = pvlib.clearsky.ineichen(
            apparent_zenith=pd.Series([zenith], index=times),
            airmass_absolute=pd.Series([am_abs], index=times),
            linke_turbidity=turbidity)
        dni = float(cs["dni"].iloc[0]) if sun_above else 0.0
        dhi = float(cs["dhi"].iloc[0])
        ghi = float(cs["ghi"].iloc[0])
    else:
        dni = dhi = ghi = 0.0
    return {"dni": dni, "dhi": dhi, "ghi": ghi,
            "zenith": zenith, "azimuth": azimuth, "elevation": elev,
            "sun_above_horizon": sun_above, "in_civil_twilight": in_twilight}


def sun_direction_vector(azimuth_pvlib_deg: float, elevation_deg: float,
                         north_offset_deg: float = NORTH_OFFSET_DEG) -> np.ndarray:
    az_bl  = math.radians(-azimuth_pvlib_deg + north_offset_deg)
    el_rad = math.radians(elevation_deg)
    return np.array([math.sin(az_bl) * math.cos(el_rad),
                     math.cos(az_bl) * math.cos(el_rad),
                     math.sin(el_rad)], dtype=np.float64)


# ─── BVH shadow testing ───────────────────────────────────────────────────────

class BVHNode:
    def __init__(self, faces: list[Face], depth: int = 0):
        self.faces = faces; self.left = None; self.right = None
        self.is_leaf = False
        all_verts = np.vstack([f.vertices for f in faces])
        self.aabb_min = all_verts.min(axis=0) - 1e-4
        self.aabb_max = all_verts.max(axis=0) + 1e-4
        if len(faces) <= 8 or depth > 20:
            self.is_leaf = True; return
        extent = self.aabb_max - self.aabb_min
        axis   = int(np.argmax(extent))
        mid    = np.median([f.centroid[axis] for f in faces])
        left_faces  = [f for f in faces if f.centroid[axis] <= mid]
        right_faces = [f for f in faces if f.centroid[axis] >  mid]
        if not left_faces or not right_faces:
            self.is_leaf = True; return
        self.left  = BVHNode(left_faces,  depth + 1)
        self.right = BVHNode(right_faces, depth + 1)

    def _ray_aabb(self, origin, inv_dir):
        t1 = (self.aabb_min - origin) * inv_dir
        t2 = (self.aabb_max - origin) * inv_dir
        return np.minimum(t1, t2).max() >= max(np.maximum(t1, t2).min(), 1e-4)

    def ray_intersects(self, origin, direction, max_dist=1e9):
        inv_dir = np.where(np.abs(direction) > 1e-12,
                           1.0 / direction, np.sign(direction) * 1e12)
        if not self._ray_aabb(origin, inv_dir):
            return False
        if self.is_leaf:
            for face in self.faces:
                v0,v1,v2 = face.vertices
                e1=v1-v0; e2=v2-v0; h=np.cross(direction,e2); a=e1@h
                if abs(a)<1e-8: continue
                f=1./a; s=origin-v0; u=f*(s@h)
                if u<0 or u>1: continue
                q=np.cross(s,e1); v=f*(direction@q)
                if v<0 or u+v>1: continue
                t=f*(e2@q)
                if 1e-4<t<max_dist: return True
            return False
        return ((self.left  is not None and self.left.ray_intersects(origin, direction, max_dist)) or
                (self.right is not None and self.right.ray_intersects(origin, direction, max_dist)))


# ─── Heat flux computation ────────────────────────────────────────────────────

def compute_heat_flux(
    faces:   list[Face],
    sun_vec: np.ndarray,
    dni:     float,
    dhi:     float,
    bvh:     BVHNode,
    offset:  float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes per-face incident irradiance and absorbed heat flux in one pass.

    Returns
    -------
    incident_scalars : (N,)   — E_i  [W/m²]  before absorptivity
    incident_vectors : (N, 3) — E_i · n̂_i
    absorbed_scalars : (N,)   — q_i = α_i · E_i  [W/m²]
    absorbed_vectors : (N, 3) — q_i · n̂_i
    """
    N = len(faces)
    E = np.zeros(N, dtype=np.float64)   # incident
    q = np.zeros(N, dtype=np.float64)   # absorbed

    for i, face in enumerate(faces):
        n = face.normal
        α = face.absorptivity

        cos_theta = float(np.dot(n, sun_vec))
        if cos_theta > 0 and dni > 0:
            origin    = face.centroid + offset * n
            in_shadow = bvh.ray_intersects(origin, sun_vec)
            direct    = dni * cos_theta * (0.0 if in_shadow else 1.0)
        else:
            direct = 0.0

        cos_tilt = float(n[2])
        sky_vf   = (1.0 + cos_tilt) / 2.0
        diffuse  = dhi * max(0.0, sky_vf)

        E[i] = direct + diffuse
        q[i] = α * E[i]

    normals_arr = np.array([f.normal for f in faces], dtype=np.float64)
    return (E, E[:, None] * normals_arr,
            q, q[:, None] * normals_arr)


# ─── PLY export ───────────────────────────────────────────────────────────────

def export_ply(faces: list[Face], colors: np.ndarray,
               output_path: Path) -> None:
    """
    Exports the mesh as a PLY file with per-face RGB colors.
    colors : (N_faces, 3) uint8
    """
    n_faces = len(faces)
    n_verts = n_faces * 3

    header = [
        "ply", "format ascii 1.0",
        f"element vertex {n_verts}",
        "property float x", "property float y", "property float z",
        f"element face {n_faces}",
        "property list uchar int vertex_indices",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]

    vert_lines = []
    face_lines = []
    for i, face in enumerate(faces):
        base = i * 3
        for v in face.vertices:
            vert_lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        face_lines.append(f"3 {base} {base+1} {base+2} {r} {g} {b}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(header + vert_lines + face_lines) + "\n")
    print(f"    PLY → {output_path.name}")


# ─── PNG renderer ─────────────────────────────────────────────────────────────

def render_flux_png(faces: list[Face], colors: np.ndarray,
                    camera_to_world: np.ndarray,
                    intrinsics: np.ndarray,
                    resolution: tuple[int, int],
                    output_path: Path) -> None:
    """
    Projects coloured mesh onto camera image plane (BlenderProc convention).
    colors : (N_faces, 3) uint8 — pre-computed by scalar_to_color()
    """
    W, H  = resolution
    img   = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf  = np.full((H, W), np.inf, dtype=np.float64)
    # Z_BIAS = 1e-3
    # EDGE_EPS = 0.05 

    c2w = np.array(camera_to_world, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    def project(pt):
        pc    = R @ pt + t
        depth = -pc[2]
        if depth <= 0.01: return None
        u = int(round(fx *  pc[0] / depth + cx))
        v = int(round(fy * -pc[1] / depth + cy))
        return u, v, depth

    def face_z(i):
        return np.mean([( R @ v + t)[2] for v in faces[i].vertices])

    order = sorted(range(len(faces)), key=face_z)

    n_drawn = 0
    for i in order:
        face  = faces[i]
        color = colors[i]
        projs = [project(v) for v in face.vertices]
        if any(p is None for p in projs): continue
        # depth = float(np.mean([p[2] for p in projs]))
        # pts   = [(p[0], p[1]) for p in projs]
        d0, d1, d2 = projs[0][2], projs[1][2], projs[2][2]
        pts = [(p[0], p[1]) for p in projs]
        if not any(0 <= p[0] < W and 0 <= p[1] < H for p in pts): continue

        p0, p1, p2 = pts
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        xmin,xmax = max(0,min(xs)), min(W-1,max(xs))
        ymin,ymax = max(0,min(ys)), min(H-1,max(ys))
        if xmin > xmax or ymin > ymax: continue

        ax,ay = p1[0]-p0[0], p1[1]-p0[1]
        bx,by = p2[0]-p0[0], p2[1]-p0[1]
        denom = ax*by - ay*bx
        if abs(denom) < 1e-8: continue

        EDGE_EPS = 0.02
        Z_BIAS = 1e-3

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                px, py = x - p0[0], y - p0[1]

                u = (px * by - py * bx) / denom
                v = (ax * py - ay * px) / denom
                w = 1.0 - u - v

                if u >= -EDGE_EPS and v >= -EDGE_EPS and w >= -EDGE_EPS:
                    pixel_depth = w * d0 + u * d1 + v * d2

                    if pixel_depth <= zbuf[y, x] + Z_BIAS:
                        zbuf[y, x] = pixel_depth
                        img[y, x, :] = color

        # for y in range(ymin, ymax+1):
        #     for x in range(xmin, xmax+1):
        #         px,py = x-p0[0], y-p0[1]
        #         u = (px*by - py*bx) / denom
        #         v = (ax*py - ay*px) / denom
        #         # if u >= 0 and v >= 0 and (u+v) <= 1:
        #             # if depth < zbuf[y, x]:
        #             #     zbuf[y, x]   = depth
        #             #     img[y, x, :] = color
        # n_drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)
    print(f"    Drew {n_drawn}/{len(faces)} faces → {output_path.name}")


# ─── Colorbar ─────────────────────────────────────────────────────────────────

def save_colorbar(label_min: float, label_max: float,
                  output_path: Path,
                  width: int = 40, height: int = 256) -> None:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        idx = int((1.0 - row / (height - 1)) * 255)
        bar[row, :] = COLORMAP[idx]
    Image.fromarray(bar).save(output_path)
    print(f"  Colorbar → {output_path.name}  ({label_min:.0f}–{label_max:.0f} W/m²)")


# ─── Timestep helpers ─────────────────────────────────────────────────────────

def get_day_timesteps(date: str, latitude: float, longitude: float,
                      interval_minutes: int = 30) -> list[str]:
    """
    Returns UTC timesteps at the given interval bounded by civil twilight.
    interval_minutes: 30 for full-day loop, 5-10 for video frames.
    """
    times = pd.date_range(f"{date} 00:00", f"{date} 23:59",
                          freq="10min", tz="UTC")
    solar = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    lit   = solar[solar["apparent_zenith"] <= CIVIL_TWILIGHT_ZENITH]
    if lit.empty: return []

    freq = f"{interval_minutes}min"

    def _round_up(dt):
        m = interval_minutes
        extra = dt.minute % m
        if extra == 0 and dt.second == 0: return dt.replace(second=0, microsecond=0)
        return (dt + pd.Timedelta(minutes=m-extra)).replace(second=0, microsecond=0)

    def _round_down(dt):
        m = interval_minutes
        return dt.replace(minute=(dt.minute // m) * m, second=0, microsecond=0)

    steps = pd.date_range(_round_up(lit.index[0]), _round_down(lit.index[-1]),
                          freq=freq, tz="UTC")
    return [s.strftime("%Y-%m-%d %H:%M:%S") for s in steps]


# ─── Per-timestep output (commented out — kept for future use) ───────────────
# def _save_flux_outputs(faces, scalars, vectors, shading,
#                        out_dir: Path, label: str,
#                        camera_frames: list, intrinsics: np.ndarray,
#                        resolution: tuple) -> None:
#     """
#     Saves scalars.npy, vectors.npy, heat_flux.ply and PNG renders
#     into out_dir for one flux type (incident or absorbed).
#     Used by normal (non-video) mode — one subfolder per timestep.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     np.save(out_dir / "scalars.npy", scalars)
#     np.save(out_dir / "vectors.npy", vectors)
#     colors = scalar_to_color(scalars, shading)
#     export_ply(faces, colors, out_dir / "heat_flux.ply")
#     for frame in camera_frames:
#         c2w   = np.array(frame["camera_to_world"])
#         idx   = frame["rgb_path"].replace(".png", "")
#         out_p = out_dir / f"{idx}_flux.png"
#         render_flux_png(faces, colors, c2w, intrinsics, resolution, out_p)
#     q_min, q_max = float(scalars.min()), float(scalars.max())
#     save_colorbar(q_min, q_max, out_dir / f"colorbar_{label}.png")
#     print(f"  {label:10s}: {q_min:.1f}–{q_max:.1f} W/m²  "
#           f"mean={scalars.mean():.1f}  "
#           f"total={sum(scalars[i]*faces[i].area for i in range(len(faces))):.0f} W")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _find_nearest_meta(output_path: Path, hhmm: str) -> Path | None:
    """
    Finds the nearest available meta_data.json to the given timestep folder.

    Camera poses are identical across all timesteps — only lighting changes.
    generate_try.py may have run at a coarser interval (e.g. 30 min) while
    heat_flux.py runs at a finer one (e.g. 10 min). This function finds the
    closest folder that has a meta_data.json so PNG renders are always produced.

    Parameters
    ----------
    output_path : Path  — root output folder (same as generate_try.py)
    hhmm        : str   — current timestep label, e.g. "10h30"

    Returns
    -------
    Path to meta_data.json, or None if none exists anywhere.
    """
    # Convert hhmm label to total minutes for distance comparison
    def to_minutes(label: str) -> int:
        h, m = label.split("h")
        return int(h) * 60 + int(m)

    target_min = to_minutes(hhmm)

    # Collect all available meta_data.json files in sibling folders
    candidates = []
    for folder in output_path.iterdir():
        if not folder.is_dir():
            continue
        meta = folder / "meta_data.json"
        if not meta.exists():
            continue
        try:
            dist = abs(to_minutes(folder.name) - target_min)
            candidates.append((dist, meta))
        except (ValueError, AttributeError):
            continue  # skip folders not named as hhmm

    if not candidates:
        return None

    # Return the closest one
    candidates.sort(key=lambda x: x[0])
    dist, best = candidates[0]
    if dist > 0:
        print(f"  ℹ  Using camera poses from {best.parent.name} "
              f"(nearest to {hhmm}, Δ={dist} min)")
    return best


@dataclass
class HeatFluxParams:
    obj_path:         Path  = Path("bproc_generator/data/example/House.obj")
    latitude:         float = tyro.MISSING
    longitude:        float = tyro.MISSING
    altitude:         float = 400.0
    date:             str   = tyro.MISSING
    """UTC date 'YYYY-MM-DD'. Civil twilight bounds computed automatically."""
    turbidity:        float = DEFAULT_TURBIDITY
    north_offset_deg: float = NORTH_OFFSET_DEG
    renders_path:     Path  = tyro.MISSING
    """
    Root folder containing the existing generate_try.py outputs
    (the folder with 10h00/, 10h30/ ... subfolders and their meta_data.json).
    Camera poses are read from here. Existing files are NEVER modified.
    """
    output_path:      Path  = tyro.MISSING
    """
    Destination folder for video frames.
    Will contain:
      output_path/incident/0000.png, 0001.png, ...
      output_path/absorbed/0000.png, 0001.png, ...
      output_path/summary.json
    Can be anywhere — Desktop, Z: drive, etc.
    """
    resolution:       tuple[int, int] = (512, 512)
    interval_minutes: int   = 30
    """Timestep interval in minutes. Use 5–10 for smoother video frames."""
    pose_index:       int   = 11
    """Which camera pose to use (0-based). Default = 11 (frame 0011)."""


@dataclass
class HeatFluxRunner(HeatFluxParams):

    def run(self):
        print(f"\n{'═'*55}")
        print(f"  Solar Heat Flux — video frames")
        print(f"  Date      : {self.date}")
        print(f"  Location  : {self.latitude}°N  {self.longitude}°E  {self.altitude}m")
        print(f"  Interval  : {self.interval_minutes} min")
        print(f"  Pose      : #{self.pose_index}")
        print(f"  Renders   : {self.renders_path}")
        print(f"  Output    : {self.output_path}")
        print(f"  Colormap  : Turbo, per-frame histogram equalization")
        print(f"{'═'*55}\n")

        # ── Step 1: Parse OBJ and build BVH (once) ───────────────────────
        print(f"  Parsing OBJ : {self.obj_path}")
        faces = parse_obj(self.obj_path)
        print(f"  Faces       : {len(faces)}")
        print(f"  Building BVH...")
        bvh = BVHNode(faces)

        # ── Step 2: Pre-compute fixed shading (same for all frames) ──────
        shading = compute_shading(faces)

        # ── Step 3: Get timesteps ─────────────────────────────────────────
        timesteps = get_day_timesteps(
            self.date, self.latitude, self.longitude, self.interval_minutes)
        if not timesteps:
            print("  ⚠  No renderable timesteps.")
            return
        print(f"  Timesteps   : {len(timesteps)}  "
              f"({timesteps[0][11:16]} → {timesteps[-1][11:16]} UTC)\n")

        # ── Step 4: Load camera pose (once, from renders_path) ────────────
        # Search renders_path subfolders for any meta_data.json.
        # All timesteps share identical poses so we only need one file.
        camera_pose  = None   # single 4×4 matrix
        intrinsics   = np.eye(3)

        for folder in sorted(self.renders_path.iterdir()):
            meta = folder / "meta_data.json"
            if not meta.exists():
                continue
            with open(meta) as f:
                rm = json.load(f)
            frames = rm.get("frames", [])
            if self.pose_index >= len(frames):
                print(f"  ⚠  pose_index={self.pose_index} out of range "
                      f"(only {len(frames)} poses available). "
                      f"Using last pose.")
                camera_pose = np.array(frames[-1]["camera_to_world"])
            else:
                camera_pose = np.array(frames[self.pose_index]["camera_to_world"])
            intrinsics = np.array(frames[0]["intrinsics"])
            print(f"  Camera pose : #{self.pose_index} loaded from "
                  f"{folder.name}/meta_data.json  "
                  f"({len(frames)} poses available)")
            break

        if camera_pose is None:
            print("  ⚠  No meta_data.json found in renders_path. "
                  "Cannot render PNGs — only .npy files will be saved.")

        # ── Step 5: Create output folders ────────────────────────────────
        inc_dir = self.output_path / "incident"
        abs_dir = self.output_path / "absorbed"
        inc_dir.mkdir(parents=True, exist_ok=True)
        abs_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 6: Loop over timesteps ───────────────────────────────────
        summary = []

        for frame_idx, date_time in enumerate(timesteps):
            fname = str(frame_idx).zfill(4)
            hhmm  = date_time[11:16]
            print(f"  [{fname}] {hhmm} UTC", end="  ")

            sol = get_solar_irradiance(
                self.latitude, self.longitude, self.altitude,
                date_time, self.turbidity)
            print(f"el={sol['elevation']:+.1f}°  "
                  f"DNI={sol['dni']:.0f}  DHI={sol['dhi']:.0f} W/m²")

            if sol["sun_above_horizon"] or sol["in_civil_twilight"]:
                sv = sun_direction_vector(
                    sol["azimuth"], sol["elevation"], self.north_offset_deg)
                E_sc, _, q_sc, _ = compute_heat_flux(
                    faces, sv, sol["dni"], sol["dhi"], bvh)
            else:
                E_sc = q_sc = np.zeros(len(faces))

            # Use a shared colormap scale: incident is the ceiling.
            # This way absorbed appears visually less intense than incident —
            # windows (α=0.10) stay blue, roof (α=0.90) stays near-red.
            # Both images are directly comparable.
            shared_max = float(E_sc.max()) if E_sc.max() > 0 else 1.0
            shared_min = float(E_sc.min())

            def color_shared(scalars):
                span = shared_max - shared_min
                if span < 1e-6:
                    norm = np.zeros_like(scalars)
                else:
                    norm = np.clip((scalars - shared_min) / span, 0, 1)
                indices = (norm * 255).astype(int)
                colors  = COLORMAP[indices].astype(np.float32)
                colors  = colors * shading[:, None]
                return np.clip(colors, 0, 255).astype(np.uint8)

            # Render incident (no absorptivity)
            colors_inc = color_shared(E_sc)
            if camera_pose is not None:
                render_flux_png(faces, colors_inc, camera_pose, intrinsics,
                                self.resolution, inc_dir / f"{fname}.png")

            # Render absorbed (with absorptivity — same scale as incident)
            colors_abs = color_shared(q_sc)
            if camera_pose is not None:
                render_flux_png(faces, colors_abs, camera_pose, intrinsics,
                                self.resolution, abs_dir / f"{fname}.png")

            # Per-frame metadata
            summary.append({
                "frame":            frame_idx,
                "filename":         f"{fname}.png",
                "date_time":        date_time,
                "solar_elevation":  sol["elevation"],
                "solar_azimuth":    sol["azimuth"],
                "DNI_Wm2":          sol["dni"],
                "DHI_Wm2":          sol["dhi"],
                "incident_max":     float(E_sc.max()),
                "absorbed_max":     float(q_sc.max()),
            })

        # ── Step 7: Write summary ─────────────────────────────────────────
        with open(self.output_path / "summary.json", "w") as f:
            json.dump({
                "date":             self.date,
                "latitude":         self.latitude,
                "longitude":        self.longitude,
                "pose_index":       self.pose_index,
                "interval_minutes": self.interval_minutes,
                "total_frames":     len(timesteps),
                "colormap":         "turbo_histogram_eq",
                "ffmpeg_command":   (
                    f"ffmpeg -r 10 -i {self.output_path}/incident/%04d.png "
                    f"-pix_fmt yuv420p {self.output_path}/incident.mp4"
                ),
                "frames":           summary,
            }, f, indent=4)

        print(f"\n{'═'*55}")
        print(f"  Done. {len(timesteps)} frames.")
        print(f"  incident/ → {inc_dir}")
        print(f"  absorbed/ → {abs_dir}")
        print(f"  To make video:")
        print(f"  ffmpeg -r 10 -i \"{inc_dir}\\%04d.png\" "
              f"-pix_fmt yuv420p \"{self.output_path}\\incident.mp4\"")
        print(f"{'═'*55}\n")


def main():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(HeatFluxRunner).run()


if __name__ == "__main__":
    main()
import blenderproc as bproc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import tyro

import os
import math
import pvlib
import pandas as pd
import bpy

sys.path.extend([str(Path(__file__).resolve().parents[2])])
from buildnet3d.utils.utils import build_translation

# Physical calibration scalars
K_SUN: float = 0.0075
NISHITA_FILL_FACTOR: float = 15.0
SHADOW_FILL_BOOST: float = 1.0
CIVIL_TWILIGHT_ZENITH: float = 96.0
DEFAULT_TURBIDITY: float = 3.0
NORTH_OFFSET_DEG: float = 180.0

@dataclass
class RenderParams:
    """Parameters for rendering synthetic building images."""
    
    load_scene: Path = Path("bproc_generator/data/example/House.obj")
    """Path to the segmented building OBJ file"""
    output_path: Path = Path("outputs/generated")
    """Output directory for rendered assets"""
    resolution: tuple[int, int] = (512, 512)
    """Rendering resolution (width, height)"""
    num_frames: int = 10
    """Number of camera frames to render"""
    enable_transparency: bool = False
    """Enable alpha channel in output images"""
    alpha: float = 1.0
    """Alpha value for transparent regions (0.0-1.0)"""
    
    # Camera parameters
    load_camera_path: Path = None
    """Optional path to predefined camera trajectory"""
    bound_thresh: int = 2
    """Pixel threshold for boundary object detection"""
    blank_thresh: int = 20
    """Pixel threshold for blank boundary detection"""
    retry_count: int = 30
    """Max attempts to find valid camera pose"""
    radius: float = 25
    """Camera orbit radius from point of interest"""
    theta_range: tuple[float, float] = (0.0, 2 * np.pi)
    """Azimuth angle range (radians)"""
    phi_range: tuple[float, float] = (0.0, 1.178)
    """Elevation angle range (radians)"""
    gamma_range: tuple[float, float] = (0.0, 0.0)
    """In-plane rotation range (radians)"""
    dist_thresh: float = 2.0
    """Minimum distance between camera poses to avoid overlap"""
    
    # Camera adjustment parameters
    delta_radius: tuple[float, float] = (0.5, 2.0)
    """Radius adjustment range during pose refinement"""
    delta_theta: tuple[float, float] = (0.05, 0.10)
    """Azimuth adjustment range during pose refinement"""
    delta_phi: tuple[float, float] = (0.05, 0.10)
    """Elevation adjustment range during pose refinement"""
    delta_poi: tuple[float, float] = (0.0, 0.5)
    """Point of interest adjustment range during pose refinement"""

    # Location and time
    latitude:  float = tyro.MISSING
    """Building location latitude [decimal degrees]."""
    longitude: float = tyro.MISSING
    """Building location longitude [decimal degrees]."""
    altitude:  float = 400.0
    """Building altitude above sea level [m].  Used by the Ineichen model."""
    date_time: str = tyro.MISSING
    """UTC date-time for sun position: 'YYYY-MM-DD HH:MM:SS'."""
    turbidity: float = DEFAULT_TURBIDITY
    """Linke turbidity factor (2 = very clear, 3 = typical, 5 = hazy)."""

    # Lighting calibration
    k_sun: float = K_SUN
    """SUN lamp energy per W/m² of DNI.  The sole exposure knob — calibrate once."""
    nishita_fill_factor: float = NISHITA_FILL_FACTOR
    """Nishita env-light fill relative to SUN lamp per unit.  Scene-independent."""
    shadow_fill_boost: float = SHADOW_FILL_BOOST
    """Extra sky fill multiplier at low elevations for golden-hour shadow softness."""

    # Sky options
    use_nishita_sky: bool = True
    """Enable the Nishita procedural sky (models DHI — diffuse sky radiation)."""
    air_density:  float = 1.0
    """Nishita air density parameter."""
    dust_density: float = 0.3
    """Nishita dust/aerosol density parameter."""

    # Sun lamp options
    use_sun: bool = True
    """Enable the directional SUN lamp (models DNI — direct beam radiation)."""
    north_offset_deg: float = NORTH_OFFSET_DEG
    """
    Azimuth rotation [°] that maps the building's south-facing façade to
    geographic South.  For House.obj this is 180° (see coordinate system
    documentation at the top of this file).  Adjust for other buildings.
    """


def get_clear_sky_irradiance(
    latitude: float,
    longitude: float,
    altitude: float,
    date_time: str,
    turbidity: float = DEFAULT_TURBIDITY,
) -> dict:
    """
    Returns physically accurate clear-sky irradiance components using the
    Ineichen model (pvlib).
 
    Parameters
    ----------
    latitude, longitude : float
        Geographic coordinates of the building [decimal degrees].
    altitude : float
        Elevation above sea level [m].  Used by the Ineichen model.
    date_time : str
        UTC date-time string "YYYY-MM-DD HH:MM:SS".
    turbidity : float
        Linke turbidity factor (2 = very clear, 3 = typical, 5 = hazy).
 
    Returns
    -------
    dict
        ghi              : Global Horizontal Irradiance        [W/m²]
        dni              : Direct Normal Irradiance            [W/m²]
        dni_effective    : DNI, zeroed during civil twilight   [W/m²]
        dhi              : Diffuse Horizontal Irradiance       [W/m²]
        ghi_correct      : DNI·cos(θ_z) + DHI                 [W/m²]
        zenith           : apparent solar zenith angle         [°]
        azimuth          : solar azimuth (N=0, E=90, CW)      [°]
        elevation        : apparent solar elevation            [°]
        airmass          : relative optical airmass
        in_civil_twilight: True if 90° ≤ zenith ≤ 96°
        sun_above_horizon: True if zenith < 90°
    """
    dt_utc = pd.Timestamp(date_time, tz="UTC")
    times = pd.DatetimeIndex([dt_utc])
 
    solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    zenith    = float(solar_pos["apparent_zenith"].iloc[0])
    azimuth   = float(solar_pos["azimuth"].iloc[0])
    elevation = float(solar_pos["apparent_elevation"].iloc[0])
 
    in_civil_twilight = 90.0 <= zenith <= CIVIL_TWILIGHT_ZENITH
    sun_above_horizon = zenith < 90.0
 
    if sun_above_horizon or in_civil_twilight:
        airmass_rel = pvlib.atmosphere.get_relative_airmass(zenith)
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel)
        cs = pvlib.clearsky.ineichen(
            apparent_zenith=pd.Series([zenith], index=times),
            airmass_absolute=pd.Series([airmass_abs], index=times),
            linke_turbidity=turbidity,
        )
        ghi = float(cs["ghi"].iloc[0])
        dni = float(cs["dni"].iloc[0])
        dhi = float(cs["dhi"].iloc[0])
 
        # During civil twilight the sun disc is below the horizon:
        # DNI contributes nothing to horizontal surfaces but the sky is still lit.
        dni_effective = 0.0 if in_civil_twilight else dni
 
        cos_zenith  = math.cos(math.radians(zenith))
        ghi_correct = max(0.0, dni * cos_zenith + dhi)
    else:
        ghi = dni = dhi = ghi_correct = 0.0
        dni_effective = 0.0
        airmass_rel = None
 
    return {
        "ghi":               ghi,
        "dni":               dni,
        "dni_effective":     dni_effective,
        "dhi":               dhi,
        "ghi_correct":       ghi_correct,
        "zenith":            zenith,
        "azimuth":           azimuth,
        "elevation":         elevation,
        "airmass":           airmass_rel,
        "in_civil_twilight": in_civil_twilight,
        "sun_above_horizon": sun_above_horizon,
    }

def sun_color_from_elevation(elevation_deg: float) -> list:
    """
    Returns a linear-light RGB sun color as a function of solar elevation.
 
    The piecewise ramp is calibrated against measured correlated colour
    temperature (CCT) data for clear-sky conditions:
 
        elevation ≤  0° :  deep red/orange   CCT ~2000 K  (civil twilight)
        elevation    3° :  orange             CCT ~2500 K
        elevation    8° :  golden yellow      CCT ~3500 K
        elevation   15° :  warm white         CCT ~4500 K
        elevation   30° :  daylight white     CCT ~5500 K
        elevation ≥ 50° :  neutral sky white  CCT ~6000 K
 
    R is always normalised to 1.0; Blender multiplies energy × color.
    """
    el = max(elevation_deg, -6.0)
 
    if el <= 0:
        t = (el + 6.0) / 6.0
        r, g, b = 1.0, 0.18 + 0.12 * t, 0.02 + 0.06 * t
 
    elif el <= 3:
        t = el / 3.0
        r, g, b = 1.0, 0.30 + 0.08 * t, 0.08 + 0.08 * t
 
    elif el <= 8:
        t = (el - 3) / 5.0
        r, g, b = 1.0, 0.38 + 0.17 * t, 0.16 + 0.14 * t
 
    elif el <= 15:
        t = (el - 8) / 7.0
        r, g, b = 1.0, 0.55 + 0.20 * t, 0.30 + 0.20 * t
 
    elif el <= 30:
        t = (el - 15) / 15.0
        r, g, b = 1.0, 0.75 + 0.17 * t, 0.50 + 0.32 * t
 
    elif el <= 50:
        t = (el - 30) / 20.0
        r, g, b = 1.0, 0.92 + 0.06 * t, 0.82 + 0.14 * t
 
    else:
        r, g, b = 1.0, 1.0, 1.0
 
    return [round(r, 3), round(g, 3), round(b, 3)]

def pvlib_to_blender_azimuth(azimuth_pvlib_deg: float,
                              north_offset_deg: float) -> float:
    """
    Converts a pvlib meteorological azimuth to a unified Blender azimuth
    (counter-clockwise degrees from Blender +Y, after north-offset correction).
 
    This single formula is used for BOTH the Nishita sky node and the SUN
    lamp, which guarantees that the sky gradient and lamp shadows are always
    collinear (no North/South inversion).
 
    Parameters
    ----------
    azimuth_pvlib_deg : float
        Solar azimuth from pvlib (N=0, E=90, S=180, W=270, clockwise).
    north_offset_deg : float
        Degrees to rotate so that the building's chosen south-facing façade
        is aligned with geographic South.  For House.obj: 180°.
 
    Returns
    -------
    float
        Azimuth in degrees, counter-clockwise from Blender +Y.
    """
    return -azimuth_pvlib_deg + north_offset_deg

def solar_to_blender_sun_rotation(azimuth_pvlib_deg: float,
                                   elevation_deg: float,
                                   north_offset_deg: float) -> tuple:
    """
    Returns the XYZ Euler rotation tuple for a Blender SUN lamp.
 
    The lamp's rotation_mode must be set to 'XYZ' before assigning this.
    At rotation (0, 0, 0) the lamp illuminates straight down (−Z world).
    Y rotation tilts the beam up toward the horizon; Z rotation spins it
    around the vertical axis to the correct azimuth.
 
    Returns
    -------
    tuple(float, float, float)
        (0.0, elevation_rad, azimuth_rad) — ready for rotation_euler.
    """
    az_blender = pvlib_to_blender_azimuth(azimuth_pvlib_deg, north_offset_deg)
    elevation_rad = math.radians(elevation_deg)
    azimuth_rad   = math.radians(az_blender)
    return (0.0, elevation_rad, azimuth_rad)

def solar_to_nishita_rotation(azimuth_pvlib_deg: float,
                               north_offset_deg: float) -> float:
    """
    Returns the sun_rotation value (radians) for Blender's ShaderNodeTexSky
    (Nishita model).
 
    Nishita's sun_rotation is 0 at +Y and increases counter-clockwise, which
    is the same convention as pvlib_to_blender_azimuth, so the mapping is a
    direct conversion with no extra sign flip.
 
    Returns
    -------
    float
        sun_rotation in radians.
    """
    az_blender = pvlib_to_blender_azimuth(azimuth_pvlib_deg, north_offset_deg)
    return math.radians(az_blender)



@dataclass
class BlenderProcRenderer(RenderParams):
    """Generates synthetic building imagery with BlenderProc"""
    
    def __post_init__(self):     
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata: dict = {
            "camera_model" : "OPEN_CV",
            "width": self.resolution[0],
            "height": self.resolution[1],
            "has_mono_prior": False,
            "has_foreground_mask": False,
            "has_sparse_sfm_points": False,
            "scene_box" : {},
            "frames": [],
        }
        
        # Initialize rendering pipeline
        bproc.init()
        #self.scene_objects = bproc.loader.load_obj(str(self.load_scene))

        # Temporarily change working directory to the OBJ folder so that
        # Blender resolves MTL texture paths (House_Diff_5k.png, etc.)
        # relative to the OBJ file rather than the process working directory.
        obj_path = self.load_scene.resolve()
        _cwd = os.getcwd()
        os.chdir(obj_path.parent)
        self.scene_objects = bproc.loader.load_obj(str(obj_path))
        os.chdir(_cwd)

        self.bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.scene_objects)

        self._assign_categories()
        self._setup_camera()
        self._setup_lighting()
        
        self.color_map = self._get_color_map()
        self.camera_idx = 0

    def _assign_categories(self):
        """Assigns semantic IDs to building components"""
        category_ids = [1, 3, 1, 1, 5, 4, 1, 2]  # Wall, Window, etc.
        for obj, cid in zip(self.scene_objects, category_ids):
            obj.set_cp("category_id", cid)
    
    def _setup_camera(self):
        """Initializes camera configuration"""
        self.camera_list: list[np.ndarray] = []
        bproc.camera.set_resolution(*self.resolution)
    
    def _setup_lighting(self):
        """Configures environment lighting"""
        irr = get_clear_sky_irradiance(
            self.latitude, self.longitude, self.altitude,
            self.date_time, self.turbidity,
        )
        self.irradiance = irr
 
        zenith    = irr["zenith"]
        azimuth   = irr["azimuth"]
        elevation = irr["elevation"]
 
        print(f"\n{'─' * 55}")
        print(f"  Date/time       : {self.date_time} UTC")
        print(f"  Solar position  : elevation {elevation:.1f}°  "
              f"zenith {zenith:.1f}°  azimuth {azimuth:.1f}°")
        print(f"  DNI             : {irr['dni']:.1f} W/m²")
        print(f"  DHI             : {irr['dhi']:.1f} W/m²")
        print(f"  GHI             : {irr['ghi_correct']:.1f} W/m²  "
              f"(= DNI·cos θ + DHI)")
        if irr["in_civil_twilight"]:
            print(f"  ⚠  Civil twilight — sun disc below horizon, no SUN lamp")
        print(f"{'─' * 55}\n")
 
        # Nishita procedural sky (DHI component)
        if self.use_nishita_sky and (irr["sun_above_horizon"] or irr["in_civil_twilight"]):
            self._setup_nishita_sky(zenith, azimuth, elevation, irr["dhi"])
        else:
            self._setup_night_sky()
 
        # Directional SUN lamp (DNI component)
        if self.use_sun and irr["sun_above_horizon"]:
            self._setup_sun_lamp(azimuth, zenith, elevation, irr["dni_effective"])
    
    def _setup_nishita_sky(self, zenith: float, azimuth: float,
                            elevation: float, dhi: float):
        """Configures Blender's Nishita sky texture driven by pvlib DHI."""
        world = bpy.data.worlds["World"]
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
 
        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type      = "NISHITA"
        sky.sun_elevation = math.radians(max(elevation, -6.0))  # clamp civil twilight
        sky.sun_rotation  = solar_to_nishita_rotation(azimuth, self.north_offset_deg)
        sky.altitude      = self.altitude
        sky.air_density   = self.air_density
        sky.dust_density  = self.dust_density
 
        bg  = nodes.new("ShaderNodeBackground")
        out = nodes.new("ShaderNodeOutputWorld")
        links.new(sky.outputs[0], bg.inputs[0])
        links.new(bg.outputs[0], out.inputs[0])
 
        # Golden-hour fill boost: linearly ramp from shadow_fill_boost at
        # horizon to 1.0 at 15° so that noon lighting is unaffected.
        if elevation < 15.0:
            t     = max(elevation, 0.0) / 15.0
            boost = self.shadow_fill_boost * (1.0 - t) + 1.0 * t
        else:
            boost = 1.0
 
        sky_strength = (dhi * self.k_sun / self.nishita_fill_factor) * boost
        bg.inputs[1].default_value = sky_strength
        self.sky_energy_actual = dhi
 
        print(f"  Nishita sky     : DHI={dhi:.1f} W/m²  el={elevation:.1f}°  "
              f"boost={boost:.2f}  → strength={sky_strength:.4f}")
        print(f"  Nishita rotation: {math.degrees(sky.sun_rotation):.1f}° "
              f"(Blender CCW from +Y)")
    
    def _setup_night_sky(self):
        """Sets the world background to a near-black sky for night scenes."""
        world = bpy.data.worlds["World"]
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()
 
        bg  = nodes.new("ShaderNodeBackground")
        out = nodes.new("ShaderNodeOutputWorld")
        bg.inputs[0].default_value = (0.005, 0.005, 0.015, 1.0)  # faint blue-black
        bg.inputs[1].default_value = 0.02
        world.node_tree.links.new(bg.outputs[0], out.inputs[0])
 
        self.sky_energy_actual = 0.0
        print("  Night sky       : minimal ambient (no solar irradiance)")

    def _setup_sun_lamp(self, azimuth: float, zenith: float,
                         elevation: float, dni: float):
        """
        Creates and configures the directional SUN lamp from pvlib DNI.
 
        The lamp azimuth uses the same pvlib_to_blender_azimuth() helper as
        the Nishita node, guaranteeing that shadows always align with the sky
        gradient — the two components can never point in different directions.
        """
        rotation = solar_to_blender_sun_rotation(azimuth, elevation,
                                                  self.north_offset_deg)
        color  = sun_color_from_elevation(elevation)
        energy = dni * self.k_sun
 
        self.sun_energy_actual = dni
        self.sun_color_actual  = color
 
        sun = bproc.types.Light()
        sun.set_type("SUN")
        sun.set_energy(energy)
        sun.set_color(color)
        sun.blender_obj.rotation_mode  = "XYZ"
        sun.blender_obj.rotation_euler = rotation
 
        dhi = self.sky_energy_actual or 0.0
        cos_z = math.cos(math.radians(zenith))
        ghi   = dni * cos_z + dhi
 
        print(f"  SUN lamp        : DNI={dni:.1f} W/m²  → energy={energy:.4f}")
        print(f"  GHI (correct)   : {ghi:.1f} W/m²  "
              f"(= {dni:.1f}·cos({zenith:.1f}°) + {dhi:.1f})")
        print(f"  Color           : R={color[0]}  G={color[1]}  B={color[2]}")
        print(f"  Lamp rotation   : Y={math.degrees(rotation[1]):.1f}°  "
              f"Z={math.degrees(rotation[2]):.1f}°")


    @staticmethod
    def _get_color_map() -> dict[str, list[float]]:
        """Provides semantic ID to RGB color mapping"""
        return {
            "0": [0, 0, 0],        # Background
            "1": [175, 200, 0],     # Wall
            "2": [0, 200, 200],     # Door
            "3": [125, 0, 200],     # Window
            "4": [175, 0, 200],     # Roof
            "5": [0, 50, 200],      # Chimney
            "6": [0, 200, 50],      # Vegetation
            "7": [150, 50, 50],     # Vehicle
            "8": [50, 175, 50],     # Furniture
            "9": [50, 50, 175],     # Misc
        }

    def _check_boundary(self, camera_pose: np.ndarray) -> tuple[list[bool], dict[str, list[bool]]]:
        """
        Validates object visibility within frame boundaries.
        Returns:
            tuple: [boundary_valid, content_valid] flags and per-edge detection results
        """
        bproc.camera.add_camera_pose(camera_pose, self.camera_idx)
        depth = bproc.camera.depth_via_raytracing(self.bvh_tree)

        # Edge analysis
        boundaries = {
            "top": [
                np.all(depth[:self.bound_thresh, :] == np.inf), 
                not np.all(depth[:self.blank_thresh, :] == np.inf),
            ],
            "bottom": [
                np.all(depth[-self.bound_thresh:, :] == np.inf),
                not np.all(depth[-self.blank_thresh:, :] == np.inf),
            ],
            "left": [
                np.all(depth[:, :self.bound_thresh] == np.inf),
                not np.all(depth[:, :self.blank_thresh] == np.inf),
            ],
            "right": [
                np.all(depth[:, -self.bound_thresh:] == np.inf),
                not np.all(depth[:, -self.blank_thresh:] == np.inf),
            ],
        }
        
        # Validation flags
        bound_valid = sum([v[0] for v in boundaries.values()]) == 4
        content_valid = sum([v[1] for v in boundaries.values()]) >= 3
        
        # Special case handling for horizontal/vertical coverage
        if boundaries["top"][1] and boundaries["bottom"][1]:
            content_valid = True
        if boundaries["left"][1] and boundaries["right"][1]:
            content_valid = True

        return [bound_valid, content_valid], boundaries

    def generate_camera_pose(self, point_of_interest: np.ndarray):
        """Generates valid camera pose through iterative refinement"""
        def random_camera_params():
            """Generates randomized spherical coordinates"""
            theta = np.random.uniform(self.theta_range[0], self.theta_range[1])
            phi = np.random.uniform(self.phi_range[0], self.phi_range[1])
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            # Perturb point of interest
            poi = point_of_interest + np.random.uniform(
                -self.delta_poi[1], self.delta_poi[1], size=3
            )
            return theta, phi, gamma, poi

        print(f" ----- Generating Camera Sample {self.camera_idx} ----- ")
        theta, phi, gamma, poi = random_camera_params()
        radius = self.radius
        retries = 0

        # Camera pose search loop
        while True:
            print(f"Retry {retries} for camera ID {self.camera_idx}", end="\r")
            location = build_translation(radius, theta, phi)
            rotation = bproc.camera.rotation_from_forward_vec(
                poi - location, inplane_rot=gamma
            )
            camera_pose = bproc.math.build_transformation_mat(location, rotation)

            # Boundary validation and adjustment
            (bound_valid, content_valid), boundaries = self._check_boundary(camera_pose)
            
            # Radius adjustment
            if not bound_valid:
                radius += np.random.uniform(*self.delta_radius)
            if not content_valid:
                radius -= np.random.uniform(*self.delta_radius)
                
            # Directional adjustments
            if not boundaries["top"][0] or not boundaries["bottom"][1]:
                poi[2] += np.random.uniform(*self.delta_poi)
                phi += np.random.uniform(*self.delta_phi)
            if not boundaries["bottom"][0] or not boundaries["top"][1]:
                poi[2] -= np.random.uniform(*self.delta_poi)
                phi -= np.random.uniform(*self.delta_phi)
            if not boundaries["left"][0] or not boundaries["right"][1]:
                theta -= np.random.uniform(*self.delta_theta)
            if not boundaries["right"][0] or not boundaries["left"][1]:
                theta += np.random.uniform(*self.delta_theta)
                
            # Pose acceptance criteria
            if bound_valid and content_valid:
                if all(np.linalg.norm(camera_pose[:3, 3] - p[:3, 3]) > self.dist_thresh
                       for p in self.camera_list):
                    self._add_camera_pose(camera_pose)
                    return
            
            retries += 1
            # Retry limit reached, reset parameters
            if retries >= self.retry_count:
                theta, phi, gamma, poi = random_camera_params()
                radius = self.radius
                retries = 0

    def _add_camera_pose(self, pose: np.ndarray):
        """Registers valid camera pose to pipeline"""
        bproc.camera.add_camera_pose(pose, self.camera_idx)
        self.camera_list.append(pose)
        self.metadata["frames"].append({
            "rgb_path": f"{self.camera_idx:04d}.png",
            "segmentation_path": f"{self.camera_idx:04d}_mask.png",
            "camera_to_world": pose.tolist(),
            "intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
        })
        self.camera_idx += 1

    def save_images(self):
        """Converts HDF5 renders to standard image formats"""
        for subdir in ["images", "normals", "depths", "semantics", "instances"]:
            (self.output_path / subdir).mkdir(exist_ok=True)

        # Process each frame
        for i in range(self.camera_idx):
            with h5py.File(self.output_path / f"{i}.hdf5", "r") as f:
                # Process RGB
                rgb = np.array(f["colors"][:])
                if self.enable_transparency:
                    alpha = np.full((*rgb.shape[:2], 1), int(self.alpha * 255), dtype=np.uint8)
                    rgb = np.concatenate([rgb[..., :3], alpha], axis=-1)
                Image.fromarray(rgb).save(self.output_path / "images" / f"{i:04d}.png")
                
                # Process normals
                normal = (f["normals"][:] * 255).astype(np.uint8)
                Image.fromarray(normal).save(self.output_path / "normals" / f"{i:04d}_normal.png")
                
                # Process depth
                depth = (f["depth"][:] * 1000).astype(np.uint16)
                Image.fromarray(depth).save(self.output_path / "depths" / f"{i:04d}_depth.png")
                
                # Process semantics
                semantic = f["category_id_segmaps"][:]
                semantics = np.zeros((*semantic.shape, 3), dtype=np.uint8)
                for j, color in self.color_map.items():
                    semantics[semantic == int(j)] = color
                Image.fromarray(semantics).save(self.output_path / "semantics" / f"{i:04d}_mask.png")
                
                # Process instances
                instance = f["instance_segmaps"][:]
                instances = np.zeros((*instance.shape, 3), dtype=np.uint8)
                for j, color in self.color_map.items():
                    instances[instance == int(j)] = color
                Image.fromarray(instances).save(self.output_path / "instances" / f"{i:04d}.png")
            
            # Cleanup HDF5
            (self.output_path / f"{i}.hdf5").unlink()
            
    def save_metadata(self):
        irr   = self.irradiance
        dhi   = self.sky_energy_actual or 0.0
        dni   = self.sun_energy_actual or 0.0
        zenith = irr.get("zenith")
        cos_z  = math.cos(math.radians(zenith)) if zenith is not None else 0.0
 
        self.metadata["sun"] = {
            "azimuth":             irr.get("azimuth"),
            "zenith":              zenith,
            "elevation":           irr.get("elevation"),
            "DNI_Wm2":             irr.get("dni"),
            "DHI_Wm2":             dhi,
            "GHI_Wm2":             dni * cos_z + dhi,
            "GHI_pvlib_Wm2":       irr.get("ghi"),
            "airmass":             irr.get("airmass"),
            "turbidity":           self.turbidity,
            "in_civil_twilight":   irr.get("in_civil_twilight"),
            "sun_color":           self.sun_color_actual,
            "k_sun":               self.k_sun,
            "nishita_fill_factor": self.nishita_fill_factor,
            "shadow_fill_boost":   self.shadow_fill_boost,
            "blender_sun_energy":  (dni * self.k_sun) if irr.get("sun_above_horizon") else 0.0,
            "blender_sky_strength":(dhi * self.k_sun / self.nishita_fill_factor),
            "north_offset_deg":    self.north_offset_deg,
            "date_time":           self.date_time,
            "latitude":            self.latitude,
            "longitude":           self.longitude,
            "altitude":            self.altitude,
        }
        with open(self.output_path / "meta_data.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

    def run(self):
        """Main rendering pipeline execution"""
        poi = bproc.object.compute_poi(self.scene_objects)
        for _ in range(self.num_frames):
            self.generate_camera_pose(poi)

        # Configure render outputs
        bproc.renderer.set_output_format(enable_transparency=self.enable_transparency)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])
        
        # Execute rendering and save results
        render_data = bproc.renderer.render()
        bproc.writer.write_hdf5(str(self.output_path), render_data)
        self.save_images()
        self.save_metadata()

def main():
    """Command-line entry point for rendering pipeline"""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(BlenderProcRenderer).run()
    
if __name__ == "__main__":
    main()
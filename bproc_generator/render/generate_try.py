# import blenderproc as bproc
# import json
# import sys
# from dataclasses import dataclass
# from pathlib import Path

# import h5py
# import numpy as np
# from PIL import Image
# import tyro

# import math
# import pvlib
# import pandas as pd
# import pytz
# import bpy

# sys.path.extend([str(Path(__file__).resolve().parents[2])])
# from buildnet3d.utils.utils import build_translation


# # ---------------------------------------------------------------------------
# # Physical calibration scalars  (W/m²  →  Blender radiometric energy)
# #
# # These are determined ONCE by rendering a Lambertian white sphere under
# # known irradiance and matching pixel luminance to the photometric ground
# # truth.  They encode the sensor/film exposure of your render setup and
# # should NOT change between scenes.
# #
# # Starting values below are calibrated for Blender Cycles with filmic
# # tone-mapping, 512×512, no additional exposure node.
# # Tweak K_SUN / K_SKY together (keep ratio ~4:1) if renders look over/under
# # exposed — the ratio controls shadow softness vs ambient fill.
# #
# # Rule of thumb:
# #   DNI ≈ 900 W/m² at noon  →  Blender SUN energy ≈ 5–8  (K_SUN ≈ 0.006–0.009)
# #   DHI ≈ 100 W/m² at noon  →  Nishita strength  ≈ 0.3–0.6 (K_SKY ≈ 0.003–0.006)
# # ---------------------------------------------------------------------------
# K_SUN: float = 0.3   # Blender SUN lamp energy per W/m² of DNI
# K_SKY: float = 0.001   # Nishita background strength per W/m² of DHI

# # Civil twilight ends at zenith 96° — sky is still visibly illuminated
# CIVIL_TWILIGHT_ZENITH: float = 96.0

# # Turbidity for Ineichen clear-sky model (Linke turbidity factor)
# # 2.0 = very clear mountain air, 3.0 = typical European clear day,
# # 4.5 = hazy urban, 6+ = very hazy.  Use pvlib.clearsky.lookup_linke_turbidity
# # for location-aware values.
# DEFAULT_TURBIDITY: float = 3.0


# # ---------------------------------------------------------------------------
# # pvlib helpers
# # ---------------------------------------------------------------------------

# def get_clear_sky_irradiance(
#     latitude: float,
#     longitude: float,
#     altitude: float,
#     date_time: str,
#     turbidity: float = DEFAULT_TURBIDITY,
# ) -> dict:
#     """
#     Returns physically accurate clear-sky irradiance components using the
#     Ineichen model (pvlib).

#     Returns
#     -------
#     dict with keys:
#         ghi   : Global Horizontal Irradiance  [W/m²]
#         dni   : Direct Normal Irradiance      [W/m²]
#         dhi   : Diffuse Horizontal Irradiance [W/m²]
#         ghi_correct : DNI·cos(θ_z) + DHI     [W/m²]  (exact formula)
#         zenith      : apparent solar zenith   [°]
#         azimuth     : solar azimuth (N=0, E=90, clockwise) [°]
#         elevation   : solar elevation         [°]
#         airmass     : relative optical airmass
#         in_civil_twilight : True if sun is below horizon but < 96° zenith
#     """
#     location = pvlib.location.Location(
#         latitude=latitude,
#         longitude=longitude,
#         altitude=altitude,
#         tz=pvlib.location.Location(latitude, longitude).tz,
#     )

#     tz_local = pytz.timezone(location.tz)
#     dt_local = pd.Timestamp(date_time, tz=tz_local)
#     dt_utc = dt_local.tz_convert("UTC")
#     times = pd.DatetimeIndex([dt_utc])

#     solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
#     zenith = float(solar_pos["apparent_zenith"].iloc[0])
#     azimuth = float(solar_pos["azimuth"].iloc[0])
#     elevation = float(solar_pos["apparent_elevation"].iloc[0])

#     in_civil_twilight = 90.0 <= zenith <= CIVIL_TWILIGHT_ZENITH
#     sun_above_horizon = zenith < 90.0

#     if sun_above_horizon or in_civil_twilight:
#         # Ineichen clear-sky model
#         airmass_rel = pvlib.atmosphere.get_relative_airmass(zenith)
#         airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel)
#         cs = pvlib.clearsky.ineichen(
#             apparent_zenith=pd.Series([zenith], index=times),
#             airmass_absolute=pd.Series([airmass_abs], index=times),
#             linke_turbidity=turbidity,
#         )
#         ghi = float(cs["ghi"].iloc[0])
#         dni = float(cs["dni"].iloc[0])
#         dhi = float(cs["dhi"].iloc[0])

#         # During civil twilight the sun disc is below the horizon:
#         # DNI contributes nothing to horizontal surfaces but the sky is
#         # still lit. We zero DNI for the lamp but keep DHI for the sky.
#         if in_civil_twilight:
#             dni_effective = 0.0
#         else:
#             dni_effective = dni

#         cos_zenith = math.cos(math.radians(zenith))
#         ghi_correct = max(0.0, dni * cos_zenith + dhi)   # true formula
#     else:
#         # Full night — no irradiance
#         ghi = dni = dhi = ghi_correct = 0.0
#         dni_effective = 0.0
#         airmass_rel = None

#     return {
#         "ghi": ghi,
#         "dni": dni,
#         "dni_effective": dni_effective,   # 0 below horizon
#         "dhi": dhi,
#         "ghi_correct": ghi_correct,
#         "zenith": zenith,
#         "azimuth": azimuth,
#         "elevation": elevation,
#         "airmass": airmass_rel,
#         "in_civil_twilight": in_civil_twilight,
#         "sun_above_horizon": sun_above_horizon,
#     }


# def get_max_elevation(latitude: float, longitude: float, date_time: str) -> float:
#     """Returns maximum solar elevation angle for the given date (degrees)."""
#     tz_str = pvlib.location.Location(latitude, longitude).tz
#     tz_local = pytz.timezone(tz_str)
#     dt_local = pd.Timestamp(date_time, tz=tz_local)
#     dt_utc = dt_local.tz_convert("UTC")
#     date = dt_utc.date()
#     times = pd.date_range(
#         start=f"{date} 00:00",
#         end=f"{date} 23:59",
#         freq="10min",
#         tz="UTC",
#     )
#     solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
#     return float(90 - solar_pos["apparent_zenith"].min())


# # ---------------------------------------------------------------------------
# # Sun color — physically motivated
# # ---------------------------------------------------------------------------

# def sun_color_from_elevation(elevation_deg: float) -> list:
#     """
#     Returns linear-light RGB sun color as a function of solar elevation.

#     Based on measured correlated colour temperature (CCT) data:
#         elevation ≤ 0°  : deep red/orange  CCT ~2000 K  (twilight scatter)
#         elevation   3°  : orange           CCT ~2500 K
#         elevation   8°  : golden           CCT ~3500 K
#         elevation  15°  : warm white       CCT ~4500 K
#         elevation  30°  : daylight white   CCT ~5500 K
#         elevation ≥ 50° : sky white        CCT ~6000 K

#     Values are normalised so R = 1.0 (Blender uses energy × color).
#     """
#     el = max(elevation_deg, -6.0)   # clamp civil twilight floor

#     if el <= 0:
#         t = (el + 6.0) / 6.0            # 0 at -6°, 1 at 0°
#         r = 1.0
#         g = 0.18 + 0.12 * t             # 0.18 → 0.30
#         b = 0.02 + 0.06 * t             # 0.02 → 0.08

#     elif el <= 3:
#         t = el / 3.0
#         r = 1.0
#         g = 0.30 + 0.08 * t             # 0.30 → 0.38
#         b = 0.08 + 0.08 * t             # 0.08 → 0.16

#     elif el <= 8:
#         t = (el - 3) / 5.0
#         r = 1.0
#         g = 0.38 + 0.17 * t             # 0.38 → 0.55
#         b = 0.16 + 0.14 * t             # 0.16 → 0.30

#     elif el <= 15:
#         t = (el - 8) / 7.0
#         r = 1.0
#         g = 0.55 + 0.20 * t             # 0.55 → 0.75
#         b = 0.30 + 0.20 * t             # 0.30 → 0.50

#     elif el <= 30:
#         t = (el - 15) / 15.0
#         r = 1.0
#         g = 0.75 + 0.17 * t             # 0.75 → 0.92
#         b = 0.50 + 0.32 * t             # 0.50 → 0.82

#     elif el <= 50:
#         t = (el - 30) / 20.0
#         r = 1.0
#         g = 0.92 + 0.06 * t             # 0.92 → 0.98
#         b = 0.82 + 0.14 * t             # 0.82 → 0.96

#     else:
#         r, g, b = 1.0, 1.0, 1.0

#     return [round(r, 3), round(g, 3), round(b, 3)]


# # ---------------------------------------------------------------------------
# # Coordinate conversion
# # ---------------------------------------------------------------------------

# def solar_to_blender_rotation(azimuth_deg: float, zenith_deg: float,
#                                north_offset_deg: float = 0.0) -> tuple:
#     """
#     Converts pvlib solar angles → Blender Euler rotation for a SUN lamp.

#     pvlib convention  : azimuth 0 = North, 90 = East, clockwise
#     Blender convention: Z is up, Y is forward (North by default)

#     Returns (elevation_rad, 0, blender_azimuth_rad)
#     """
#     elevation_rad = math.radians(90 - zenith_deg)
#     # Negate + apply north offset to go from meteorological to Blender azimuth
#     blender_az_rad = math.radians(-(azimuth_deg + north_offset_deg))
#     return (elevation_rad, 0.0, blender_az_rad)


# def solar_azimuth_to_nishita(azimuth_deg: float, north_offset_deg: float = 0.0) -> float:
#     """
#     Converts pvlib azimuth (N=0, CW) → Blender Nishita sun_rotation (radians).

#     Nishita's sun_rotation is measured from South (= +Y in default Blender
#     coordinates) anti-clockwise when viewed from above.
#     South = azimuth 180° in pvlib.

#     nishita_angle = -(azimuth - 180°) converted to radians, + north offset
#     """
#     nishita_deg = -(azimuth_deg - 180.0 + north_offset_deg)
#     return math.radians(nishita_deg)


# # ---------------------------------------------------------------------------
# # Dataclasses
# # ---------------------------------------------------------------------------

# @dataclass
# class RenderParams:
#     """Parameters for rendering synthetic building images."""

#     load_scene: Path = Path("bproc_generator/data/example/House.obj")
#     """Path to the segmented building OBJ file"""
#     output_path: Path = Path("outputs/generated")
#     """Output directory for rendered assets"""
#     resolution: tuple[int, int] = (512, 512)
#     """Rendering resolution (width, height)"""
#     num_frames: int = 10
#     """Number of camera frames to render"""
#     enable_transparency: bool = False
#     """Enable alpha channel in output images"""
#     alpha: float = 1.0
#     """Alpha value for transparent regions (0.0–1.0)"""

#     # Camera parameters
#     load_camera_path: Path = None
#     """Optional path to predefined camera trajectory"""
#     bound_thresh: int = 2
#     """Pixel threshold for boundary object detection"""
#     blank_thresh: int = 20
#     """Pixel threshold for blank boundary detection"""
#     retry_count: int = 30
#     """Max attempts to find valid camera pose"""
#     radius: float = 25
#     """Camera orbit radius from point of interest"""
#     theta_range: tuple[float, float] = (0.0, 2 * np.pi)
#     """Azimuth angle range (radians)"""
#     phi_range: tuple[float, float] = (0.0, 1.178)
#     """Elevation angle range (radians)"""
#     gamma_range: tuple[float, float] = (0.0, 0.0)
#     """In-plane rotation range (radians)"""
#     dist_thresh: float = 2.0
#     """Minimum distance between camera poses to avoid overlap"""

#     # Camera adjustment parameters
#     delta_radius: tuple[float, float] = (0.5, 2.0)
#     delta_theta: tuple[float, float] = (0.05, 0.10)
#     delta_phi: tuple[float, float] = (0.05, 0.10)
#     delta_poi: tuple[float, float] = (0.0, 0.5)

#     # Location / time  (required)
#     latitude: float = tyro.MISSING
#     """Building location latitude (degrees)"""
#     longitude: float = tyro.MISSING
#     """Building location longitude (degrees)"""
#     altitude: float = 400.0
#     """Building location altitude above sea level (metres). Used for Ineichen model."""
#     date_time: str = tyro.MISSING
#     """Date and time for sun position (YYYY-MM-DD HH:MM:SS) in local wall-clock time."""
#     turbidity: float = DEFAULT_TURBIDITY
#     """Linke turbidity factor for Ineichen clear-sky model (2=very clear, 3=typical, 5=hazy)."""

#     # Lighting calibration  (one-time physical constants, NOT per-scene tuning)
#     k_sun: float = K_SUN
#     """Blender SUN lamp energy per W/m² of DNI. Calibrate once, then freeze."""
#     k_sky: float = K_SKY
#     """Nishita background strength per W/m² of DHI. Calibrate once, then freeze."""

#     # Procedural sky options
#     use_hdr_background: bool = True
#     """Enable Nishita procedural sky (DHI component)."""
#     air_density: float = 1.0
#     """Nishita air density parameter."""
#     dust_density: float = 0.3
#     """Nishita dust/aerosol density parameter."""

#     # Sun lamp options
#     use_sun: bool = True
#     """Enable sun light source (DNI component)."""
#     north_offset_deg: float = 0.0
#     """Rotation offset to align building model with true North (degrees).
#     Set to 0 if the building's Y axis points North in the .obj file."""


# @dataclass
# class BlenderProcRenderer(RenderParams):
#     """Generates synthetic building imagery with BlenderProc."""

#     def __post_init__(self):
#         self.output_path.mkdir(parents=True, exist_ok=True)
#         self.metadata: dict = {
#             "camera_model": "OPEN_CV",
#             "width": self.resolution[0],
#             "height": self.resolution[1],
#             "has_mono_prior": False,
#             "has_foreground_mask": False,
#             "has_sparse_sfm_points": False,
#             "scene_box": {},
#             "frames": [],
#         }
#         self.metadata["location"] = {
#             "latitude": self.latitude,
#             "longitude": self.longitude,
#             "altitude": self.altitude,
#         }
#         self.metadata["capture_time"] = self.date_time

#         # Sun state (filled in _setup_lighting)
#         self.irradiance: dict = {}
#         self.sun_energy_actual: float | None = None
#         self.hdr_energy_actual: float | None = None
#         self.sun_color_actual: list | None = None

#         bproc.init()
#         self.scene_objects = bproc.loader.load_obj(str(self.load_scene))
#         self.bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.scene_objects)

#         self._assign_categories()
#         self._setup_camera()
#         self._setup_lighting()

#         self.color_map = self._get_color_map()
#         self.camera_idx = 0

#     # ------------------------------------------------------------------
#     # Scene setup
#     # ------------------------------------------------------------------

#     def _assign_categories(self):
#         category_ids = [1, 3, 1, 1, 5, 4, 1, 2]
#         for obj, cid in zip(self.scene_objects, category_ids):
#             obj.set_cp("category_id", cid)

#     def _setup_camera(self):
#         self.camera_list: list[np.ndarray] = []
#         bproc.camera.set_resolution(*self.resolution)

#     def _setup_lighting(self):
#         """
#         Physically accurate lighting pipeline
#         ──────────────────────────────────────
#         1. pvlib Ineichen → DNI, DHI, GHI  [W/m²]
#         2. DHI × k_sky   → Nishita background strength
#         3. DNI × k_sun   → Blender SUN lamp energy
#         4. GHI = DNI·cos(θ_z) + DHI  stored in metadata (correct formula)

#         Civil twilight (90° ≤ zenith ≤ 96°):
#             - No SUN lamp (disc below horizon)
#             - Nishita sky still active with DHI from model
#             - Result: blue-hour ambient glow, no hard shadows
#         """
#         irr = get_clear_sky_irradiance(
#             self.latitude, self.longitude, self.altitude, self.date_time, self.turbidity
#         )
#         self.irradiance = irr

#         zenith = irr["zenith"]
#         azimuth = irr["azimuth"]
#         elevation = irr["elevation"]

#         print(f"\n{'─'*50}")
#         print(f"  Solar position  : elevation {elevation:.1f}°  zenith {zenith:.1f}°  azimuth {azimuth:.1f}°")
#         print(f"  GHI             : {irr['ghi_correct']:.1f} W/m²  (= DNI·cos θ + DHI)")
#         print(f"  DNI             : {irr['dni']:.1f} W/m²")
#         print(f"  DHI             : {irr['dhi']:.1f} W/m²")
#         if irr["in_civil_twilight"]:
#             print(f"  ⚠  Civil twilight — sun disc below horizon, no direct lamp")
#         print(f"{'─'*50}\n")

#         # ── Nishita procedural sky (DHI) ──────────────────────────────
#         if self.use_hdr_background and (irr["sun_above_horizon"] or irr["in_civil_twilight"]):
#             self._setup_nishita_sky(zenith, azimuth, elevation, irr["dhi"])
#         else:
#             self._setup_night_sky()

#         # ── Directional SUN lamp (DNI) ────────────────────────────────
#         if self.use_sun and irr["sun_above_horizon"]:
#             self._setup_sun_lamp(azimuth, zenith, elevation, irr["dni_effective"])

#     def _setup_nishita_sky(self, zenith: float, azimuth: float,
#                             elevation: float, dhi: float):
#         """Configures Blender's Nishita sky texture driven by pvlib DHI."""
#         world = bpy.data.worlds["World"]
#         world.use_nodes = True
#         nodes = world.node_tree.nodes
#         links = world.node_tree.links
#         nodes.clear()

#         sky = nodes.new("ShaderNodeTexSky")
#         sky.sky_type = "NISHITA"
#         sky.sun_elevation = math.radians(max(elevation, -6.0))  # clamp civil twilight
#         sky.sun_rotation = solar_azimuth_to_nishita(azimuth, self.north_offset_deg)
#         sky.altitude = self.altitude
#         sky.air_density = self.air_density
#         sky.dust_density = self.dust_density

#         bg = nodes.new("ShaderNodeBackground")
#         out = nodes.new("ShaderNodeOutputWorld")
#         links.new(sky.outputs[0], bg.inputs[0])
#         links.new(bg.outputs[0], out.inputs[0])

#         # Scale DHI [W/m²] → Nishita strength
#         # Nishita at strength=1 already represents a physically plausible sky;
#         # k_sky maps physical irradiance onto that scale.
#         sky_strength = dhi * self.k_sky
#         bg.inputs[1].default_value = sky_strength
#         self.hdr_energy_actual = dhi

#         print(f"  Nishita sky     : DHI={dhi:.1f} W/m²  →  strength={sky_strength:.4f}")

#     def _setup_night_sky(self):
#         """Sets world background to black for night scenes."""
#         world = bpy.data.worlds["World"]
#         world.use_nodes = True
#         nodes = world.node_tree.nodes
#         nodes.clear()
#         bg = nodes.new("ShaderNodeBackground")
#         bg.inputs[0].default_value = (0.005, 0.005, 0.015, 1)  # very faint blue-black
#         bg.inputs[1].default_value = 0.02
#         out = nodes.new("ShaderNodeOutputWorld")
#         world.node_tree.links.new(bg.outputs[0], out.inputs[0])
#         self.hdr_energy_actual = 0.0
#         print("  Night sky       : minimal ambient (no solar irradiance)")

#     def _setup_sun_lamp(self, azimuth: float, zenith: float,
#                          elevation: float, dni: float):
#         """Creates and configures the directional SUN lamp from DNI."""
#         sun_rotation = solar_to_blender_rotation(azimuth, zenith, self.north_offset_deg)
#         color = sun_color_from_elevation(elevation)
#         energy = dni * self.k_sun

#         self.sun_energy_actual = dni
#         self.sun_color_actual = color

#         sun = bproc.types.Light()
#         sun.set_type("SUN")
#         sun.set_energy(energy)
#         sun.set_color(color)
#         sun.blender_obj.rotation_euler = sun_rotation

#         dhi = self.hdr_energy_actual or 0.0
#         cos_z = math.cos(math.radians(zenith))
#         ghi = dni * cos_z + dhi

#         print(f"  SUN lamp        : DNI={dni:.1f} W/m²  →  energy={energy:.4f}")
#         print(f"  GHI (correct)   : {ghi:.1f} W/m²  (= {dni:.1f}·cos({zenith:.1f}°) + {dhi:.1f})")
#         print(f"  Color           : R={color[0]}  G={color[1]}  B={color[2]}")
#         print(f"  Azimuth/Zenith  : {azimuth:.1f}° / {zenith:.1f}°")

#     # ------------------------------------------------------------------
#     # Camera
#     # ------------------------------------------------------------------

#     def _check_boundary(self, camera_pose: np.ndarray) -> tuple[list[bool], dict]:
#         bproc.camera.add_camera_pose(camera_pose, self.camera_idx)
#         depth = bproc.camera.depth_via_raytracing(self.bvh_tree)

#         boundaries = {
#             "top": [
#                 np.all(depth[:self.bound_thresh, :] == np.inf),
#                 not np.all(depth[:self.blank_thresh, :] == np.inf),
#             ],
#             "bottom": [
#                 np.all(depth[-self.bound_thresh:, :] == np.inf),
#                 not np.all(depth[-self.blank_thresh:, :] == np.inf),
#             ],
#             "left": [
#                 np.all(depth[:, :self.bound_thresh] == np.inf),
#                 not np.all(depth[:, :self.blank_thresh] == np.inf),
#             ],
#             "right": [
#                 np.all(depth[:, -self.bound_thresh:] == np.inf),
#                 not np.all(depth[:, -self.blank_thresh:] == np.inf),
#             ],
#         }

#         bound_valid = sum(v[0] for v in boundaries.values()) == 4
#         content_valid = sum(v[1] for v in boundaries.values()) >= 3

#         if boundaries["top"][1] and boundaries["bottom"][1]:
#             content_valid = True
#         if boundaries["left"][1] and boundaries["right"][1]:
#             content_valid = True

#         return [bound_valid, content_valid], boundaries

#     def generate_camera_pose(self, point_of_interest: np.ndarray):
#         def random_camera_params():
#             theta = np.random.uniform(*self.theta_range)
#             phi = np.random.uniform(*self.phi_range)
#             gamma = np.random.uniform(*self.gamma_range)
#             poi = point_of_interest + np.random.uniform(
#                 -self.delta_poi[1], self.delta_poi[1], size=3
#             )
#             return theta, phi, gamma, poi

#         print(f" ----- Generating Camera Sample {self.camera_idx} ----- ")
#         theta, phi, gamma, poi = random_camera_params()
#         radius = self.radius
#         retries = 0

#         while True:
#             print(f"Retry {retries} for camera ID {self.camera_idx}", end="\r")
#             location = build_translation(radius, theta, phi)
#             rotation = bproc.camera.rotation_from_forward_vec(
#                 poi - location, inplane_rot=gamma
#             )
#             camera_pose = bproc.math.build_transformation_mat(location, rotation)

#             (bound_valid, content_valid), boundaries = self._check_boundary(camera_pose)

#             if not bound_valid:
#                 radius += np.random.uniform(*self.delta_radius)
#             if not content_valid:
#                 radius -= np.random.uniform(*self.delta_radius)

#             if not boundaries["top"][0] or not boundaries["bottom"][1]:
#                 poi[2] += np.random.uniform(*self.delta_poi)
#                 phi += np.random.uniform(*self.delta_phi)
#             if not boundaries["bottom"][0] or not boundaries["top"][1]:
#                 poi[2] -= np.random.uniform(*self.delta_poi)
#                 phi -= np.random.uniform(*self.delta_phi)
#             if not boundaries["left"][0] or not boundaries["right"][1]:
#                 theta -= np.random.uniform(*self.delta_theta)
#             if not boundaries["right"][0] or not boundaries["left"][1]:
#                 theta += np.random.uniform(*self.delta_theta)

#             if bound_valid and content_valid:
#                 if all(np.linalg.norm(camera_pose[:3, 3] - p[:3, 3]) > self.dist_thresh
#                        for p in self.camera_list):
#                     self._add_camera_pose(camera_pose)
#                     return

#             retries += 1
#             if retries >= self.retry_count:
#                 theta, phi, gamma, poi = random_camera_params()
#                 radius = self.radius
#                 retries = 0

#     def _add_camera_pose(self, pose: np.ndarray):
#         bproc.camera.add_camera_pose(pose, self.camera_idx)
#         self.camera_list.append(pose)
#         self.metadata["frames"].append({
#             "rgb_path": f"{self.camera_idx:04d}.png",
#             "segmentation_path": f"{self.camera_idx:04d}_mask.png",
#             "camera_to_world": pose.tolist(),
#             "intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
#         })
#         self.camera_idx += 1

#     # ------------------------------------------------------------------
#     # Output
#     # ------------------------------------------------------------------

#     def save_images(self):
#         for subdir in ["images", "normals", "depths", "semantics", "instances"]:
#             (self.output_path / subdir).mkdir(exist_ok=True)

#         for i in range(self.camera_idx):
#             with h5py.File(self.output_path / f"{i}.hdf5", "r") as f:
#                 rgb = np.array(f["colors"][:])
#                 if self.enable_transparency:
#                     alpha = np.full((*rgb.shape[:2], 1), int(self.alpha * 255), dtype=np.uint8)
#                     rgb = np.concatenate([rgb[..., :3], alpha], axis=-1)
#                 Image.fromarray(rgb).save(self.output_path / "images" / f"{i:04d}.png")

#                 normal = (f["normals"][:] * 255).astype(np.uint8)
#                 Image.fromarray(normal).save(self.output_path / "normals" / f"{i:04d}_normal.png")

#                 depth = (f["depth"][:] * 1000).astype(np.uint16)
#                 Image.fromarray(depth).save(self.output_path / "depths" / f"{i:04d}_depth.png")

#                 semantic = f["category_id_segmaps"][:]
#                 semantics = np.zeros((*semantic.shape, 3), dtype=np.uint8)
#                 for j, color in self.color_map.items():
#                     semantics[semantic == int(j)] = color
#                 Image.fromarray(semantics).save(self.output_path / "semantics" / f"{i:04d}_mask.png")

#                 instance = f["instance_segmaps"][:]
#                 instances = np.zeros((*instance.shape, 3), dtype=np.uint8)
#                 for j, color in self.color_map.items():
#                     instances[instance == int(j)] = color
#                 Image.fromarray(instances).save(self.output_path / "instances" / f"{i:04d}.png")

#             (self.output_path / f"{i}.hdf5").unlink()

#     def save_metadata(self):
#         irr = self.irradiance
#         dhi = self.hdr_energy_actual or 0.0
#         dni = self.sun_energy_actual or 0.0
#         zenith = irr.get("zenith")
#         cos_z = math.cos(math.radians(zenith)) if zenith is not None else 0.0

#         self.metadata["sun"] = {
#             "azimuth": irr.get("azimuth"),
#             "zenith": zenith,
#             "elevation": irr.get("elevation"),
#             "DNI_Wm2": irr.get("dni"),
#             "DHI_Wm2": dhi,
#             "GHI_Wm2": dni * cos_z + dhi,        # correct: DNI·cos(θ_z) + DHI
#             "GHI_pvlib_Wm2": irr.get("ghi"),      # pvlib's own GHI for cross-check
#             "airmass": irr.get("airmass"),
#             "turbidity": self.turbidity,
#             "in_civil_twilight": irr.get("in_civil_twilight"),
#             "sun_color": self.sun_color_actual,
#             "k_sun": self.k_sun,
#             "k_sky": self.k_sky,
#             "blender_sun_energy": (dni * self.k_sun) if irr.get("sun_above_horizon") else 0.0,
#             "blender_sky_strength": (dhi * self.k_sky),
#             "date_time": self.date_time,
#             "latitude": self.latitude,
#             "longitude": self.longitude,
#             "altitude": self.altitude,
#         }
#         with open(self.output_path / "meta_data.json", "w") as f:
#             json.dump(self.metadata, f, indent=4)

#     def run(self):
#         poi = bproc.object.compute_poi(self.scene_objects)
#         for _ in range(self.num_frames):
#             self.generate_camera_pose(poi)

#         bproc.renderer.set_output_format(enable_transparency=self.enable_transparency)
#         bproc.renderer.enable_depth_output(activate_antialiasing=False)
#         bproc.renderer.enable_normals_output()
#         bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

#         render_data = bproc.renderer.render()
#         bproc.writer.write_hdf5(str(self.output_path), render_data)
#         self.save_images()
#         self.save_metadata()

#     @staticmethod
#     def _get_color_map() -> dict[str, list[float]]:
#         return {
#             "0": [0, 0, 0],
#             "1": [175, 200, 0],
#             "2": [0, 200, 200],
#             "3": [125, 0, 200],
#             "4": [175, 0, 200],
#             "5": [0, 50, 200],
#             "6": [0, 200, 50],
#             "7": [150, 50, 50],
#             "8": [50, 175, 50],
#             "9": [50, 50, 175],
#         }


# # ---------------------------------------------------------------------------

# def main():
#     tyro.extras.set_accent_color("bright_yellow")
#     tyro.cli(BlenderProcRenderer).run()


# if __name__ == "__main__":
#     main()


import blenderproc as bproc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import tyro

import math
import pvlib
import pandas as pd
import pytz
import bpy

sys.path.extend([str(Path(__file__).resolve().parents[2])])
from buildnet3d.utils.utils import build_translation


# ---------------------------------------------------------------------------
# Lighting calibration
#
# WHY A SINGLE K_SUN + DERIVED SKY STRENGTH
# -------------------------------------------
# Blender's ShaderNodeTexSky (Nishita) is a full hemispherical env light,
# not just a background colour.  Its internal radiance already scales with
# sun elevation, so per unit of "strength" it contributes ~37x more fill
# than a SUN lamp contributes per unit of energy.  Using two independent
# scalars (K_SUN, K_SKY) means their ratio must secretly encode this 37x
# factor -- which is why empirical tuning always converges to K_SUN/K_SKY~300
# regardless of scene (300 = 37 * DNI/DHI ratio ~8).
#
# Clean model:
#   sun_energy   = DNI * K_SUN
#   sky_strength = DHI * K_SUN / NISHITA_FILL_FACTOR
#                = DHI * K_SUN / 37
#
# This way K_SUN is the ONLY exposure knob. Shadow hardness is controlled
# by NISHITA_FILL_FACTOR (scene-independent, derived from Blender internals).
# At low sun elevations SHADOW_FILL_BOOST softens shadows for golden-hour look.
#
# Your empirical calibration:  K_SUN=0.3, K_SKY=0.001
#   -> K_SUN/K_SKY = 300 = NISHITA_FILL_FACTOR * (DNI/DHI) = 37 * ~8  CHECK
# ---------------------------------------------------------------------------
K_SUN: float = 0.0075             # SUN lamp energy per W/m2 DNI  (your calibration)
NISHITA_FILL_FACTOR: float = 15.0   # Nishita env-light over-fill vs SUN lamp
SHADOW_FILL_BOOST: float = 1.0      # Extra sky fill at low elevations (golden hour)

# Civil twilight ends at zenith 96° — sky is still visibly illuminated
CIVIL_TWILIGHT_ZENITH: float = 96.0

# Turbidity for Ineichen clear-sky model (Linke turbidity factor)
# 2.0 = very clear mountain air, 3.0 = typical European clear day,
# 4.5 = hazy urban, 6+ = very hazy.  Use pvlib.clearsky.lookup_linke_turbidity
# for location-aware values.
DEFAULT_TURBIDITY: float = 3.0


# ---------------------------------------------------------------------------
# pvlib helpers
# ---------------------------------------------------------------------------

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

    Returns
    -------
    dict with keys:
        ghi   : Global Horizontal Irradiance  [W/m²]
        dni   : Direct Normal Irradiance      [W/m²]
        dhi   : Diffuse Horizontal Irradiance [W/m²]
        ghi_correct : DNI·cos(θ_z) + DHI     [W/m²]  (exact formula)
        zenith      : apparent solar zenith   [°]
        azimuth     : solar azimuth (N=0, E=90, clockwise) [°]
        elevation   : solar elevation         [°]
        airmass     : relative optical airmass
        in_civil_twilight : True if sun is below horizon but < 96° zenith
    """
    dt_utc = pd.Timestamp(date_time, tz="UTC")
    times = pd.DatetimeIndex([dt_utc])

    solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    zenith = float(solar_pos["apparent_zenith"].iloc[0])
    azimuth = float(solar_pos["azimuth"].iloc[0])
    elevation = float(solar_pos["apparent_elevation"].iloc[0])

    in_civil_twilight = 90.0 <= zenith <= CIVIL_TWILIGHT_ZENITH
    sun_above_horizon = zenith < 90.0

    if sun_above_horizon or in_civil_twilight:
        # Ineichen clear-sky model
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
        # DNI contributes nothing to horizontal surfaces but the sky is
        # still lit. We zero DNI for the lamp but keep DHI for the sky.
        if in_civil_twilight:
            dni_effective = 0.0
        else:
            dni_effective = dni

        cos_zenith = math.cos(math.radians(zenith))
        ghi_correct = max(0.0, dni * cos_zenith + dhi)   # true formula
    else:
        # Full night — no irradiance
        ghi = dni = dhi = ghi_correct = 0.0
        dni_effective = 0.0
        airmass_rel = None

    return {
        "ghi": ghi,
        "dni": dni,
        "dni_effective": dni_effective,   # 0 below horizon
        "dhi": dhi,
        "ghi_correct": ghi_correct,
        "zenith": zenith,
        "azimuth": azimuth,
        "elevation": elevation,
        "airmass": airmass_rel,
        "in_civil_twilight": in_civil_twilight,
        "sun_above_horizon": sun_above_horizon,
    }


def get_max_elevation(latitude: float, longitude: float, date_time: str) -> float:
    """Returns maximum solar elevation angle for the given date (degrees)."""
    tz_str = pvlib.location.Location(latitude, longitude).tz
    tz_local = pytz.timezone(tz_str)
    dt_local = pd.Timestamp(date_time, tz=tz_local)
    dt_utc = dt_local.tz_convert("UTC")
    date = dt_utc.date()
    times = pd.date_range(
        start=f"{date} 00:00",
        end=f"{date} 23:59",
        freq="10min",
        tz="UTC",
    )
    solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    return float(90 - solar_pos["apparent_zenith"].min())


# ---------------------------------------------------------------------------
# Sun color — physically motivated
# ---------------------------------------------------------------------------

def sun_color_from_elevation(elevation_deg: float) -> list:
    """
    Returns linear-light RGB sun color as a function of solar elevation.

    Based on measured correlated colour temperature (CCT) data:
        elevation ≤ 0°  : deep red/orange  CCT ~2000 K  (twilight scatter)
        elevation   3°  : orange           CCT ~2500 K
        elevation   8°  : golden           CCT ~3500 K
        elevation  15°  : warm white       CCT ~4500 K
        elevation  30°  : daylight white   CCT ~5500 K
        elevation ≥ 50° : sky white        CCT ~6000 K

    Values are normalised so R = 1.0 (Blender uses energy × color).
    """
    el = max(elevation_deg, -6.0)   # clamp civil twilight floor

    if el <= 0:
        t = (el + 6.0) / 6.0            # 0 at -6°, 1 at 0°
        r = 1.0
        g = 0.18 + 0.12 * t             # 0.18 → 0.30
        b = 0.02 + 0.06 * t             # 0.02 → 0.08

    elif el <= 3:
        t = el / 3.0
        r = 1.0
        g = 0.30 + 0.08 * t             # 0.30 → 0.38
        b = 0.08 + 0.08 * t             # 0.08 → 0.16

    elif el <= 8:
        t = (el - 3) / 5.0
        r = 1.0
        g = 0.38 + 0.17 * t             # 0.38 → 0.55
        b = 0.16 + 0.14 * t             # 0.16 → 0.30

    elif el <= 15:
        t = (el - 8) / 7.0
        r = 1.0
        g = 0.55 + 0.20 * t             # 0.55 → 0.75
        b = 0.30 + 0.20 * t             # 0.30 → 0.50

    elif el <= 30:
        t = (el - 15) / 15.0
        r = 1.0
        g = 0.75 + 0.17 * t             # 0.75 → 0.92
        b = 0.50 + 0.32 * t             # 0.50 → 0.82

    elif el <= 50:
        t = (el - 30) / 20.0
        r = 1.0
        g = 0.92 + 0.06 * t             # 0.92 → 0.98
        b = 0.82 + 0.14 * t             # 0.82 → 0.96

    else:
        r, g, b = 1.0, 1.0, 1.0

    return [round(r, 3), round(g, 3), round(b, 3)]


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def solar_to_blender_rotation(azimuth_deg: float, zenith_deg: float,
                               north_offset_deg: float = 180.0) -> tuple:
    """
    Converts pvlib solar angles → Blender Euler rotation for a SUN lamp.

    pvlib convention  : azimuth 0 = North, 90 = East, clockwise
    Blender convention: Z is up, Y is forward (North by default)

    Returns (elevation_rad, 0, blender_azimuth_rad)
    """
    elevation_rad = math.radians(90 - zenith_deg)
    # elevation_rad = math.radians(zenith_deg - 90)
    # Negate + apply north offset to go from meteorological to Blender azimuth
    blender_az_rad = math.radians(180 - azimuth_deg - north_offset_deg)

    # return (elevation_rad, 0.0, blender_az_rad)
    return (0.0, elevation_rad, blender_az_rad)


def solar_azimuth_to_nishita(azimuth_deg: float, north_offset_deg: float = 180.0) -> float:
    """
    Converts pvlib azimuth (N=0, CW) → Blender Nishita sun_rotation (radians).

    Blender's ShaderNodeTexSky (Nishita) sun_rotation is measured clockwise
    from North when viewed from above — the SAME convention as a meteorological
    azimuth — so the mapping is a direct pass-through (no negation).

    The SUN lamp uses -(azimuth + offset) because Blender lamp rotation goes
    counter-clockwise.  The Nishita node is independent and goes clockwise,
    so we must NOT negate here, otherwise the sky gradient and the lamp point
    in opposite directions (180° flip = North/South inversion).
    """
    return math.radians(azimuth_deg + north_offset_deg -180)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    """Alpha value for transparent regions (0.0–1.0)"""

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
    delta_theta: tuple[float, float] = (0.05, 0.10)
    delta_phi: tuple[float, float] = (0.05, 0.10)
    delta_poi: tuple[float, float] = (0.0, 0.5)

    # Location / time  (required)
    latitude: float = tyro.MISSING
    """Building location latitude (degrees)"""
    longitude: float = tyro.MISSING
    """Building location longitude (degrees)"""
    altitude: float = 400.0
    """Building location altitude above sea level (metres). Used for Ineichen model."""
    date_time: str = tyro.MISSING
    """Date and time for sun position (YYYY-MM-DD HH:MM:SS) in local wall-clock time."""
    turbidity: float = DEFAULT_TURBIDITY
    """Linke turbidity factor for Ineichen clear-sky model (2=very clear, 3=typical, 5=hazy)."""

    # Lighting calibration  (one-time physical constants, NOT per-scene tuning)
    k_sun: float = K_SUN
    """Overall exposure scalar: SUN lamp energy per W/m2 of DNI. The only knob you need."""
    nishita_fill_factor: float = NISHITA_FILL_FACTOR
    """How much more fill Nishita produces vs SUN lamp per unit. Scene-independent (~37)."""
    shadow_fill_boost: float = SHADOW_FILL_BOOST
    """Extra sky fill multiplier at low elevations for golden-hour softness."""

    # Procedural sky options
    use_hdr_background: bool = True
    """Enable Nishita procedural sky (DHI component)."""
    air_density: float = 1.0
    """Nishita air density parameter."""
    dust_density: float = 0.3
    """Nishita dust/aerosol density parameter."""

    # Sun lamp options
    use_sun: bool = True
    """Enable sun light source (DNI component)."""
    north_offset_deg: float = 180.0
    """Rotation offset to align building model with true North (degrees).
    Set to 0 if the building's Y axis points North in the .obj file."""


@dataclass
class BlenderProcRenderer(RenderParams):
    """Generates synthetic building imagery with BlenderProc."""

    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata: dict = {
            "camera_model": "OPEN_CV",
            "width": self.resolution[0],
            "height": self.resolution[1],
            "has_mono_prior": False,
            "has_foreground_mask": False,
            "has_sparse_sfm_points": False,
            "scene_box": {},
            "frames": [],
        }
        self.metadata["location"] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }
        self.metadata["capture_time"] = self.date_time

        # Sun state (filled in _setup_lighting)
        self.irradiance: dict = {}
        self.sun_energy_actual: float | None = None
        self.hdr_energy_actual: float | None = None
        self.sun_color_actual: list | None = None

        bproc.init()
        # self.scene_objects = bproc.loader.load_obj(str(self.load_scene))
        self.scene_objects = bproc.loader.load_obj(
            str(self.load_scene.resolve())
        )
        self.bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.scene_objects)

        self._assign_categories()
        self._setup_camera()
        self._setup_lighting()

        self.color_map = self._get_color_map()
        self.camera_idx = 0

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _assign_categories(self):
        category_ids = [1, 3, 1, 1, 5, 4, 1, 2]
        for obj, cid in zip(self.scene_objects, category_ids):
            obj.set_cp("category_id", cid)

    def _setup_camera(self):
        self.camera_list: list[np.ndarray] = []
        bproc.camera.set_resolution(*self.resolution)

    def _setup_lighting(self):
        """
        Physically accurate lighting pipeline
        ──────────────────────────────────────
        1. pvlib Ineichen → DNI, DHI, GHI  [W/m²]
        2. DHI * k_sun / nishita_fill_factor  -> Nishita background strength
        3. DNI × k_sun   → Blender SUN lamp energy
        4. GHI = DNI·cos(θ_z) + DHI  stored in metadata (correct formula)

        Civil twilight (90° ≤ zenith ≤ 96°):
            - No SUN lamp (disc below horizon)
            - Nishita sky still active with DHI from model
            - Result: blue-hour ambient glow, no hard shadows
        """
        irr = get_clear_sky_irradiance(
            self.latitude, self.longitude, self.altitude, self.date_time, self.turbidity
        )
        self.irradiance = irr

        zenith = irr["zenith"]
        azimuth = irr["azimuth"]
        elevation = irr["elevation"]

        print(f"\n{'─'*50}")
        print(f"  Solar position  : elevation {elevation:.1f}°  zenith {zenith:.1f}°  azimuth {azimuth:.1f}°")
        print(f"  GHI             : {irr['ghi_correct']:.1f} W/m²  (= DNI·cos θ + DHI)")
        print(f"  DNI             : {irr['dni']:.1f} W/m²")
        print(f"  DHI             : {irr['dhi']:.1f} W/m²")
        if irr["in_civil_twilight"]:
            print(f"  ⚠  Civil twilight — sun disc below horizon, no direct lamp")
        print(f"{'─'*50}\n")

        # ── Nishita procedural sky (DHI) ──────────────────────────────
        if self.use_hdr_background and (irr["sun_above_horizon"] or irr["in_civil_twilight"]):
            self._setup_nishita_sky(zenith, azimuth, elevation, irr["dhi"])
        else:
            self._setup_night_sky()

        # ── Directional SUN lamp (DNI) ────────────────────────────────
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
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(max(elevation, -6.0))  # clamp civil twilight
        sky.sun_rotation = solar_azimuth_to_nishita(azimuth, self.north_offset_deg)
        sky.altitude = self.altitude
        sky.air_density = self.air_density
        sky.dust_density = self.dust_density

        bg = nodes.new("ShaderNodeBackground")
        out = nodes.new("ShaderNodeOutputWorld")
        links.new(sky.outputs[0], bg.inputs[0])
        links.new(bg.outputs[0], out.inputs[0])

        # Sky strength derived from DNI/DHI ratio and Nishita fill factor.
        #
        # sky_strength = DHI * K_SUN / NISHITA_FILL_FACTOR
        #
        # This guarantees:
        #  - The fill-to-key ratio tracks physical DNI/DHI throughout the day
        #  - At noon (DNI=900, DHI=110): sky_strength = 110*0.3/37 = 0.89  -> sharp shadows
        #  - At sunrise (DNI=200, DHI=60): sky_strength = 60*0.3/37 = 0.49 -> softer fill
        #
        # SHADOW_FILL_BOOST adds extra softness at low elevations (el < 15 deg)
        # to reproduce the warm diffuse look of golden hour without affecting noon.
        boost = 1.0
        if elevation < 15.0:
            # Linearly ramp from shadow_fill_boost at horizon to 1.0 at 15 deg
            t = max(elevation, 0.0) / 15.0
            boost = self.shadow_fill_boost * (1.0 - t) + 1.0 * t

        sky_strength = (dhi * self.k_sun / self.nishita_fill_factor) * boost
        bg.inputs[1].default_value = sky_strength
        self.hdr_energy_actual = dhi

        print(f"  Nishita sky     : DHI={dhi:.1f} el={elevation:.1f}deg boost={boost:.2f} -> strength={sky_strength:.4f}")

    def _setup_night_sky(self):
        """Sets world background to black for night scenes."""
        world = bpy.data.worlds["World"]
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs[0].default_value = (0.005, 0.005, 0.015, 1)  # very faint blue-black
        bg.inputs[1].default_value = 0.02
        out = nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs[0], out.inputs[0])
        self.hdr_energy_actual = 0.0
        print("  Night sky       : minimal ambient (no solar irradiance)")

    def _setup_sun_lamp(self, azimuth: float, zenith: float,
                         elevation: float, dni: float):
        """Creates and configures the directional SUN lamp from DNI."""
        sun_rotation = solar_to_blender_rotation(azimuth, zenith, self.north_offset_deg)
        color = sun_color_from_elevation(elevation)
        energy = dni * self.k_sun

        self.sun_energy_actual = dni
        self.sun_color_actual = color

        sun = bproc.types.Light()
        sun.set_type("SUN")
        sun.set_energy(energy)
        sun.set_color(color)
        # sun.blender_obj.rotation_euler = sun_rotation
        sun.blender_obj.rotation_mode = 'XYZ'
        sun.blender_obj.rotation_euler = sun_rotation

        dhi = self.hdr_energy_actual or 0.0
        cos_z = math.cos(math.radians(zenith))
        ghi = dni * cos_z + dhi

        print(f"  SUN lamp        : DNI={dni:.1f} W/m²  →  energy={energy:.4f}")
        print(f"  GHI (correct)   : {ghi:.1f} W/m²  (= {dni:.1f}·cos({zenith:.1f}°) + {dhi:.1f})")
        print(f"  Color           : R={color[0]}  G={color[1]}  B={color[2]}")
        print(f"  Azimuth/Zenith  : {azimuth:.1f}° / {zenith:.1f}°")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _check_boundary(self, camera_pose: np.ndarray) -> tuple[list[bool], dict]:
        bproc.camera.add_camera_pose(camera_pose, self.camera_idx)
        depth = bproc.camera.depth_via_raytracing(self.bvh_tree)

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

        bound_valid = sum(v[0] for v in boundaries.values()) == 4
        content_valid = sum(v[1] for v in boundaries.values()) >= 3

        if boundaries["top"][1] and boundaries["bottom"][1]:
            content_valid = True
        if boundaries["left"][1] and boundaries["right"][1]:
            content_valid = True

        return [bound_valid, content_valid], boundaries

    def generate_camera_pose(self, point_of_interest: np.ndarray):
        def random_camera_params():
            theta = np.random.uniform(*self.theta_range)
            phi = np.random.uniform(*self.phi_range)
            gamma = np.random.uniform(*self.gamma_range)
            poi = point_of_interest + np.random.uniform(
                -self.delta_poi[1], self.delta_poi[1], size=3
            )
            return theta, phi, gamma, poi

        print(f" ----- Generating Camera Sample {self.camera_idx} ----- ")
        theta, phi, gamma, poi = random_camera_params()
        radius = self.radius
        retries = 0

        while True:
            print(f"Retry {retries} for camera ID {self.camera_idx}", end="\r")
            location = build_translation(radius, theta, phi)
            rotation = bproc.camera.rotation_from_forward_vec(
                poi - location, inplane_rot=gamma
            )
            camera_pose = bproc.math.build_transformation_mat(location, rotation)

            (bound_valid, content_valid), boundaries = self._check_boundary(camera_pose)

            if not bound_valid:
                radius += np.random.uniform(*self.delta_radius)
            if not content_valid:
                radius -= np.random.uniform(*self.delta_radius)

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

            if bound_valid and content_valid:
                if all(np.linalg.norm(camera_pose[:3, 3] - p[:3, 3]) > self.dist_thresh
                       for p in self.camera_list):
                    self._add_camera_pose(camera_pose)
                    return

            retries += 1
            if retries >= self.retry_count:
                theta, phi, gamma, poi = random_camera_params()
                radius = self.radius
                retries = 0

    def _add_camera_pose(self, pose: np.ndarray):
        bproc.camera.add_camera_pose(pose, self.camera_idx)
        self.camera_list.append(pose)
        self.metadata["frames"].append({
            "rgb_path": f"{self.camera_idx:04d}.png",
            "segmentation_path": f"{self.camera_idx:04d}_mask.png",
            "camera_to_world": pose.tolist(),
            "intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
        })
        self.camera_idx += 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_images(self):
        for subdir in ["images", "normals", "depths", "semantics", "instances"]:
            (self.output_path / subdir).mkdir(exist_ok=True)

        for i in range(self.camera_idx):
            with h5py.File(self.output_path / f"{i}.hdf5", "r") as f:
                rgb = np.array(f["colors"][:])
                if self.enable_transparency:
                    alpha = np.full((*rgb.shape[:2], 1), int(self.alpha * 255), dtype=np.uint8)
                    rgb = np.concatenate([rgb[..., :3], alpha], axis=-1)
                Image.fromarray(rgb).save(self.output_path / "images" / f"{i:04d}.png")

                normal = (f["normals"][:] * 255).astype(np.uint8)
                Image.fromarray(normal).save(self.output_path / "normals" / f"{i:04d}_normal.png")

                depth = (f["depth"][:] * 1000).astype(np.uint16)
                Image.fromarray(depth).save(self.output_path / "depths" / f"{i:04d}_depth.png")

                semantic = f["category_id_segmaps"][:]
                semantics = np.zeros((*semantic.shape, 3), dtype=np.uint8)
                for j, color in self.color_map.items():
                    semantics[semantic == int(j)] = color
                Image.fromarray(semantics).save(self.output_path / "semantics" / f"{i:04d}_mask.png")

                instance = f["instance_segmaps"][:]
                instances = np.zeros((*instance.shape, 3), dtype=np.uint8)
                for j, color in self.color_map.items():
                    instances[instance == int(j)] = color
                Image.fromarray(instances).save(self.output_path / "instances" / f"{i:04d}.png")

            (self.output_path / f"{i}.hdf5").unlink()

    def save_metadata(self):
        irr = self.irradiance
        dhi = self.hdr_energy_actual or 0.0
        dni = self.sun_energy_actual or 0.0
        zenith = irr.get("zenith")
        cos_z = math.cos(math.radians(zenith)) if zenith is not None else 0.0

        self.metadata["sun"] = {
            "azimuth": irr.get("azimuth"),
            "zenith": zenith,
            "elevation": irr.get("elevation"),
            "DNI_Wm2": irr.get("dni"),
            "DHI_Wm2": dhi,
            "GHI_Wm2": dni * cos_z + dhi,        # correct: DNI·cos(θ_z) + DHI
            "GHI_pvlib_Wm2": irr.get("ghi"),      # pvlib's own GHI for cross-check
            "airmass": irr.get("airmass"),
            "turbidity": self.turbidity,
            "in_civil_twilight": irr.get("in_civil_twilight"),
            "sun_color": self.sun_color_actual,
            "k_sun": self.k_sun,
            "nishita_fill_factor": self.nishita_fill_factor,
            "shadow_fill_boost": self.shadow_fill_boost,
            "blender_sun_energy": (dni * self.k_sun) if irr.get("sun_above_horizon") else 0.0,
            "blender_sky_strength": (dhi * self.k_sun / self.nishita_fill_factor),
            "date_time": self.date_time,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }
        with open(self.output_path / "meta_data.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

    def run(self):
        poi = bproc.object.compute_poi(self.scene_objects)
        for _ in range(self.num_frames):
            self.generate_camera_pose(poi)

        bproc.renderer.set_output_format(enable_transparency=self.enable_transparency)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

        render_data = bproc.renderer.render()
        bproc.writer.write_hdf5(str(self.output_path), render_data)
        self.save_images()
        self.save_metadata()

    @staticmethod
    def _get_color_map() -> dict[str, list[float]]:
        return {
            "0": [0, 0, 0],
            "1": [175, 200, 0],
            "2": [0, 200, 200],
            "3": [125, 0, 200],
            "4": [175, 0, 200],
            "5": [0, 50, 200],
            "6": [0, 200, 50],
            "7": [150, 50, 50],
            "8": [50, 175, 50],
            "9": [50, 50, 175],
        }


# ---------------------------------------------------------------------------

def main():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(BlenderProcRenderer).run()


if __name__ == "__main__":
    main()
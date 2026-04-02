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

#sys.path.extend(["/home/chexu/buildnet3d"])
#sys.path.extend([r"C:\Users\sefares\buildnet3d"])
sys.path.extend([str(Path(__file__).resolve().parents[2])])
from buildnet3d.utils.utils import build_translation

@dataclass
class RenderParams:
    """Parameters for rendering synthetic building images."""
    
    load_scene: Path = Path("bproc_generator/data/example/House.obj")
    """Path to the segmented building OBJ file"""
    output_path: Path = Path("outputs/generated")
    """Output directory for rendered assets"""
    resolution: tuple[int, int] = (512, 512)
    """Rendering resolution (width, height)"""
    #num_frames: int = 20
    num_frames: int = 5
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

    ## Sunlight parameters
    latitude: float = tyro.MISSING
    """Building location latitude (degrees)"""
    longitude: float = tyro.MISSING
    """Building location longitude (degrees)"""
    date_time: str = tyro.MISSING
    """Date and time for sun position (YYYY-MM-DD HH:MM:SS)"""
    
    # Lighting parameters
    use_hdr_background: bool = True
    """Enable procedural Nishita sky as DHI component — no HDR file needed"""
    # use_hdr_background: bool = False
    # """Enable HDR environment lighting"""
    # background_path: Path = Path("bproc_generator/data/example/zwartkops_straight_sunset_4k.hdr")
    """Path to HDR environment map"""
    # hdr_strength: float = 1.0
    hdr_strength: float = 50.0
    """Environment lighting intensity"""
    hdr_rotation: tuple[float, float, float] = (0.0, 0.0, 0.3926)
    """Environment rotation in radians (x,y,z)"""

    ## optional sun parameters
    use_sun: bool = True
    """Enable sun light source"""
    sun_energy: float = 800.0
    # sun_energy: float = 10.0
    # around 6 times higher than HDR
    """Sun light intensity"""
    north_offset_deg: float = 0.0
    """Rotation offset to align building model with true North (degrees).
        Set to 0 if the building's Y axis points North in the .obj file."""

## varying sun intensity and color based on zenith angle
def get_max_elevation(latitude: float, longitude: float, date_time: str) -> float:
    tz_str = pvlib.location.Location(latitude, longitude).tz
    tz_local = pytz.timezone(tz_str)
    dt_local = pd.Timestamp(date_time, tz=tz_local)
    dt_utc = dt_local.tz_convert("UTC")
    date = dt_utc.date()
    times = pd.date_range(
        start=f"{date} 00:00",
        end=f"{date} 23:59",
        freq="10min",
        tz="UTC"
    )
    solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    max_zenith = solar_pos["apparent_zenith"].min()
    return 90 - max_zenith

def sun_intensity_from_zenith(zenith_deg: float, max_energy: float = 800.0) -> float:
    """
    Computes sun intensity based on zenith angle.
    Lower sun (high zenith) = less intense, higher sun = more intense.
    """
    # Intensity follows a sine curve — peaks at noon, fades at horizon
    elevation_deg = 90 - zenith_deg
    intensity = max_energy * math.sin(math.radians(elevation_deg))
    return intensity

def hdr_intensity_from_zenith(zenith_deg: float, max_hdr: float = 50.0) -> float:
    """
    Computes DHI (diffuse sky light) intensity based on zenith angle.
    Follows the same sine curve as DNI but at 15% of peak DNI.
    Under clear sky: DHI ~ 15% of DNI at all times.
    """
    elevation_deg = 90 - zenith_deg
    intensity = max_hdr * math.sin(math.radians(elevation_deg))
    return intensity


def sun_color_from_zenith(zenith_deg: float) -> list:
    """
    Returns RGB color based on solar elevation following photography golden hour rules:
    - 0°–3°:   deep orange/reddish
    - 3°–8°:   warm orange
    - 8°–15°:  yellow-warm
    - 15°–30°: bright warm-neutral
    - 30°–50°: bright neutral
    - 50°+:    white / cool-neutral midday
    """
    elevation_deg = 90 - zenith_deg

    if elevation_deg <= 3:
        t = elevation_deg / 3.0
        r, g, b = 1.0, 0.25 + 0.10 * t, 0.05 + 0.10 * t

    elif elevation_deg <= 8:
        t = (elevation_deg - 3) / 5.0
        r, g, b = 1.0, 0.35 + 0.15 * t, 0.15 + 0.10 * t

    elif elevation_deg <= 15:
        t = (elevation_deg - 8) / 7.0
        r, g, b = 1.0, 0.50 + 0.20 * t, 0.25 + 0.20 * t

    elif elevation_deg <= 30:
        t = (elevation_deg - 15) / 15.0
        r, g, b = 1.0, 0.70 + 0.20 * t, 0.45 + 0.25 * t

    elif elevation_deg <= 50:
        t = (elevation_deg - 30) / 20.0
        r, g, b = 1.0, 0.90 + 0.08 * t, 0.70 + 0.25 * t

    else:
        r, g, b = 1.0, 1.0, 1.0

    return [round(r, 2), round(g, 2), round(b, 2)]


## conversion function
def solar_to_blender_rotation(azimuth_deg, zenith_deg, north_offset_deg=0):
    """
    Converts PSA solar angles to Blender sun rotation.
    azimuth_deg: 0=North, 90=East, clockwise
    zenith_deg: 0=overhead, 90=horizon
    north_offset_deg: building's North alignment in the .obj file
    """
    elevation_rad = math.radians(90 - zenith_deg)
    blender_azimuth_rad = math.radians(-(azimuth_deg + north_offset_deg))
    return (elevation_rad, 0, blender_azimuth_rad)    

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
        ## adding location and time info for sun position calculation
        self.metadata["location"] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            }
        self.metadata["capture_time"] = self.date_time

        ## sun state
        self.sun_azimuth = None
        self.sun_zenith = None
        self.sun_energy_actual = None
        self.sun_color_actual = None
        self.hdr_energy_actual = None

        # Initialize rendering pipeline
        bproc.init()
        self.scene_objects = bproc.loader.load_obj(str(self.load_scene))
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
        #if self.use_hdr_background:
        #    bproc.world.set_world_background_hdr_img(
        #        str(self.background_path),
        #        strength=self.hdr_strength,
        #        rotation_euler=self.hdr_rotation,
        #    )

        # Compute sun position first
        zenith = None
        azimuth = None
        if self.use_sun:
            tz_str = pvlib.location.Location(self.latitude, self.longitude).tz
            tz_local = pytz.timezone(tz_str)
            dt_local = pd.Timestamp(self.date_time, tz=tz_local)
            dt_utc = dt_local.tz_convert("UTC")
            dt = pd.DatetimeIndex([dt_utc])

            solar_pos = pvlib.solarposition.get_solarposition(
                dt, self.latitude, self.longitude
            )

            azimuth = solar_pos["azimuth"].values[0]
            zenith = solar_pos["apparent_zenith"].values[0]
            self.sun_azimuth = azimuth
            self.sun_zenith = zenith

        # # DHI component — HDR background
        # if self.use_hdr_background:
        #     if self.use_sun and zenith is not None and zenith < 90:
        #         effective_hdr_strength = hdr_intensity_from_zenith(zenith, self.hdr_strength)
        #     else:
        #         effective_hdr_strength = self.hdr_strength * 0.05  # night ambient
        #     bproc.world.set_world_background_hdr_img(
        #         str(self.background_path),
        #         strength=effective_hdr_strength,
        #         rotation_euler=self.hdr_rotation,
        #     )
        #     self.hdr_energy_actual = effective_hdr_strength

        # DHI component — Nishita procedural sky
        if self.use_hdr_background and zenith is not None and zenith < 90:
            world = bpy.data.worlds["World"]
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            nodes.clear()

            # Physically-based sky matching sun position
            sky = nodes.new("ShaderNodeTexSky")
            sky.sky_type = "NISHITA"
            sky.sun_elevation = math.radians(90 - zenith)
            sky.sun_rotation  = math.radians(azimuth)
            sky.altitude      = 400.0   # Lausanne altitude in meters
            sky.air_density   = 1.5
            sky.dust_density  = 0.3

            bg  = nodes.new("ShaderNodeBackground")
            # bg.inputs[1].default_value = self.hdr_strength / 150.0
            # bg.inputs[1].default_value = self.hdr_strength / 100.0

            effective_hdr_strength = hdr_intensity_from_zenith(zenith, self.hdr_strength)
            bg.inputs[1].default_value = effective_hdr_strength / 50.0
            self.hdr_energy_actual = effective_hdr_strength

            out = nodes.new("ShaderNodeOutputWorld")
            links.new(sky.outputs[0], bg.inputs[0])
            links.new(bg.outputs[0], out.inputs[0])

            # self.hdr_energy_actual = hdr_intensity_from_zenith(zenith, self.hdr_strength)
            # print(f"  Sky      : Nishita procedural (elevation={90-zenith:.1f}°)")

        elif self.use_hdr_background:
            # Night time — set sky to black
            world = bpy.data.worlds["World"]
            world.use_nodes = True
            nodes = world.node_tree.nodes
            nodes.clear()
            bg  = nodes.new("ShaderNodeBackground")
            bg.inputs[0].default_value = (0, 0, 0, 1)
            bg.inputs[1].default_value = 0.0
            out = nodes.new("ShaderNodeOutputWorld")
            world.node_tree.links.new(bg.outputs[0], out.inputs[0])
            self.hdr_energy_actual = 0.0

        # DNI component — direct sun light
        if self.use_sun and zenith is not None:
            if zenith >= 90:
                print(f"Sun is below horizon (zenith={zenith:.1f}°), skipping sun light.")
                return

            sun_rotation = solar_to_blender_rotation(
                azimuth, zenith, self.north_offset_deg
            )
            energy = sun_intensity_from_zenith(zenith, self.sun_energy)
            color = sun_color_from_zenith(zenith)

            self.sun_energy_actual = energy
            self.sun_color_actual = color

            sun = bproc.types.Light()
            sun.set_type("SUN")
            sun.set_energy(energy)
            sun.set_color(color)
            sun.blender_obj.rotation_euler = sun_rotation

            dhi = self.hdr_energy_actual or 0.0
            print(f"Sun placed:")
            print(f"  Azimuth  : {azimuth:.1f}°")
            print(f"  Zenith   : {zenith:.1f}°")
            print(f"  DNI      : {energy:.1f} W/m²")
            print(f"  DHI      : {dhi:.1f} W/m²")
            print(f"  GHI      : {energy + dhi:.1f} W/m²")
            print(f"  Color    : {color}")

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
        """Saves camera metadata in JSON format"""
        # Save sun info to metadata
        self.metadata["sun"] = {
            "azimuth": self.sun_azimuth,
            "zenith": self.sun_zenith,
            "DNI": self.sun_energy_actual,
            "DHI": self.hdr_energy_actual,  # new
            "GHI": (self.sun_energy_actual or 0) + (self.hdr_energy_actual or 0),  # new
            "color": self.sun_color_actual,
            "date_time": self.date_time,
            "latitude": self.latitude,
            "longitude": self.longitude,
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
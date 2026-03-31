import subprocess
import os
import pvlib
import pandas as pd

LAT = 46.5197
LON = 6.6323
DATE = "2024-06-21"
BASE = r"C:\Users\sefares\Desktop\renders_day2"

# Find sunrise and sunset automatically
times = pd.date_range(
    start=f"{DATE} 19:30",
    end=f"{DATE} 22:00",
    freq="1min",
    tz="UTC"
)
solar_pos = pvlib.solarposition.get_solarposition(times, LAT, LON)
above_horizon = solar_pos["apparent_zenith"] < 90

sunrise = times[above_horizon][0]
sunset  = times[above_horizon][-1]

print(f"Sunrise: {sunrise.strftime('%H:%M')} UTC")
print(f"Sunset:  {sunset.strftime('%H:%M')} UTC")

# Generate 30min intervals from sunrise to sunset
render_times = pd.date_range(
    start=sunrise.floor("30min"),
    end=sunset.floor("30min"),
    freq="30min",
    tz="UTC"
)

print(f"Rendering {len(render_times)} time slots...")

for t in render_times:
    datetime_str = t.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = os.path.join(BASE, t.strftime("%H-%M"))

    print(f"\n{'='*50}")
    print(f"Rendering {datetime_str}...")

    cmd = [
        "blenderproc", "run",
        "bproc_generator/render/generate.py",
        "--latitude", str(LAT),
        "--longitude", str(LON),
        "--date_time", datetime_str,
        "--output_path", out_dir,
        "--num_frames", "5",
        "--resolution", "512", "512"
    ]

    subprocess.run(cmd)

print("\nAll done!")
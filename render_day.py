import subprocess
import os

LAT = "46.5197"
LON = "6.6323"
DATE = "2024-06-21"
BASE = r"C:\Users\sefares\Desktop\renders_day"

times = [
    "04:30", "05:00", "05:30", "06:00", "06:30", "07:00", "07:30",
    "08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00",
    "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
    "15:00", "15:30", "16:00", "16:30", "17:00", "17:30", "18:00",
    "18:30", "19:00"
]

for t in times:
    datetime_str = f"{DATE} {t}:00"
    out_dir = os.path.join(BASE, t.replace(":", "-"))
    
    print(f"\n{'='*50}")
    print(f"Rendering {datetime_str}...")
    print(f"{'='*50}")
    
    cmd = [
        "blenderproc", "run",
        "bproc_generator/render/generate.py",
        "--latitude", LAT,
        "--longitude", LON,
        "--date_time", datetime_str,
        "--output_path", out_dir,
        "--num_frames", "3",
        "--resolution", "512", "512"
    ]
    
    subprocess.run(cmd)

print("\nAll done!")
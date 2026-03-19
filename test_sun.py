import pvlib
import pandas as pd
import math

def solar_to_blender_rotation(azimuth_deg, zenith_deg, north_offset_deg=0):
    elevation_rad = math.radians(90 - zenith_deg)
    blender_azimuth_rad = math.radians(-(azimuth_deg + north_offset_deg))
    return (elevation_rad, 0, blender_azimuth_rad)

# ---- CHANGE THESE TO TEST DIFFERENT CONDITIONS ----
latitude         = 46.5
longitude        = 6.6
north_offset_deg = 0.0
tests = [
    "2024-06-21 04:00:00",   # early morning
    "2024-06-21 06:00:00",   # sunrise
    "2024-06-21 08:00:00",   # morning
    "2024-06-21 10:00:00",   # mid morning
    "2024-06-21 12:00:00",   # noon
    "2024-06-21 14:00:00",   # afternoon
    "2024-06-21 18:00:00",   # evening
    "2024-06-21 21:00:00",   # near sunset
    "2024-06-21 00:00:00",   # night
]
# ---------------------------------------------------

print(f"\n{'='*65}")
print(f"  Sun Position Test — lat={latitude}, lon={longitude}")
print(f"{'='*65}")
print(f"{'Time (UTC)':<22} {'Azimuth':>10} {'Zenith':>10} {'Status':<15} {'Blender X':>10} {'Blender Z':>10}")
print(f"{'-'*65}")

for date_time in tests:
    dt = pd.DatetimeIndex([pd.Timestamp(date_time, tz="UTC")])
    solar_pos = pvlib.solarposition.get_solarposition(dt, latitude, longitude)

    azimuth = solar_pos["azimuth"].values[0]
    zenith  = solar_pos["apparent_zenith"].values[0]

    if zenith >= 90:
        status = "⚠️  below horizon"
        print(f"{date_time:<22} {azimuth:>9.1f}° {zenith:>9.1f}° {status}")
    else:
        status = "✅ above horizon"
        blender = solar_to_blender_rotation(azimuth, zenith, north_offset_deg)
        print(
            f"{date_time:<22} {azimuth:>9.1f}° {zenith:>9.1f}° "
            f"{status:<15} {math.degrees(blender[0]):>9.1f}° {math.degrees(blender[2]):>9.1f}°"
        )

print(f"{'='*65}")
print(f"\nExpected for Lausanne on June 21:")
print(f"  06:00 → azimuth ≈  80° (East),  zenith ≈ 75° (low)")
print(f"  12:00 → azimuth ≈ 175° (South), zenith ≈ 27° (high)")
print(f"  18:00 → azimuth ≈ 270° (West),  zenith ≈ 55° (medium)")
print(f"  00:00 → below horizon")
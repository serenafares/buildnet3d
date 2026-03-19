import pvlib
import pandas as pd
import math

def solar_to_blender_rotation(azimuth_deg, zenith_deg, north_offset_deg=0):
    elevation_rad = math.radians(90 - zenith_deg)
    blender_azimuth_rad = math.radians(-(azimuth_deg + north_offset_deg))
    return (elevation_rad, 0, blender_azimuth_rad)

def sun_intensity_from_zenith(zenith_deg: float, max_energy: float = 5.0) -> float:
    """Intensity peaks at noon, fades toward horizon"""
    elevation_deg = 90 - zenith_deg
    intensity = max_energy * math.sin(math.radians(elevation_deg))
    return max(0.1, intensity)

def get_max_elevation(latitude: float, date_time: str) -> float:
    """Gets the maximum solar elevation for that specific day at that location"""
    # Sample the full day to find peak elevation
    date = pd.Timestamp(date_time, tz="UTC").date()
    times = pd.date_range(
        start=f"{date} 00:00", 
        end=f"{date} 23:59", 
        freq="10min", 
        tz="UTC"
    )
    solar_pos = pvlib.solarposition.get_solarposition(
        times, latitude, longitude
    )
    max_zenith = solar_pos["apparent_zenith"].min()
    return 90 - max_zenith  # convert to elevation


def sun_color_from_zenith(zenith_deg: float, latitude: float, date_time: str) -> list:
    """
    Warm orange at horizon, white at the day's maximum elevation.
    Accounts for seasonal variation — winter stays warm longer.
    """
    elevation_deg = 90 - zenith_deg
    max_elev = get_max_elevation(latitude, date_time)
    
    # Normalize relative to today's maximum — 0=horizon, 1=peak of day
    t = min(1.0, elevation_deg / max_elev)
    
    r = 1.0
    g = 0.4 + 0.6 * t
    b = 0.2 + 0.8 * t
    return [round(r, 2), round(g, 2), round(b, 2)]

def color_to_emoji(color):
    """Visual representation of the color"""
    r, g, b = color
    if b > 0.9 and g > 0.9:
        return "⬜ white"
    elif b < 0.5 and g < 0.7:
        return "🟠 warm orange"
    elif b < 0.7:
        return "🟡 yellow-warm"
    else:
        return "🔆 bright"

# ---- CHANGE THESE TO TEST DIFFERENT CONDITIONS ----
latitude         = 46.5
longitude        = 6.6
north_offset_deg = 0.0
max_energy       = 5.0
tests = [
    "2024-06-21 04:00:00",   # early morning
    "2024-06-21 06:00:00",   # sunrise
    "2024-06-21 08:00:00",   # morning
    "2024-06-21 10:00:00",   # mid morning
    "2024-06-21 12:00:00",   # noon
    "2024-06-21 14:00:00",   # afternoon
    "2024-06-21 18:00:00",   # evening
    "2024-06-21 21:00:00",   # near sunset
    "2024-06-21 00:00:00",
    "2024-12-21 08:00:00",   # winter morning
    "2024-12-21 10:00:00",   # winter mid-morning
    "2024-12-21 12:00:00",   # winter noon
]

# ---------------------------------------------------

print(f"\n{'='*80}")
print(f"  Sun Position + Lighting Test — lat={latitude}, lon={longitude}")
print(f"{'='*80}")
print(f"{'Time (UTC)':<22} {'Azimuth':>9} {'Zenith':>9} {'Intensity':>10} {'Color RGB':>18} {'Mood'}")
print(f"{'-'*80}")

for date_time in tests:
    dt = pd.DatetimeIndex([pd.Timestamp(date_time, tz="UTC")])
    solar_pos = pvlib.solarposition.get_solarposition(dt, latitude, longitude)

    azimuth = solar_pos["azimuth"].values[0]
    zenith  = solar_pos["apparent_zenith"].values[0]

    if zenith >= 90:
        print(f"{date_time:<22} {azimuth:>9.1f}° {zenith:>9.1f}°  ⚠️  below horizon — no light")
    else:
        intensity = sun_intensity_from_zenith(zenith, max_energy)
        color = sun_color_from_zenith(zenith, latitude, date_time)
        mood      = color_to_emoji(color)
        print(
            f"{date_time:<22} {azimuth:>9.1f}° {zenith:>9.1f}° "
            f"{intensity:>9.2f}W  {str(color):>18}  {mood}"
        )

print(f"{'='*80}")
print(f"\nExpected behaviour:")
print(f"  Sunrise/Sunset → low intensity, warm orange color")
print(f"  Noon           → high intensity (~{max_energy}W), white color")
print(f"  Night          → no light placed")

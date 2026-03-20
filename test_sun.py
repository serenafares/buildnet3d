import pvlib
import pandas as pd
import math

# ─── Helper functions (mirrors generate.py) ───────────────────────────────────

def get_max_elevation(latitude, longitude, date_time):
    date = pd.Timestamp(date_time, tz="UTC").date()
    times = pd.date_range(
        start=f"{date} 00:00",
        end=f"{date} 23:59",
        freq="10min",
        tz="UTC"
    )
    solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    return 90 - solar_pos["apparent_zenith"].min()

def sun_intensity_from_zenith(zenith_deg, max_energy=1000.0):
    elevation_deg = 90 - zenith_deg
    return max_energy * math.sin(math.radians(elevation_deg))

def hdr_intensity_from_zenith(zenith_deg, max_hdr=150.0):
    elevation_deg = 90 - zenith_deg
    return max_hdr * math.sin(math.radians(elevation_deg))



def sun_color_from_zenith(zenith_deg: float, latitude: float, longitude: float, date_time: str) -> list:
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
        # Deep orange/reddish — just above horizon
        t = elevation_deg / 3.0
        r, g, b = 1.0, 0.25 + 0.10 * t, 0.05 + 0.10 * t

    elif elevation_deg <= 8:
        # Warm orange
        t = (elevation_deg - 3) / 5.0
        r, g, b = 1.0, 0.35 + 0.15 * t, 0.15 + 0.10 * t

    elif elevation_deg <= 15:
        # Yellow-warm
        t = (elevation_deg - 8) / 7.0
        r, g, b = 1.0, 0.50 + 0.20 * t, 0.25 + 0.20 * t

    elif elevation_deg <= 30:
        # Bright warm-neutral
        t = (elevation_deg - 15) / 15.0
        r, g, b = 1.0, 0.70 + 0.20 * t, 0.45 + 0.25 * t

    elif elevation_deg <= 50:
        # Bright neutral
        t = (elevation_deg - 30) / 20.0
        r, g, b = 1.0, 0.90 + 0.08 * t, 0.70 + 0.25 * t

    else:
        # White / cool-neutral midday
        r, g, b = 1.0, 1.0, 1.0

    return [round(r, 2), round(g, 2), round(b, 2)]


def color_to_label(color):
    r, g, b = color
    if b >= 0.95 and g >= 0.98:
        return "⬜ white"
    elif b >= 0.70 and g >= 0.90:
        return "🔆 bright neutral"
    elif b >= 0.45 and g >= 0.70:
        return "🔆 bright warm"
    elif b >= 0.25 and g >= 0.50:
        return "🟡 yellow-warm"
    elif b >= 0.15 and g >= 0.35:
        return "🟠 warm orange"
    else:
        return "🔴 deep orange"

# ─── Settings ─────────────────────────────────────────────────────────────────

latitude  = 46.5
longitude = 6.6
max_dni   = 1000.0
max_dhi   = 150.0

seasons = {
    "☀️  Summer (June 21)":   "2024-06-21",
    "❄️  Winter (Dec 21)":    "2024-12-21",
}

# ─── Run ──────────────────────────────────────────────────────────────────────

for season_label, date in seasons.items():
    print(f"\n{'='*85}")
    print(f"  {season_label} — lat={latitude}, lon={longitude}")
    print(f"{'='*85}")
    print(f"{'Time':>8} {'Azimuth':>9} {'Zenith':>9} {'DNI':>8} {'DHI':>8} {'GHI':>8} {'Color RGB':>18} {'Mood'}")
    print(f"{'-'*85}")

    # Every 30 minutes
    times = pd.date_range(
        start=f"{date} 00:00",
        end=f"{date} 23:30",
        freq="30min",
        tz="UTC"
    )

    for t in times:
        dt = pd.DatetimeIndex([t])
        solar_pos = pvlib.solarposition.get_solarposition(dt, latitude, longitude)
        azimuth = solar_pos["azimuth"].values[0]
        zenith  = solar_pos["apparent_zenith"].values[0]
        time_str = t.strftime("%H:%M")

        if zenith >= 90:
            print(f"{time_str:>8} {azimuth:>9.1f}° {zenith:>9.1f}°  ⚠️  below horizon")
        else:
            dni   = sun_intensity_from_zenith(zenith, max_dni)
            dhi   = hdr_intensity_from_zenith(zenith, max_dhi)
            ghi   = dni + dhi
            color = sun_color_from_zenith(zenith, latitude, longitude, f"{date} 12:00:00")
            mood  = color_to_label(color)
            print(
                f"{time_str:>8} {azimuth:>9.1f}° {zenith:>9.1f}° "
                f"{dni:>7.1f}W {dhi:>7.1f}W {ghi:>7.1f}W "
                f"{str(color):>18}  {mood}"
            )

    print(f"{'='*85}")
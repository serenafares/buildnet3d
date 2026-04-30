@echo off
setlocal enabledelayedexpansion

set LAT=46.5197
set LON=6.6323
set DATE=2026-03-21
set BASE_OUT=C:\Users\sefares\Desktop\renders

blenderproc run bproc_generator/render/generate_try.py ^
  --latitude 46.5 ^
  --longitude 6.6 ^
  --date 2024-06-21 ^
  --output-path outputs/generated

echo Done.
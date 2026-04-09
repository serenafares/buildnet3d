@echo off
setlocal enabledelayedexpansion

set LAT=46.5197
set LON=6.6323
set DATE=2024-06-21
set BASE_OUT=C:\Users\sefares\Desktop\renders

for %%H in (04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21) do (
    for %%M in (00 30) do (
        set OUT=!BASE_OUT!\%%H_%%M
        echo Running %%H:%%M UTC ...
        blenderproc run bproc_generator/render/generate_try.py ^
            --latitude %LAT% ^
            --longitude %LON% ^
            --date_time "%DATE% %%H:%%M:00" ^
            --output_path !OUT!
    )
)

echo Done.
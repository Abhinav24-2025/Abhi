# Runoff component animation

Animated stacked-area chart of basin runoff split into **snowmelt, glacier melt,
rainfall runoff and baseflow**, with an optional observed-discharge line and a
side panel showing each component's current value and % share.

## Install
    pip install matplotlib pandas numpy
    # for .mp4 output you also need ffmpeg on PATH (GIF works without it)

## Input
CSV, one row per time step. See `example_input.csv` (template only, values are made up):

    date,snowmelt,glacier_melt,rainfall,baseflow,observed

`observed` is optional. If your columns are named differently, map them with `--cols`.

## Run
    python runoff_animation.py --demo --out demo.mp4                 # synthetic test data
    python runoff_animation.py --csv my_basin.csv --basin "My basin" --out runoff.mp4
    python runoff_animation.py --csv out.csv --date-col Date \
        --cols "Qsnow=snowmelt,Qglac=glacier_melt,Qrain=rainfall,Qbase=baseflow" \
        --observed-col Qobs --out runoff.gif

Useful options: `--start/--end` (subset period), `--resample W|MS` (weekly/monthly mean),
`--window 365` (scrolling 1-year view), `--step` (time steps per frame — controls length),
`--fps`, `--dpi`, `--units`.

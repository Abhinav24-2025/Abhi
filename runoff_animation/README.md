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

### SPHY-style monthly output (detected automatically)
A table with `Years, Months, QALLDTS, STotDTS, RTotDTS, GTotDTS, BTotDTS` works as-is
(.xlsx, .csv or tab-separated .txt). The year only needs to be on the January row,
and month names may have trailing spaces. Column mapping used:

| Column  | Plotted as |
|---------|------------|
| STotDTS | Snowmelt |
| GTotDTS | Glacier melt |
| RTotDTS | Rainfall runoff |
| BTotDTS | Baseflow |
| QALLDTS | not plotted, only used to check that the four components add up to it |

    python runoff_animation.py --csv sphy_output.xlsx --basin "My basin" --out runoff.mp4
    python runoff_animation.py --csv sphy_output.xlsx --sheet Sheet2 --out runoff.gif

For .xlsx input also `pip install openpyxl`.

## Run
    python runoff_animation.py --demo --out demo.mp4                 # synthetic test data
    python runoff_animation.py --csv my_basin.csv --basin "My basin" --out runoff.mp4
    python runoff_animation.py --csv out.csv --date-col Date \
        --cols "Qsnow=snowmelt,Qglac=glacier_melt,Qrain=rainfall,Qbase=baseflow" \
        --observed-col Qobs --out runoff.gif

Useful options: `--start/--end` (subset period), `--resample W|MS` (weekly/monthly mean),
`--window 365` (scrolling 1-year view), `--step` (time steps per frame — controls length),
`--fps`, `--dpi`, `--units`.

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

---

# Basin map animation (runoff flowing through the river network)

`basin_flow_animation.py` animates coloured particles (snowmelt = white, glacier melt = cyan,
rainfall = blue, baseflow = amber) travelling down your real river network to the outlet,
over a hillshade of your DEM, with live values per component and a timeline.

    pip install geopandas rasterio scipy matplotlib pandas openpyxl   # + ffmpeg for .mp4
    python basin_flow_animation.py --boundary basin.shp --rivers rivers.shp --dem dem.tif \
        --data sphy_output.xlsx --basin "My basin" --start 2000-01 --end 2002-12 --out basin_flow.mp4

Inputs can be in any CRS (they are reprojected to UTM). Rivers can be digitised in any
direction: flow direction comes from the network path to the outlet. The outlet is the lowest
river point on the DEM unless you give `--outlet lon,lat`. Optional `--glaciers rgi.shp` makes
glacier-melt particles start inside glacier outlines.

**What is data and what is illustrative:** the amount of each component through time comes
from your table (basin-outlet totals). Where particles *start* is illustrative (snowmelt on
high ground, glacier melt on the highest ground or in glacier outlines, rain and baseflow
everywhere), because the table has no per-cell values. Values are interpolated between months,
and particle speed is not to scale. The video footnote states this.

Tuning: `--frames-per-step` (frames per month), `--fps`, `--travel-frames` (how fast particles
cross the basin), `--max-spawn` (particle density), `--snap` (metres, to join river lines that
do not share vertices exactly).

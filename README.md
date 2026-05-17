# County-Level Power Outages, Data Centers, and Socioeconomic Context

## General description

This project builds a **county-level (FIPS-keyed)** dataset that combines **EAGLE-I power outage signals**, **data center locations with estimated power use and carbon**, **USDA ERS–style demographics** (population, education, poverty, unemployment), and **county temperature** statistics. You can run pipelines to **clean and align** sources, **aggregate** outages into events, **merge** everything on `fips_code`, and optionally run **correlation** and **machine-learning** experiments (e.g. predicting total outage hours from features).

The scientific question motivating the code is whether **data center intensity / energy use** and other county factors are associated with **outage burden**, at a coarse geographic resolution suitable for public county data.

---

## Project structure

```
Merge/                          # repository root (run many scripts from here)
├── DS/                         # data centers + Census county boundaries
├── Outage/                     # outage aggregation + aggregated CSV
├── Population/                 # demographic CSV processing + merged table
├── Temperature/                # temperature CSV + FIPS assignment
├── eaglei_outages_2023.csv     # raw EAGLE-I–style outages (if present)
├── Correlation analysis for a lot of new variables.py
├── Inspect.py
├── Let's try to do some ML magics here.py
├── Merged for ML.csv           # produced after correlation script (if run)
├── Correlation_Matrix_*.csv    # produced after correlation script (if run)
├── README.md
├── Outage and DS.iml           # IDE module file (not part of analysis)
└── .idea/                      # JetBrains metadata (not part of analysis)
```

**Path convention:** Inputs and outputs are resolved with **`pathlib` and `Path(__file__)`** (repo root or the script’s own folder). You can run any script from **any working directory**; you no longer need to `cd` into a specific folder first.

---

## What you can do with this project (core code, datasets, and outcomes)

Each row below is: **capability → what to run → what you get.**

| You can… | Run | Result |
|----------|-----|--------|
| **Assign each data center a county FIPS** using lat/lon and a Census county shapefile | `python "DS/Data center to fip.py"` | `DS/Merge_BA_Carbon_with_FIPS.csv` — one row per facility with `fips_code` (needs `DS/Merge_BA_Carbon.csv` and `DS/tl_2024_us_county.shp`) |
| **Roll up all data centers in a county** into list-style fields (names, power kWh, carbon tons) | `python "DS/Group data centers by fip codes.py"` | `DS/Grouped_By_FIPS_List.csv` — one row per `fips_code` |
| **Turn 15-minute EAGLE-I outage rows into longer “events”** (gap ≤ 2 h merges samples; `sum ≥ 10` filter) | `python "Outage/aggregate outages.py"` | `Outage/Aggregated_Outage_Events.csv` — reads **`eaglei_outages_2023.csv`** from the **repo root** |
| **Build wide county demographic tables** from long-form USDA-style CSVs | Each `python "Population/Population data concate.py"` (and education / poverty / unemployment) | `Population/Processed_*.csv` |
| **Merge those processed tables into one demographics file** | `python "Population/Merge to once file.py"` | `Population/DemographicsData.csv` |
| **Add FIPS to county temperature rows** using county name + state and an adjacency/GEOID lookup file | `python "Temperature/Assign FIPS to counties.py"` | `Temperature/Temperature Data with FIPS.csv` |
| **Merge outages, data centers, demographics, and temperature; compute Pearson correlations; export a modeling table** | `python "Correlation analysis for a lot of new variables.py"` | Repo root: `Merged for ML.csv`, `Correlation_Matrix_Only_Counties_With_Data_Centers.csv`, `Correlation_Matrix_All_Counties.csv`, plus plots |
| **Quick correlation heatmaps (outages + data centers only)** | `python Inspect.py` | Same style of plots as part of the full pipeline, without demographics or temperature |
| **Train a baseline regressor and run interpretability methods** on the merged table | `python "Let's try to do some ML magics here.py"` (after `Merged for ML.csv` exists) | Console metrics (MAE, RMSE), importance tables/plots; target is `sum_duration_hrs`; LIME prints a feature list in plain terminals if the notebook viewer is unavailable |

**Core datasets for analysis (after pipelines):**

- `Outage/Aggregated_Outage_Events.csv` — outage-derived targets and list columns.
- `DS/Grouped_By_FIPS_List.csv` — data-center-related features per county.
- `Population/DemographicsData.csv` — socioeconomic / population features per county.
- `Temperature/Temperature Data with FIPS.csv` — temperature metrics with `FIPS` (renamed to `fips_code` in the correlation script).

**Raw inputs you must obtain separately** (not all are committed): `Merge_BA_Carbon.csv`, raw Population source CSVs, `tl_2024_us_county.*`, `county_adjacency2024.txt`, and optionally a fresh `eaglei_outages_2023.csv` to regenerate aggregates.

---

## Outdated, redundant, or non-essential code and data

These items are **overlapped by other parts of the repo**, **only needed to regenerate** intermediates, or **not part of the analysis** at all.

### Redundant or superseded scripts

- **`Inspect.py`** — **Subset** of the full merge: only **outages** plus **`Grouped_By_FIPS_List`**. Paths are fixed to match the rest of the repo. **Prefer `Correlation analysis for a lot of new variables.py`** when you want demographics, temperature, and `Merged for ML.csv`.

### Intermediate datasets (safe to treat as rebuild-only once downstream exists)

- **`DS/Merge_BA_Carbon_with_FIPS.csv`** — Facility-level table after spatial join. **Downstream analysis uses** `Grouped_By_FIPS_List.csv`. Keep this file if you need per–data center rows or want to rerun grouping without redoing the spatial join.
- **`Population/Processed_*.csv`** — Stepping stones to **`DemographicsData.csv`**. If `DemographicsData.csv` is up to date, you do not need to open the processed files for modeling.
- **`eaglei_outages_2023.csv`** (raw) — Only required to **recompute** `Aggregated_Outage_Events.csv`. Correlation and ML use the **aggregated** file.
- **`Temperature/Temperature Data.csv`** — Superseded for merged analysis by **`Temperature Data with FIPS.csv`** once FIPS assignment has been run; keep the original if you need to refresh FIPS mapping.

### Auxiliary / non-analysis files

- **`.idea/`**, **`Outage and DS.iml`** — IDE configuration; not used by the Python pipelines.
- **`DS/tl_2024_us_county.shp.iso.xml`**, **`DS/tl_2024_us_county.shp.ea.iso.xml`** — Shapefile metadata; GeoPandas needs the core `.shp/.shx/.dbf/.prj` set, not necessarily these XML files.

---

## Dependencies

| Area | Packages |
|------|----------|
| Tables and plots | **pandas**, **numpy**, **matplotlib**, **seaborn** |
| Spatial join (`Data center to fip.py`) | **geopandas**, **shapely** |
| Correlation / ML script | **scikit-learn** |
| Interpretability (ML script) | **shap**, **lime** |
| Optional deep model path | **pytorch-tabnet** (guarded import; skip if not installed) |

Python **3.x** is assumed; use a virtual environment and install versions compatible with your GeoPandas stack on Windows.

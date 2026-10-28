# Harvest Compass

Harvest Compass is a Python and Streamlit dashboard for the NISR 2026 Agriculture Productivity track. It brings together three NISR Seasonal Agricultural Surveys (SAS) so agricultural planners can compare district crop yield and production estimates across seasons and years.

## What it does

- Filters SAS estimates by season, crop, year, and district.
- Provides English and Kinyarwanda interface text and sidebar navigation.
- Shows a three-year yield trend for the selected district alongside NISR's national estimate when published.
- Maps district yield or production estimates as a choropleth using the NISR district boundaries.
- Screens an uploaded cassava leaf photo for four common disease categories or a healthy leaf.
- Ranks district yield estimates for the selected year, crop, and season.
- Shows the selected district's production estimate and its change from the 2023 estimate when comparable data exist.
- Explains data limits so descriptive comparisons are not mistaken for causal evidence.

## Run locally on Windows

From the project folder, create the environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the app and CPU-only PyTorch versions for reproducible local and cloud installs. Use Python 3.12.

Run the app through that environment's Python (this works even when `streamlit` is not on the global `PATH`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open the local URL printed in the terminal. If the `py` launcher is unavailable, use `python -m venv .venv` if Python is on `PATH`, or install Python first.

The NISR annual workbooks must be in `data/`:

- `SAS_2023_Annual_Tables.xlsx`
- `SAS_2024_Annual_Tables.xlsx`
- `SAS_2025_Annual_Tables.xlsx`

## Cassava image model

The Cassava AI check uses the Kaggle [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) dataset. Accept Kaggle's competition rules, download the ZIP, and place it in the project folder. Extract the labeled training files only (the ZIP's `train_images/`, `train.csv`, and `label_num_to_disease_map.json`) into `data/cassava/`. The training script uses a pretrained MobileNetV3-Small feature extractor; the first run downloads its public ImageNet weights.

Train a CPU model and save the artifact used by the app:

```powershell
.\.venv\Scripts\python.exe scripts\train_cassava.py --epochs 18
```

The script reports held-out validation accuracy and per-class accuracy. The trained file is `models/cassava_mobilenet_v3_small.pt`; validation details are saved beside it as JSON. Keep the Kaggle ZIP and extracted image directory out of Git because they are large. The dashboard's similarity check can reject many unrelated images, but it cannot guarantee recognition of every other crop. The Kaggle photos were collected in Uganda, so validate with Rwandan field photos before relying on results. Treat predictions as screening suggestions and ask agricultural extension staff to verify suspected disease.

## Data

- `data/SAS_2023_Annual_Tables.xlsx` — [NISR 2023 SAS](https://alpha.statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2023)
- `data/SAS_2024_Annual_Tables.xlsx` — [NISR 2024 SAS](https://www.statistics.gov.rw/statistical-publications/agriculture-environment/seasonal-agricultural-survey-2024-annual-report-2024)
- `data/SAS_2025_Annual_Tables.xlsx` — [NISR 2025 SAS](https://statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2025-)
- The app reads district yield and production tables from each annual workbook and uses the reported `National` estimate when available.
- `assets/rwanda_districts.geojson` — NISR district boundary layer, served by the NAEB GIS service: [FeatureServer layer](https://gis.naeb.gov.rw/server/rest/services/Hosted/Administrative_Boundaries_WFL1/FeatureServer/0).

District and year comparisons are descriptive. Changes or gaps identify patterns for follow-up; they do not establish why yields changed or prove an intervention caused an outcome. Check each NISR report for survey design, definitions, and uncertainty.

## Deploy on Streamlit Community Cloud

1. Push the public repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and connect the account to public repositories.
3. Create an app from `Kali-AI02/Data_Duo2026`, branch `main`, entrypoint `app.py`; choose Python 3.12 in advanced settings.
4. Wait for the dependency build and app health check to finish, then open the generated `streamlit.app` URL and exercise each page on desktop and phone.
5. Paste that public URL into the NISR submission form together with the public GitHub URL.

The project includes the trained model, annual SAS workbooks, district boundaries, and Python dependencies, so deployment does not need the 6.2 GB Kaggle ZIP or extracted image folder. The app needs network access only to install pinned dependencies during its cloud build; the dashboard itself reads project-local data and model files.

See [SUBMISSION_BRIEF.md](SUBMISSION_BRIEF.md) for the rubric-aligned project explanation, sources, validation results, limitations, and AI disclosure.

## AI assistance disclosure

This repository was created with AI assistance. The team should review and understand the code and analysis, verify outputs against NISR publications, and disclose AI assistance in the competition submission as required by the competition rules.

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from cassava_detector import CLASS_NAMES, MODEL_FILE, extract_embedding, load_model_bundle, prepare_image


ROOT = Path(__file__).parent
LOGO_FILE = ROOT / "assets" / "nisr_logo.jpeg"
DISTRICT_GEOJSON = ROOT / "assets" / "rwanda_districts.geojson"
BOUNDARY_SOURCE = "https://gis.naeb.gov.rw/server/rest/services/Hosted/Administrative_Boundaries_WFL1/FeatureServer/0"
YEAR_FILES = {
    2023: "SAS_2023_Annual_Tables.xlsx",
    2024: "SAS_2024_Annual_Tables.xlsx",
    2025: "SAS_2025_Annual_Tables.xlsx",
}
YEAR_TABLES = {
    2023: {"yield": {"A": "Table 16", "B": "Table 17", "C": "Table 18"},
           "production": {"A": "Table 22", "B": "Table 23", "C": "Table 24"}},
    2024: {"yield": {"A": "Table 16", "B": "Table 17", "C": "Table 18"},
           "production": {"A": "Table 21", "B": "Table 22", "C": "Table 23"}},
    2025: {"yield": {"A": "Table 19", "B": "Table 20", "C": "Table 21"},
           "production": {"A": "Table 24", "B": "Table 25", "C": "Table 26"}},
}
SOURCE_URLS = {
    2023: "https://alpha.statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2023",
    2024: "https://www.statistics.gov.rw/statistical-publications/agriculture-environment/seasonal-agricultural-survey-2024-annual-report-2024",
    2025: "https://statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2025-",
}
DISTRICTS = {
    "Nyarugenge", "Gasabo", "Kicukiro", "Nyanza", "Gisagara", "Ruhango", "Muhanga", "Kamonyi",
    "Karongi", "Rutsiro", "Rubavu", "Nyabihu", "Ngororero", "Rusizi", "Nyamasheke", "Nyamagabe",
    "Huye", "Nyaruguru", "Musanze", "Burera", "Gakenke", "Gicumbi", "Rulindo", "Gatsibo",
    "Nyagatare", "Kayonza", "Rwamagana", "Kirehe", "Ngoma", "Bugesera",
}
GROUP_HEADERS = {
    "cereals", "tubers and roots", "bananas", "banana", "legumes", "beans",
    "other crops", "total", "district/crop", "district", "all crops",
}
CROP_NAMES = {
    "sweet potatoes": "Sweet potato", "sweet potato": "Sweet potato",
    "irish potatoes": "Irish potato", "irish potato": "Irish potato",
    "beans": "Bean", "bean": "Bean", "soya beans": "Soybean", "soya bean": "Soybean",
    "soybeans": "Soybean", "soybean": "Soybean", "ground nuts": "Groundnut", "groundnut": "Groundnut",
    "paddy rice": "Paddy rice", "cooking banana": "Cooking banana", "dessert banana": "Dessert banana",
    "banana for beer": "Banana for beer",
}


@st.cache_data(show_spinner=False)
def load_sas(root: str, modified: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    del modified  # Included in the cache key so replaced workbooks are picked up.
    yields: list[pd.DataFrame] = []
    production: list[pd.DataFrame] = []

    for year, filename in YEAR_FILES.items():
        path = str(Path(root) / "data" / filename)
        for season, sheet in YEAR_TABLES[year]["yield"].items():
            yields.append(_read_wide_table(path, sheet, year, season, "Yield (kg/ha)"))
        for season, sheet in YEAR_TABLES[year]["production"].items():
            production.append(_read_wide_table(path, sheet, year, season, "Production (MT)"))

    return pd.concat(yields, ignore_index=True), pd.concat(production, ignore_index=True)


def _read_wide_table(path: str, sheet: str, year: int, season: str, metric: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_index = next(
        (i for i in range(min(5, len(raw)))
         if any(str(value).strip().lower().startswith("district") for value in raw.iloc[i].tolist())),
        None,
    )
    if header_index is None:
        # Some Season C tables omit the district label in the header row but keep it in column B.
        header_index, district_col = 3, 1
    else:
        district_col = next(
            j for j, value in enumerate(raw.iloc[header_index].tolist())
            if str(value).strip().lower().startswith("district")
        )

    header_values = raw.iloc[header_index].tolist()
    # Some older NISR tables place crop names on the row above the district heading.
    header_values = [
        value if not pd.isna(value) and str(value).strip() else next(
            (raw.iat[previous, column_index]
             for previous in range(header_index - 1, max(-1, header_index - 3), -1)
             if not pd.isna(raw.iat[previous, column_index]) and str(raw.iat[previous, column_index]).strip()),
            value,
        )
        for column_index, value in enumerate(header_values)
    ]
    data_start = header_index + 1
    col_to_crop: dict[str, str] = {}
    for column_index, value in enumerate(header_values):
        if column_index == district_col or pd.isna(value):
            continue
        crop_name = " ".join(str(value).replace("\n", " ").split())
        key = crop_name.lower()
        if not key or key in GROUP_HEADERS or key.startswith("unnamed") or key.startswith("total"):
            continue
        col_to_crop[f"crop_{column_index}"] = CROP_NAMES.get(key, crop_name)

    table = pd.DataFrame({"District": raw.iloc[data_start:, district_col].astype("string").str.strip()})
    for column_index in range(len(header_values)):
        internal = f"crop_{column_index}"
        if internal in col_to_crop:
            table[internal] = raw.iloc[data_start:, column_index].values
    if not col_to_crop:
        raise ValueError(f"No crop columns found in {sheet} ({year}, Season {season}).")

    long = table.melt(
        id_vars="District", value_vars=list(col_to_crop),
        var_name="CropKey", value_name="Value",
    )
    long["Value"] = pd.to_numeric(
        long["Value"].astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )
    long = long.dropna(subset=["District", "Value"])
    long["District"] = long["District"].astype(str).str.strip()
    long["Crop"] = long["CropKey"].map(col_to_crop)
    long = long[long["District"].isin(DISTRICTS | {"National", "National estimate", "National average"})]
    long["Year"] = year
    long["Season"] = season
    long["Metric"] = metric
    return long.drop(columns="CropKey")


@st.cache_data(show_spinner=False)
def load_district_boundaries(path: str, modified: float) -> dict:
    del modified
    with open(path, encoding="utf-8") as boundary_file:
        return json.load(boundary_file)


@st.cache_resource(show_spinner=False)
def load_cassava_bundle(path: str, modified: float):
    del modified
    return load_model_bundle(Path(path))


st.set_page_config(
    page_title="Harvest Compass | Rwanda",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    display_mode = st.radio("Display mode", ["Light", "Dark"], horizontal=True)

if display_mode == "Dark":
    theme = {"paper":"#111a1c", "sidebar":"#162326", "ink":"#edf4ef", "muted":"#b8c8c0", "border":"#30413e", "hero":"#193b32", "card":"#1b2b2d", "insight":"#243a30"}
else:
    theme = {"paper":"#f5f7f1", "sidebar":"#eef2e8", "ink":"#162d26", "muted":"#64756b", "border":"#dfe5da", "hero":"#163c2f", "card":"#ffffff", "insight":"#e7efdb"}

logo_data = base64.b64encode(LOGO_FILE.read_bytes()).decode("ascii") if LOGO_FILE.exists() else ""

style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:__INK__; --muted:__MUTED__; --paper:__PAPER__; --text-color:__INK__; --background-color:__PAPER__; --secondary-background-color:__SIDEBAR__; --primary-color:#287d55; color-scheme:__MODE__; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:__INK__; }
    .stApp, [data-testid="stAppViewContainer"] { background:__PAPER__; color:__INK__; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:__SIDEBAR__; border-right:1px solid __BORDER__; }
    .block-container { padding-top:2rem; max-width:1440px; }
    h1,h2,h3,h4,h5,h6 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:__INK__; }
    .topbar { display:flex; align-items:center; gap:16px; background:#1749a3; color:white; border-radius:0 0 18px 18px; padding:12px 22px; margin:-2rem -1rem 1.5rem; box-shadow:0 5px 18px rgba(18,44,85,.18); }
    .topbar img { width:60px; height:60px; object-fit:contain; background:white; border-radius:12px; padding:4px; }
    .topbar-title { color:#fff; font:700 1.08rem 'Manrope',sans-serif; } .topbar-caption { color:#e3eaff; font-size:.78rem; margin-top:2px; }
    .hero { background:__HERO__; border-radius:24px; padding:1.65rem 2rem; color:#fff; margin-bottom:1.25rem; position:relative; overflow:hidden; }
    .hero:after { content:'✳'; position:absolute; right:3%; top:-35%; font-size:18rem; line-height:1; color:rgba(217,243,106,.09); transform:rotate(-12deg); }
    .eyebrow { color:#c7e66c; font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .hero h1 { color:white; font-size:2.3rem; margin:.55rem 0 .35rem; }
    .hero p { color:#d1dfd7; max-width:750px; margin:0; font-size:1rem; }
    .kpi { background:__CARD__; border:1px solid __BORDER__; border-radius:18px; padding:1.1rem 1.2rem; min-height:118px; box-shadow:0 6px 20px rgba(22,45,38,.035); }
    .kpi-label { color:__MUTED__; font-size:.78rem; font-weight:600; }
    .kpi-value { color:__INK__; font:800 1.8rem 'Manrope',sans-serif; margin:.35rem 0 .15rem; }
    .kpi-note { color:__MUTED__; font-size:.75rem; }
    .section-title { margin:1.2rem 0 .2rem; }
    .section-sub { color:__MUTED__; margin:0 0 .9rem; }
    .insight { background:__INSIGHT__; border-left:4px solid #77a33a; padding:1rem 1.1rem; border-radius:0 14px 14px 0; margin:.35rem 0; color:__INK__; }
    .source-note { font-size:.78rem; color:__MUTED__; padding-top:1rem; border-top:1px solid __BORDER__; }
    div[data-testid="stMetric"] { background:__CARD__; border:1px solid __BORDER__; border-radius:16px; padding:14px; }
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:__INK__; }
    [data-testid="stTabs"] button, [data-testid="stTabs"] button p { color:__INK__; }
    [data-testid="stCaptionContainer"] { color:__MUTED__; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div { background-color:__CARD__; border-color:__BORDER__; }
    [data-baseweb="select"] *, [data-baseweb="input"] input, [data-testid="stRadio"] label { color:__INK__; }
    @media(max-width:700px) { .topbar { margin:-1.5rem -1rem 1rem; } .topbar img { width:48px;height:48px; } .hero { padding:1.25rem; } .hero h1 {font-size:1.75rem;} }
    </style>
"""
for token, value in {"INK": theme["ink"], "MUTED": theme["muted"], "PAPER": theme["paper"], "SIDEBAR": theme["sidebar"], "BORDER": theme["border"], "HERO": theme["hero"], "CARD": theme["card"], "INSIGHT": theme["insight"], "MODE": display_mode.lower()}.items():
    style = style.replace(f"__{token}__", value)
st.markdown(style, unsafe_allow_html=True)

logo_html = f'<img src="data:image/jpeg;base64,{logo_data}" alt="NISR logo">' if logo_data else ""
st.markdown(
    f'<div class="topbar">{logo_html}<div><div class="topbar-title">Harvest Compass · Rwanda Agriculture</div><div class="topbar-caption">National Institute of Statistics of Rwanda · SAS 2023–2025</div></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Agriculture data for district-level decisions</div>
      <h1>Harvest Compass</h1>
      <p>Track crop yields across seasons and years. Compare your district with Rwanda’s reported estimates and spot patterns worth investigating.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing_files = [name for name in YEAR_FILES.values() if not (ROOT / "data" / name).exists()]
if missing_files:
    st.error("Missing source workbook(s): " + ", ".join(missing_files))
    st.stop()

try:
    cache_key = tuple((ROOT / "data" / name).stat().st_mtime for name in YEAR_FILES.values())
    yields, production = load_sas(str(ROOT), cache_key)
except Exception as exc:
    st.error(f"Could not read the NISR tables: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Explore the harvest")
    st.caption("Compare the same crop across districts, seasons, and survey years.")
    season = st.selectbox("Growing season", ["A", "B", "C"], format_func=lambda x: f"Season {x}")
    focus_year = st.selectbox("District snapshot", [2025, 2024, 2023])
    season_yields = yields[yields["Season"].eq(season)]
    crops = sorted(season_yields.loc[season_yields["District"].isin(DISTRICTS) & season_yields["Value"].gt(0), "Crop"].dropna().unique())
    if not crops:
        st.error(f"No crop yield records were found for Season {season}.")
        st.stop()
    preferred = next((crop for crop in crops if crop.lower() == "maize"), crops[0])
    crop = st.selectbox("Crop", crops, index=crops.index(preferred))
    focus_rows = season_yields[(season_yields["Year"].eq(focus_year)) & season_yields["Crop"].eq(crop)]
    district_options = sorted(focus_rows.loc[focus_rows["District"].isin(DISTRICTS), "District"].unique())
    if not district_options:
        district_options = sorted(season_yields.loc[season_yields["Crop"].eq(crop) & season_yields["District"].isin(DISTRICTS), "District"].unique())
    if not district_options:
        st.error("No district estimates are available for this selection.")
        st.stop()
    preferred_district = "Nyagatare" if "Nyagatare" in district_options else district_options[0]
    district = st.selectbox("District", district_options, index=district_options.index(preferred_district))
    st.markdown("---")
    st.caption("Official NISR Seasonal Agricultural Survey estimates · 2023–2025")

crop_yield = yields[
    yields["Year"].eq(focus_year)
    & yields["Season"].eq(season)
    & yields["Crop"].eq(crop)
    & yields["District"].isin(DISTRICTS)
    & yields["Value"].gt(0)
].copy().sort_values("Value", ascending=False).reset_index(drop=True)
crop_yield["Rank"] = crop_yield.index + 1
selected_rows = crop_yield[crop_yield["District"].eq(district)]
selected_yield = float(selected_rows.iloc[0]["Value"]) if not selected_rows.empty else None
selected_rank = int(selected_rows.iloc[0]["Rank"]) if not selected_rows.empty else None

district_trend = yields[
    yields["District"].eq(district) & yields["Crop"].eq(crop) & yields["Season"].eq(season) & yields["Value"].gt(0)
].copy()
district_trend["Series"] = district
national_trend = yields[
    yields["District"].str.lower().isin(["national", "national estimate", "national average"])
    & yields["Crop"].eq(crop) & yields["Season"].eq(season) & yields["Value"].gt(0)
].copy()
national_trend["Series"] = "Rwanda"
trend = pd.concat([district_trend, national_trend], ignore_index=True)

baseline = district_trend.loc[district_trend["Year"].eq(2023), "Value"]
change = None
if selected_yield is not None and focus_year != 2023 and not baseline.empty and float(baseline.iloc[0]) != 0:
    change = (selected_yield / float(baseline.iloc[0]) - 1) * 100

national_current = national_trend.loc[national_trend["Year"].eq(focus_year), "Value"]
national_yield = float(national_current.iloc[0]) if not national_current.empty else None
national_gap = ((selected_yield / national_yield - 1) * 100) if selected_yield and national_yield else None
prod = production[
    production["Year"].eq(focus_year) & production["Season"].eq(season)
    & production["Crop"].eq(crop) & production["District"].eq(district) & production["Value"].gt(0)
]
production_value = float(prod.iloc[0]["Value"]) if not prod.empty else None

st.markdown(f"#### {district} · {crop} · Season {season} · {focus_year}")
cols = st.columns(4)
cards = [
    ("Estimated yield", f"{selected_yield:,.0f}" if selected_yield is not None else "—", "kg per hectare · NISR estimate"),
    ("Change since 2023", f"{change:+.1f}%" if change is not None else "—", "same crop, district, and season"),
    ("Vs Rwanda estimate", f"{national_gap:+.1f}%" if national_gap is not None else "—", "district compared with official national yield"),
    ("Estimated production", f"{production_value:,.0f}" if production_value is not None else "—", "metric tonnes · same season"),
]
for col, (label, value, note) in zip(cols, cards):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

tab_overview, tab_map, tab_districts, tab_ai, tab_method = st.tabs(["Three-year view", "District map", "District comparison", "Cassava AI check", "Data & method"])
with tab_overview:
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(f"### {crop} yield trend")
        st.markdown(f'<p class="section-sub">{district} versus the official Rwanda estimate · Season {season} · kg/ha</p>', unsafe_allow_html=True)
        if not trend.empty:
            trend_fig = px.line(trend.sort_values("Year"), x="Year", y="Value", color="Series", markers=True,
                                color_discrete_map={district: "#176b4a", "Rwanda": "#d48a37"})
            trend_fig.update_traces(marker_size=10, line_width=3)
            trend_fig.update_layout(height=390, margin=dict(l=0,r=10,t=12,b=5), paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Survey year", yaxis_title="Yield (kg/ha)",
                                    font=dict(family="DM Sans", color="#52645a"), legend_title="")
            trend_fig.update_xaxes(tickmode="array", tickvals=[2023, 2024, 2025], showgrid=False)
            trend_fig.update_yaxes(showgrid=True, gridcolor="#e7ece5", rangemode="tozero")
            st.plotly_chart(trend_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No comparable yield history is available for this selection.")
    with right:
        st.markdown("### What changed?")
        if selected_yield is None:
            st.markdown('<div class="insight"><strong>No estimate for this year</strong><br>The selected district has no positive estimate in this year and season. Try another year or crop.</div>', unsafe_allow_html=True)
        elif national_gap is None:
            st.markdown('<div class="insight"><strong>District estimate available</strong><br>The national reference is not reported for this crop and season, so a district-to-national comparison is unavailable.</div>', unsafe_allow_html=True)
        elif national_gap >= 0:
            st.markdown(f'<div class="insight"><strong>Above the national estimate</strong><br>{district} is {national_gap:.1f}% above Rwanda’s reported {crop} yield in {focus_year}, Season {season}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight"><strong>Below the national estimate</strong><br>{district} is {abs(national_gap):.1f}% below Rwanda’s reported {crop} yield in {focus_year}, Season {season}. This is a prompt for local follow-up, not an explanation of the gap.</div>', unsafe_allow_html=True)
        st.markdown("#### District position")
        if selected_rank is not None:
            st.metric("Among districts with a positive estimate", f"{selected_rank} of {len(crop_yield)}")
        else:
            st.info("A district ranking is unavailable for this selection.")
        if change is not None:
            st.caption(f"Compared with 2023, the reported estimate changed by {change:+.1f}% for this same season and district.")

with tab_map:
    st.markdown(f"### Rwanda district map · {crop} · Season {season} · {focus_year}")
    map_metric = st.radio("Map indicator", ["Yield (kg/ha)", "Production (MT)"], horizontal=True)
    map_data = yields if map_metric == "Yield (kg/ha)" else production
    map_rows = map_data[
        map_data["Year"].eq(focus_year) & map_data["Season"].eq(season)
        & map_data["Crop"].eq(crop) & map_data["District"].isin(DISTRICTS)
        & map_data["Value"].gt(0)
    ].copy()
    if DISTRICT_GEOJSON.exists():
        try:
            boundaries = load_district_boundaries(str(DISTRICT_GEOJSON), DISTRICT_GEOJSON.stat().st_mtime)
            map_fig = px.choropleth_map(
                map_rows, geojson=boundaries, locations="District", featureidkey="properties.district",
                color="Value", hover_name="District", hover_data={"Value": ":,.0f"},
                color_continuous_scale=[[0, "#e7f1dc"], [0.5, "#77a33a"], [1, "#176b4a"]],
                labels={"Value": map_metric}, map_style="white-bg", center={"lat": -1.94, "lon": 29.87}, zoom=8.0,
            )
            map_fig.update_traces(marker_line_color="#ffffff", marker_line_width=1.5, colorbar_title=map_metric,
                                  colorbar=dict(tickfont=dict(color=theme["ink"]), title_font=dict(color=theme["ink"])))
            map_fig.update_layout(height=590, margin=dict(l=0, r=0, t=5, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="DM Sans", color=theme["ink"]))
            st.plotly_chart(map_fig, width="stretch", config={"displayModeBar": False})
            st.caption("Hover over a district to see its published estimate. Districts without a positive estimate for this selection are not colored.")
            st.markdown(f'<div class="source-note">Administrative boundaries: <a href="{BOUNDARY_SOURCE}" target="_blank">NISR district boundary layer</a>. Estimates: NISR Seasonal Agricultural Survey.</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Could not load the Rwanda district boundaries: {exc}")
    else:
        st.error("The Rwanda district boundary file is missing from assets/rwanda_districts.geojson.")

with tab_districts:
    st.markdown(f"### District comparison · {crop} · Season {season} · {focus_year}")
    st.markdown("Select a district in the sidebar to highlight it. Only districts with a positive published estimate are ranked.")
    if not crop_yield.empty:
        chart = crop_yield.sort_values("Value", ascending=True).copy()
        chart["Focus"] = chart["District"].eq(district)
        district_fig = px.bar(chart, x="Value", y="District", orientation="h", color="Focus",
                              color_discrete_map={True: "#176b4a", False: "#b9c9b6"},
                              hover_data={"Value": ":,.0f", "Focus": False})
        district_fig.update_layout(showlegend=False, height=720, margin=dict(l=0,r=15,t=5,b=5),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis_title="Yield (kg/ha)", yaxis_title="", font=dict(family="DM Sans", color="#52645a"))
        district_fig.update_xaxes(showgrid=True, gridcolor="#e7ece5", zeroline=False)
        district_fig.update_yaxes(showgrid=False, automargin=True)
        st.plotly_chart(district_fig, width="stretch", config={"displayModeBar": False})
        ranking = crop_yield[["Rank", "District", "Value"]].rename(columns={"Value": "Yield (kg/ha)"})
        st.dataframe(ranking, hide_index=True, width="stretch", height=340)
    else:
        st.info("There are no positive district estimates to rank for this selection.")

with tab_ai:
    st.markdown("### Cassava leaf health check")
    st.markdown("Upload a clear, close photo of one cassava leaf. The model screens for four common disease categories or a healthy leaf.")
    st.info("This prototype was trained on cassava photos from Uganda. Results are a screening aid, not a confirmed diagnosis. Ask an extension worker to verify suspected disease.")
    metrics_file = MODEL_FILE.with_suffix(".metrics.json")
    if metrics_file.exists():
        saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        validation_accuracy = saved_metrics.get("validation_accuracy")
        if validation_accuracy is not None:
            st.warning(f"Prototype performance: {validation_accuracy:.1%} accuracy on {saved_metrics.get('validation_count', 0):,} held-out Kaggle images. This baseline can miss or misclassify disease; do not use it alone to make treatment decisions.")
    if not MODEL_FILE.exists():
        st.warning("The trained model file is not installed yet. Run the training command in the README after extracting the Kaggle ZIP.")
    uploaded_image = st.file_uploader("Upload a cassava leaf photo", type=["jpg", "jpeg", "png", "webp"], key="cassava_leaf_upload")
    if uploaded_image is not None:
        try:
            source_image = Image.open(uploaded_image)
            if min(source_image.size) < 96:
                st.warning("This image is too small to check. Please upload a sharper, higher-resolution cassava leaf photo.")
            elif uploaded_image.size > 15 * 1024 * 1024:
                st.warning("This image is larger than 15 MB. Please resize it and try again.")
            elif not MODEL_FILE.exists():
                st.caption("The uploaded image can be previewed, but the trained model is not available yet.")
                st.image(source_image, caption="Uploaded image", width="stretch")
            else:
                st.image(source_image, caption="Uploaded image", width="stretch")
                with st.spinner("Checking whether this looks like a cassava leaf…"):
                    model, embedding_bank, ood_threshold, model_metrics = load_cassava_bundle(
                        str(MODEL_FILE), MODEL_FILE.stat().st_mtime
                    )
                    tensor = prepare_image(source_image)
                    with torch.inference_mode():
                        embedding = extract_embedding(model, tensor)
                        similarity = float((F.normalize(embedding, dim=1) @ embedding_bank.T).max().item())
                        probabilities = torch.softmax(model(tensor), dim=1)[0]
                        confidence, prediction = probabilities.max(dim=0)
                    confidence_value = float(confidence.item())
                    predicted_name = CLASS_NAMES[int(prediction.item())]
                    if similarity < ood_threshold or confidence_value < 0.50:
                        st.warning("I can’t confidently recognize a cassava leaf in this image. It may not be cassava, or it may be too blurry or far away. Please try again with a clear, close photo of one cassava leaf.")
                    elif predicted_name == CLASS_NAMES[4]:
                        st.success(f"The image looks like a healthy cassava leaf ({confidence_value:.0%} model confidence).")
                    else:
                        st.error(f"Possible {predicted_name} detected ({confidence_value:.0%} model confidence). Please ask a local agriculture extension worker to verify it.")
                    with st.expander("How this result was produced"):
                        st.write(f"Cassava-image similarity check: {similarity:.3f} (model cutoff: {ood_threshold:.3f}).")
                        heldout_accuracy = model_metrics.get("validation_accuracy")
                        if heldout_accuracy is not None:
                            st.write(f"Held-out accuracy: {heldout_accuracy:.1%} on {model_metrics.get('validation_count', 0):,} images.")
                        class_scores = model_metrics.get("per_class_accuracy", {})
                        if class_scores:
                            scores = pd.DataFrame([{"Class": label, "Validation accuracy": f"{score:.1%}"} for label, score in class_scores.items()])
                            st.dataframe(scores, hide_index=True, width="stretch")
                        st.caption("The similarity check can reject many unrelated images, but it cannot guarantee recognition of every non-cassava plant. Model confidence is not a disease probability.")
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be read as an image. Please upload a JPG, PNG, or WebP photo of a cassava leaf.")

with tab_method:
    st.markdown("### Data and method")
    st.markdown("The three-year comparison uses NISR Seasonal Agricultural Survey annual tables. It compares the same crop, season, and district across years when estimates are available.")
    st.markdown("**Yield** is shown in kilograms per hectare; **production** is shown in metric tonnes. The dashboard uses the NISR-published `National` row when available for the national reference line.")
    st.markdown(f"**District map:** official NISR administrative district boundaries from the [NISR GIS boundary layer]({BOUNDARY_SOURCE}); map colors represent the selected NISR survey estimate.")
    st.markdown("**Cassava AI check:** the classifier is trained from the [Kaggle Cassava Leaf Disease Classification dataset](https://www.kaggle.com/competitions/cassava-leaf-disease-classification), provided by Makerere University AI Lab. The photos were collected in Uganda; performance in Rwanda needs local validation.")
    st.markdown("Rows reported as a dash or zero are not ranked. The survey tables are descriptive estimates: a change or district gap does not by itself identify its cause, guarantee a trend, or establish an intervention effect.")
    for year in [2023, 2024, 2025]:
        st.markdown(f"- [{year} NISR Seasonal Agricultural Survey tables]({SOURCE_URLS[year]})")

st.markdown("### From insight to action")
st.markdown("Use a persistent yield gap to prioritize local verification with farmers and extension staff. Check planting dates, access to inputs, soil and water conditions, and post-harvest constraints before choosing an intervention.")
st.markdown('<div class="source-note">All figures come from NISR Seasonal Agricultural Survey annual tables for 2023, 2024, and 2025. Cross-year comparisons are descriptive; consult the linked reports for survey design, definitions, and uncertainty details.</div>', unsafe_allow_html=True)

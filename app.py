from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from cassava_detector import CLASS_NAMES, MODEL_FILE, extract_embedding, load_model_bundle, prepare_image
from i18n import PAGES, class_label, crop_label, tr


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
        path = Path(root) / "data" / filename
        with pd.ExcelFile(path, engine="openpyxl") as workbook:
            for season, sheet in YEAR_TABLES[year]["yield"].items():
                yields.append(_read_wide_table(workbook, sheet, year, season, "Yield (kg/ha)"))
            for season, sheet in YEAR_TABLES[year]["production"].items():
                production.append(_read_wide_table(workbook, sheet, year, season, "Production (MT)"))

    return pd.concat(yields, ignore_index=True), pd.concat(production, ignore_index=True)


def _read_wide_table(path: str | pd.ExcelFile, sheet: str, year: int, season: str, metric: str) -> pd.DataFrame:
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
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=132)
    st.markdown("### Harvest Compass")
    language_choice = st.selectbox(
        "Language / Ururimi", ["English", "Kinyarwanda"],
        format_func=lambda value: "ENG · English" if value == "English" else "KINY · Kinyarwanda",
        label_visibility="visible",
    )
    language = "rw" if language_choice == "Kinyarwanda" else "en"
    display_mode = st.radio(
        tr("Display mode", language), ["Light", "Dark"], horizontal=True,
        format_func=lambda value: tr(value, language),
    )
    active_page = st.radio(
        tr("Navigation", language), PAGES,
        format_func=lambda value: tr(value, language),
        label_visibility="visible", key="dashboard_page",
    )
    st.markdown("---")
    st.markdown(f"#### {tr('Filters', language)}")

if display_mode == "Dark":
    theme = {"paper":"#111a1c", "sidebar":"#162326", "ink":"#edf4ef", "muted":"#b8c8c0", "border":"#30413e", "hero":"#193b32", "card":"#1b2b2d", "insight":"#243a30"}
else:
    theme = {"paper":"#f5f7f1", "sidebar":"#eef2e8", "ink":"#162d26", "muted":"#64756b", "border":"#dfe5da", "hero":"#163c2f", "card":"#ffffff", "insight":"#e7efdb"}

style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:__INK__; --muted:__MUTED__; --paper:__PAPER__; --text-color:__INK__; --background-color:__PAPER__; --secondary-background-color:__SIDEBAR__; --primary-color:#287d55; color-scheme:__MODE__; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:__INK__; }
    .stApp, [data-testid="stAppViewContainer"] { background:__PAPER__; color:__INK__; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:__SIDEBAR__; border-right:1px solid __BORDER__; }
    .block-container { padding-top:2rem; max-width:1440px; }
    .topbar { background:#1749a3; color:#fff; border-radius:0 0 18px 18px; padding:.85rem 1.4rem; margin:-2rem -1rem 1.2rem; box-shadow:0 5px 18px rgba(18,44,85,.18); }
    .topbar-title { color:#fff; font:700 1rem 'Manrope',sans-serif; }
    .topbar-caption { color:#e3eaff; font-size:.76rem; margin-top:2px; }
    h1,h2,h3,h4,h5,h6 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:__INK__; }
    [data-testid="stSidebar"] img { border-radius:12px; }
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
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.3rem .45rem; border-radius:8px; }
    @media(max-width:700px) { .topbar { margin:-1.5rem -1rem 1rem; } .hero { padding:1.25rem; } .hero h1 {font-size:1.75rem;} }
    </style>
"""
for token, value in {"INK": theme["ink"], "MUTED": theme["muted"], "PAPER": theme["paper"], "SIDEBAR": theme["sidebar"], "BORDER": theme["border"], "HERO": theme["hero"], "CARD": theme["card"], "INSIGHT": theme["insight"], "MODE": display_mode.lower()}.items():
    style = style.replace(f"__{token}__", value)
st.markdown(style, unsafe_allow_html=True)

st.markdown(
    f'<div class="topbar"><div class="topbar-title">{tr("Harvest Compass · Rwanda Agriculture", language)}</div><div class="topbar-caption">{tr("National Institute of Statistics of Rwanda · SAS 2023–2025", language)}</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">{tr('Agriculture data for district-level decisions', language)}</div>
      <h1>Harvest Compass</h1>
      <p>{tr('Track crop yields across seasons and years. Compare your district with Rwanda’s reported estimates and spot patterns worth investigating.', language)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing_files = [name for name in YEAR_FILES.values() if not (ROOT / "data" / name).exists()]
if missing_files:
    st.error(f"{tr('Missing source workbook(s):', language)} " + ", ".join(missing_files))
    st.stop()

try:
    cache_key = tuple((ROOT / "data" / name).stat().st_mtime for name in YEAR_FILES.values())
    yields, production = load_sas(str(ROOT), cache_key)
except Exception as exc:
    st.error(tr("Could not read the NISR tables: {error}", language, error=exc))
    st.stop()

with st.sidebar:
    st.markdown(f"### {tr('Explore the harvest', language)}")
    st.caption(tr("Compare the same crop across districts, seasons, and survey years.", language))
    season = st.selectbox(tr("Growing season", language), ["A", "B", "C"], format_func=lambda x: tr("Season {season}", language, season=x))
    focus_year = st.selectbox(tr("District snapshot", language), [2025, 2024, 2023])
    season_yields = yields[yields["Season"].eq(season)]
    crops = sorted(season_yields.loc[season_yields["District"].isin(DISTRICTS) & season_yields["Value"].gt(0), "Crop"].dropna().unique())
    if not crops:
        st.error(tr("No crop yield records were found for Season {season}.", language, season=season))
        st.stop()
    preferred = next((crop for crop in crops if crop.lower() == "maize"), crops[0])
    crop = st.selectbox(tr("Crop", language), crops, index=crops.index(preferred), format_func=lambda value: crop_label(value, language))
    focus_rows = season_yields[(season_yields["Year"].eq(focus_year)) & season_yields["Crop"].eq(crop)]
    district_options = sorted(focus_rows.loc[focus_rows["District"].isin(DISTRICTS), "District"].unique())
    if not district_options:
        district_options = sorted(season_yields.loc[season_yields["Crop"].eq(crop) & season_yields["District"].isin(DISTRICTS), "District"].unique())
    if not district_options:
        st.error(tr("No district estimates are available for this selection.", language))
        st.stop()
    preferred_district = "Nyagatare" if "Nyagatare" in district_options else district_options[0]
    district = st.selectbox(tr("District", language), district_options, index=district_options.index(preferred_district))
    st.markdown("---")
    st.caption(tr("Official NISR Seasonal Agricultural Survey estimates · 2023–2025", language))

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
national_trend["Series"] = tr("Rwanda", language)
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

crop_text = crop_label(crop, language)
st.markdown(f"#### {tr('{district} · {crop} · Season {season} · {year}', language, district=district, crop=crop_text, season=season, year=focus_year)}")
cols = st.columns(4)
cards = [
    (tr("Estimated yield", language), f"{selected_yield:,.0f}" if selected_yield is not None else "—", tr("kg per hectare · NISR estimate", language)),
    (tr("Change since 2023", language), f"{change:+.1f}%" if change is not None else "—", tr("same crop, district, and season", language)),
    (tr("Vs Rwanda estimate", language), f"{national_gap:+.1f}%" if national_gap is not None else "—", tr("district compared with official national yield", language)),
    (tr("Estimated production", language), f"{production_value:,.0f}" if production_value is not None else "—", tr("metric tonnes · same season", language)),
]
for col, (label, value, note) in zip(cols, cards):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

if active_page == "Three-year view":
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(f"### {crop_text} {tr('yield trend', language)}")
        st.markdown(f'<p class="section-sub">{tr("{district} versus the official Rwanda estimate · Season {season} · kg/ha", language, district=district, season=season)}</p>', unsafe_allow_html=True)
        if not trend.empty:
            trend_fig = px.line(trend.sort_values("Year"), x="Year", y="Value", color="Series", markers=True,
                                labels={"Year": tr("Survey year", language), "Value": tr("Yield (kg/ha)", language), "Series": tr("Series", language)},
                                color_discrete_map={district: "#176b4a", tr("Rwanda", language): "#d48a37"})
            trend_fig.update_traces(marker_size=10, line_width=3)
            trend_fig.update_layout(height=390, margin=dict(l=0,r=10,t=12,b=5), paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", xaxis_title=tr("Survey year", language), yaxis_title=tr("Yield (kg/ha)", language),
                                    font=dict(family="DM Sans", color=theme["ink"]), legend_title="")
            trend_fig.update_xaxes(tickmode="array", tickvals=[2023, 2024, 2025], showgrid=False)
            trend_fig.update_yaxes(showgrid=True, gridcolor="#e7ece5", rangemode="tozero")
            st.plotly_chart(trend_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info(tr("No comparable yield history is available for this selection.", language))
    with right:
        st.markdown(f"### {tr('What changed?', language)}")
        if selected_yield is None:
            st.markdown(f'<div class="insight"><strong>{tr("No estimate for this year", language)}</strong><br>{tr("The selected district has no positive estimate in this year and season. Try another year or crop.", language)}</div>', unsafe_allow_html=True)
        elif national_gap is None:
            st.markdown(f'<div class="insight"><strong>{tr("District estimate available", language)}</strong><br>{tr("The national reference is not reported for this crop and season, so a district-to-national comparison is unavailable.", language)}</div>', unsafe_allow_html=True)
        elif national_gap >= 0:
            st.markdown(f'<div class="insight"><strong>{tr("Above the national estimate", language)}</strong><br>{tr("{district} is {gap:.1f}% above Rwanda’s reported {crop} yield in {year}, Season {season}.", language, district=district, gap=national_gap, crop=crop_text, year=focus_year, season=season)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight"><strong>{tr("Below the national estimate", language)}</strong><br>{tr("{district} is {gap:.1f}% below Rwanda’s reported {crop} yield in {year}, Season {season}. This is a prompt for local follow-up, not an explanation of the gap.", language, district=district, gap=abs(national_gap), crop=crop_text, year=focus_year, season=season)}</div>', unsafe_allow_html=True)
        st.markdown(f"#### {tr('District position', language)}")
        if selected_rank is not None:
            st.metric(tr("Among districts with a positive estimate", language), tr("{rank} of {total}", language, rank=selected_rank, total=len(crop_yield)))
        else:
            st.info(tr("A district ranking is unavailable for this selection.", language))
        if change is not None:
            st.caption(tr("Compared with 2023, the reported estimate changed by {change:+.1f}% for this same season and district.", language, change=change))

if active_page == "District map":
    st.markdown(f"### {tr('Rwanda district map · {crop} · Season {season} · {year}', language, crop=crop_text, season=season, year=focus_year)}")
    map_options = ["Yield (kg/ha)", "Production (MT)"]
    map_metric = st.radio(tr("Map indicator", language), map_options, horizontal=True, format_func=lambda value: tr(value, language))
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
                color="Value", hover_name="District", hover_data={"District": False, "Value": ":,.0f"},
                color_continuous_scale=[[0, "#e7f1dc"], [0.5, "#77a33a"], [1, "#176b4a"]],
                labels={"Value": tr(map_metric, language)}, map_style="white-bg", center={"lat": -1.94, "lon": 29.87}, zoom=8.0,
            )
            map_fig.update_traces(marker_line_color="#ffffff", marker_line_width=1.5, colorbar_title=tr(map_metric, language),
                                  colorbar=dict(tickfont=dict(color=theme["ink"]), title_font=dict(color=theme["ink"])))
            map_fig.update_layout(height=590, margin=dict(l=0, r=0, t=5, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="DM Sans", color=theme["ink"]))
            st.plotly_chart(map_fig, width="stretch", config={"displayModeBar": False})
            st.caption(tr("Hover over a district to see its published estimate. Districts without a positive estimate for this selection are not colored.", language))
            st.markdown(f'<div class="source-note">{tr("Administrative boundaries", language)}: <a href="{BOUNDARY_SOURCE}" target="_blank">{tr("NISR district boundary layer", language)}</a>. {tr("Estimates: NISR Seasonal Agricultural Survey.", language)}</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(tr("Could not load the Rwanda district boundaries: {error}", language, error=exc))
    else:
        st.error(tr("The Rwanda district boundary file is missing from assets/rwanda_districts.geojson.", language))

if active_page == "District comparison":
    st.markdown(f"### {tr('District comparison · {crop} · Season {season} · {year}', language, crop=crop_text, season=season, year=focus_year)}")
    st.markdown(f"{tr('Select a district in the sidebar to highlight it.', language)} {tr('Only districts with a positive published estimate are ranked.', language)}")
    if not crop_yield.empty:
        chart = crop_yield.sort_values("Value", ascending=True).copy()
        chart["Focus"] = chart["District"].eq(district)
        district_fig = px.bar(chart, x="Value", y="District", orientation="h", color="Focus",
                              color_discrete_map={True: "#176b4a", False: "#b9c9b6"},
                              hover_data={"Value": ":,.0f", "Focus": False},
                              labels={"Value": tr("Yield (kg/ha)", language), "District": tr("District", language)})
        district_fig.update_layout(showlegend=False, height=720, margin=dict(l=0,r=15,t=5,b=5),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    xaxis_title=tr("Yield (kg/ha)", language), yaxis_title="", font=dict(family="DM Sans", color=theme["ink"]))
        district_fig.update_xaxes(showgrid=True, gridcolor="#e7ece5", zeroline=False)
        district_fig.update_yaxes(showgrid=False, automargin=True)
        st.plotly_chart(district_fig, width="stretch", config={"displayModeBar": False})
        ranking = crop_yield[["Rank", "District", "Value"]].rename(columns={"Rank": tr("Rank", language), "District": tr("District", language), "Value": tr("Yield (kg/ha)", language)})
        st.dataframe(ranking, hide_index=True, width="stretch", height=340)
    else:
        st.info(tr("No positive district estimates to rank for this selection.", language))

if active_page == "Cassava AI check":
    st.markdown(f"### {tr('Cassava leaf health check', language)}")
    st.markdown(tr("Upload a clear, close photo of one cassava leaf. The model screens for four common disease categories or a healthy leaf.", language))
    st.info(tr("This prototype was trained on cassava photos from Uganda. Results are a screening aid, not a confirmed diagnosis. Ask an extension worker to verify suspected disease.", language))
    metrics_file = MODEL_FILE.with_suffix(".metrics.json")
    if metrics_file.exists():
        saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        validation_accuracy = saved_metrics.get("validation_accuracy")
        if validation_accuracy is not None:
            st.warning(tr("Prototype performance: {accuracy:.1%} accuracy on {count:,} held-out Kaggle images. This baseline can miss or misclassify disease; do not use it alone to make treatment decisions.", language, accuracy=validation_accuracy, count=saved_metrics.get('validation_count', 0)))
    if not MODEL_FILE.exists():
        st.warning(tr("The trained model file is not installed yet. Run the training command in the README after extracting the Kaggle ZIP.", language))
    uploaded_image = st.file_uploader(tr("Upload a cassava leaf photo", language), type=["jpg", "jpeg", "png", "webp"], key="cassava_leaf_upload")
    if uploaded_image is not None:
        try:
            source_image = Image.open(uploaded_image)
            if min(source_image.size) < 96:
                st.warning(tr("This image is too small to check. Please upload a sharper, higher-resolution cassava leaf photo.", language))
            elif uploaded_image.size > 15 * 1024 * 1024:
                st.warning(tr("This image is larger than 15 MB. Please resize it and try again.", language))
            elif not MODEL_FILE.exists():
                st.caption(tr("The uploaded image can be previewed, but the trained model is not available yet.", language))
                st.image(source_image, caption=tr("Uploaded image", language), width="stretch")
            else:
                st.image(source_image, caption=tr("Uploaded image", language), width="stretch")
                with st.spinner(tr("Checking whether this looks like a cassava leaf…", language)):
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
                        st.warning(tr("I can’t confidently recognize a cassava leaf in this image. It may not be cassava, or it may be too blurry or far away. Please try again with a clear, close photo of one cassava leaf.", language))
                    elif predicted_name == CLASS_NAMES[4]:
                        st.success(tr("The image looks like a healthy cassava leaf ({confidence:.0%} model confidence).", language, confidence=confidence_value))
                    else:
                        st.error(tr("Possible {disease} detected ({confidence:.0%} model confidence). Please ask a local agriculture extension worker to verify it.", language, disease=class_label(predicted_name, language), confidence=confidence_value))
                    with st.expander(tr("How this result was produced", language)):
                        st.write(tr("Cassava-image similarity check: {similarity:.3f} (model cutoff: {threshold:.3f}).", language, similarity=similarity, threshold=ood_threshold))
                        heldout_accuracy = model_metrics.get("validation_accuracy")
                        if heldout_accuracy is not None:
                            st.write(tr("Held-out accuracy: {accuracy:.1%} on {count:,} images.", language, accuracy=heldout_accuracy, count=model_metrics.get('validation_count', 0)))
                        class_scores = model_metrics.get("per_class_accuracy", {})
                        if class_scores:
                            scores = pd.DataFrame([{tr("Class", language): class_label(label, language), tr("Validation accuracy", language): f"{score:.1%}"} for label, score in class_scores.items()])
                            st.dataframe(scores, hide_index=True, width="stretch")
                        st.caption(tr("The similarity check can reject many unrelated images, but it cannot guarantee recognition of every non-cassava plant. Model confidence is not a disease probability.", language))
        except (UnidentifiedImageError, OSError):
            st.error(tr("This file could not be read as an image. Please upload a JPG, PNG, or WebP photo of a cassava leaf.", language))

if active_page == "Data & method":
    st.markdown(f"### {tr('Data and method', language)}")
    st.markdown(tr("The three-year comparison uses NISR Seasonal Agricultural Survey annual tables. It compares the same crop, season, and district across years when estimates are available.", language))
    st.markdown(tr("Yield is shown in kilograms per hectare; production is shown in metric tonnes. The dashboard uses the NISR-published National row when available for the national reference line.", language))
    st.markdown(f"**{tr('District map', language)}:** {tr('Official NISR administrative district boundaries; map colors represent the selected NISR survey estimate.', language)}")
    st.markdown(tr("Cassava AI check: the classifier is trained from the Kaggle Cassava Leaf Disease Classification dataset, provided by Makerere University AI Lab. The photos were collected in Uganda; performance in Rwanda needs local validation.", language))
    st.markdown(tr("Rows reported as a dash or zero are not ranked. The survey tables are descriptive estimates: a change or district gap does not by itself identify its cause, guarantee a trend, or establish an intervention effect.", language))
    for year in [2023, 2024, 2025]:
        st.markdown(f"- [{tr('{year} NISR Seasonal Agricultural Survey tables', language, year=year)}]({SOURCE_URLS[year]})")

    st.markdown(f"### {tr('From insight to action', language)}")
    st.markdown(tr("Use a persistent yield gap to prioritize local verification with farmers and extension staff. Check planting dates, access to inputs, soil and water conditions, and post-harvest constraints before choosing an intervention.", language))
    st.markdown(f'<div class="source-note">{tr("All figures come from NISR Seasonal Agricultural Survey annual tables for 2023, 2024, and 2025. Cross-year comparisons are descriptive; consult the linked reports for survey design, definitions, and uncertainty details.", language)}</div>', unsafe_allow_html=True)

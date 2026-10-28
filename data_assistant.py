"""Small, deterministic question answering over the dashboard's NISR tables."""

from __future__ import annotations

import re
import pandas as pd

SOURCE_URLS = {
    2023: "https://alpha.statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2023",
    2024: "https://www.statistics.gov.rw/statistical-publications/agriculture-environment/seasonal-agricultural-survey-2024-annual-report-2024",
    2025: "https://statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2025-",
}
NATIONAL_ROWS = {"National", "National estimate", "National average"}

ALIASES = {
    "Maize": ("ibigori", "corn"), "Bean": ("ibishyimbo", "beans"),
    "Cassava": ("imyumbati",), "Irish potato": ("ibirayi", "irish potatoes"),
    "Sweet potato": ("ibijumba", "sweet potatoes"), "Paddy rice": ("umuceri", "paddy rice"),
    "Rice": ("umuceri",), "Groundnut": ("ubunyobwa", "groundnuts", "peanuts"),
    "Banana": ("ibitoki",), "Sorghum": ("amasaka",),
}
ALIASES_RW = {crop: aliases[0] for crop, aliases in ALIASES.items()}

MESSAGES = {
    "welcome": {
        "en": "Ask about a crop, district, season, or year in the NISR 2023–2025 tables. For example: “Which district had the highest maize yield in Season A 2025?”",
        "rw": "Baza ku gihingwa, akarere, igihe cyangwa umwaka biri mu mbonerahamwe za NISR za 2023–2025. Urugero: “Ni akahe karere gafite umusaruro mwinshi w'ibigori mu gihembwe A cya 2025?”",
    },
    "unsupported": {
        "en": "I can answer questions about the NISR 2023–2025 crop yield and production tables. I can’t verify causes, forecasts, weather, prices, or treatment advice from these data. Try asking for a district ranking or a selected district’s reported estimate.",
        "rw": "Nshobora gusubiza ibibazo ku mibare ya NISR y'umusaruro kuri hegitari n'uwose kuva mu 2023 kugeza mu 2025. Aya makuru ntasobanura impamvu, iteganyagihe, ibiciro cyangwa inama zo kuvura. Baza ku rutonde rw'uturere cyangwa imibare yatangajwe ku karere wahisemo.",
    },
    "not_found": {
        "en": "I couldn’t find a positive reported estimate for that selection in the loaded NISR tables. Try another year, season, or crop. The dashboard uses the sidebar selection for details you didn’t specify.",
        "rw": "Nta mubare urenze zeru nabonye muri ayo mahitamo mu mbonerahamwe za NISR ziri muri porogaramu. Gerageza undi mwaka, igihe cyangwa igihingwa. Iyo utavuze ibipimo byose, nkoresha ibyatoranyijwe ku ruhande rw'ibumoso.",
    },
    "data_scope": {
        "en": "Harvest Compass compares NISR Seasonal Agricultural Survey crop estimates across districts, seasons, and 2023–2025. Yield is measured in kg/ha and production in metric tonnes. These are descriptive survey estimates, not explanations of why results changed.",
        "rw": "Harvest Compass igereranya imibare y'ubushakashatsi bwa NISR ku bihingwa mu turere, bihe n'imyaka ya 2023–2025. Umusaruro kuri hegitari upimwa muri kg/ha, naho umusaruro wose ugapimwa muri toni metiriki. Iyi ni imibare y'ubushakashatsi; ntisobanura impamvu imibare yahindutse.",
    },
}


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _detect(query: str, values: list[str], aliases: dict[str, tuple[str, ...]] | None = None) -> str | None:
    normalized = query.casefold()
    candidates: list[tuple[int, str]] = []
    for value in values:
        if _contains(normalized, value.casefold()):
            candidates.append((len(value), value))
        for alias in (aliases or {}).get(value, ()):
            if _contains(normalized, alias.casefold()):
                candidates.append((len(alias), value))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _source(year: int) -> str:
    return f"[NISR SAS {year}]({SOURCE_URLS[year]})"


def answer_question(
    question: str,
    yields: pd.DataFrame,
    production: pd.DataFrame,
    *,
    default_crop: str,
    default_district: str,
    default_season: str,
    default_year: int,
    language: str = "en",
) -> str:
    """Answer a narrow set of questions using only loaded survey rows."""
    q = " ".join(question.casefold().split())
    lang = "rw" if language == "rw" else "en"
    if not q:
        return MESSAGES["welcome"][lang]

    if any(word in q for word in ("what is harvest compass", "what does this app", "what can you do", "iyi porogaramu ikora iki")):
        return MESSAGES["data_scope"][lang]

    advice_terms = (
        "why", "cause", "reason", "predict", "forecast", "weather", "price", "fertilizer",
        "fertiliser", "pesticide", "spray", "treat", "treatment", "medicine", "recommend",
        "inama", "impamvu", "iteganyagihe", "igiciro", "umuti", "kuvura",
    )
    if any(term in q for term in advice_terms):
        return MESSAGES["unsupported"][lang]

    all_crops = sorted(set(yields["Crop"].dropna().astype(str)) | set(production["Crop"].dropna().astype(str)), key=len, reverse=True)
    crop = _detect(q, all_crops, ALIASES) or default_crop
    district_values = sorted(
        (set(yields["District"].dropna().astype(str)) & set(production["District"].dropna().astype(str))) - NATIONAL_ROWS,
        key=len,
        reverse=True,
    )
    district = _detect(q, district_values) or default_district
    season_match = re.search(r"\bseason\s*([abc])\b|\bigihe(?:mbwe)?(?:\s+cya)?\s*([abc])\b", q)
    season = (season_match.group(1) or season_match.group(2)).upper() if season_match else default_season
    year_match = re.search(r"\b(20\d{2})\b", q)
    if year_match and year_match.group(1) not in {"2023", "2024", "2025"}:
        return MESSAGES["not_found"][lang]
    year = int(year_match.group(1)) if year_match else default_year

    is_production = any(term in q for term in ("production", "tonne", "tonnes", "metric ton", "umusaruro wose", "toni"))
    frame = production if is_production else yields
    metric_label = "metric tonnes" if is_production else "kg/ha"
    higher = any(term in q for term in ("highest", "top", "most", "best", "mwinshi", "hejuru"))
    lower = any(term in q for term in ("lowest", "bottom", "least", "fewest", "muke", "hasi"))
    compare_districts = any(term in q for term in ("compare districts", "district comparison", "gereranya uturere"))
    wants_rank = higher or lower or compare_districts or any(term in q for term in ("rank", "ranking", "urutonde", "position"))
    wants_trend = any(term in q for term in ("trend", "over time", "across years", "each year", "every year", "compare years", "uko byagiye", "mu myaka"))

    rows = frame[
        frame["Crop"].eq(crop)
        & frame["Season"].eq(season)
        & frame["Value"].gt(0)
    ].copy()

    if wants_rank:
        rows = rows[rows["Year"].eq(year) & rows["District"].isin(district_values)]
        if rows.empty:
            return MESSAGES["not_found"][lang]
        if compare_districts:
            leaders = rows.sort_values("Value", ascending=False).head(3)
            listing = "; ".join(f"{row.District}: {row.Value:,.0f} {metric_label}" for row in leaders.itertuples())
            if lang == "rw":
                return f"Uturere dutatu twa mbere ku musaruro wa {ALIASES_RW.get(crop, crop)} mu gihe {season} cya {year}: {listing}. {_source(year)}"
            return f"Top districts for {crop} {('production' if is_production else 'yield')}, Season {season} {year}: {listing}. {_source(year)}"
        if higher or lower:
            rows = rows.sort_values("Value", ascending=not higher)
            row = rows.iloc[0]
            result_district = row["District"]
            result_value = float(row["Value"])
            if lang == "rw":
                crop_rw = ALIASES_RW.get(crop, crop)
                verb = "umusaruro mwinshi" if higher else "umusaruro muke"
                return f"Mu gihe {season} cya {year}, {result_district} ni ko karere kagize {verb} wa {crop_rw}: **{result_value:,.0f} {metric_label}**. {_source(year)}"
            direction = "highest" if higher else "lowest"
            return f"For {crop}, Season {season} {year}, **{result_district}** had the {direction} reported {('production' if is_production else 'yield')}: **{result_value:,.0f} {metric_label}**. {_source(year)}"
        rows = rows.sort_values("Value", ascending=False).reset_index(drop=True)
        selected = rows[rows["District"].eq(district)]
        if selected.empty:
            return MESSAGES["not_found"][lang]
        rank = int(selected.index[0]) + 1
        result_district = district
        result_value = float(selected.iloc[0]["Value"])
        source = _source(year)
        if lang == "rw":
            crop_rw = ALIASES_RW.get(crop, crop)
            return f"Mu turere {len(rows)} dufite imibare, {result_district} iri ku mwanya wa {rank} ku rutonde rw'umusaruro wa {crop_rw} mu gihe {season} cya {year}: **{result_value:,.0f} {metric_label}**. {source}"
        return f"Among {len(rows)} districts with a positive estimate, **{result_district}** ranks **{rank}** for {crop} {('production' if is_production else 'yield')}, Season {season} {year}: **{result_value:,.0f} {metric_label}**. {source}"

    if wants_trend:
        rows = rows[rows["District"].eq(district) & rows["Year"].between(2023, 2025)].sort_values("Year")
        if rows.empty:
            return MESSAGES["not_found"][lang]
        entries = "; ".join(f"{int(row.Year)}: {row.Value:,.0f} {metric_label}" for row in rows.itertuples())
        sources = ", ".join(_source(int(y)) for y in rows["Year"].tolist())
        if lang == "rw":
            return f"{district}, {ALIASES_RW.get(crop, crop)}, igihe {season}: {entries}. Iyi ni imibare yatangajwe, si ibisobanuro by'impamvu yahindutse. {sources}"
        return f"{district}, {crop}, Season {season}: {entries}. These are reported estimates; they don’t explain why values changed. Sources: {sources}"

    national_terms = ("rwanda", "national", "igihugu", "u rwanda")
    target = next((value for value in NATIONAL_ROWS if value in set(rows["District"])), None) if any(term in q for term in national_terms) else district
    rows = rows[rows["Year"].eq(year) & rows["District"].eq(target)]
    if rows.empty:
        return MESSAGES["not_found"][lang]
    value = float(rows.iloc[0]["Value"])
    source = _source(year)
    if lang == "rw":
        return f"{target}, {ALIASES_RW.get(crop, crop)}, igihe {season} mu {year}: **{value:,.0f} {metric_label}**. {source}"
    measure = "production" if is_production else "yield"
    return f"The reported {measure} for **{crop} in {target}**, Season {season} {year}, is **{value:,.0f} {metric_label}**. {source}"


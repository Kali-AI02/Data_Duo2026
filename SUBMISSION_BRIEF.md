# Harvest Compass — NISR 2026 Hackathon Entry

## Project summary

Harvest Compass is a bilingual English/Kinyarwanda dashboard for comparing crop yield and production across Rwanda's districts, seasons, and years. It turns official Seasonal Agricultural Survey (SAS) tables into a district map, a district ranking, and a three-year trend. It also includes an experimental cassava-leaf image screening feature.

## 1. Problem understanding and national relevance

NISR publishes detailed agricultural survey tables, but it takes effort for a farmer, extension worker, or planner to compare a crop across years, seasons, and districts. Harvest Compass makes those comparisons easier to inspect and highlights differences that local teams may want to verify.

The project aligns with NST2's priority-food-crop productivity indicators, including the index on productivity increase for priority food crops and productivity/production for specific priority food crops. It is intended as an evidence-navigation tool: a district gap is a reason to investigate locally, not proof of its cause or a recommendation to apply a particular intervention.

## 2. Data and methodology

- **NISR SAS 2023, 2024, and 2025 annual tables:** district yield and production by crop and season. The app selects the relevant workbook table for each year/season, parses numeric estimates, canonicalizes crop names, and keeps national rows for national comparisons when published.
- **NISR-copyrighted district boundaries:** a local GeoJSON copy of the NISR district layer served by the NAEB GIS service. District names are joined to the SAS estimates for the choropleth map.
- **Kaggle Cassava Leaf Disease Classification:** labeled cassava images collected in Uganda, used for the experimental five-class image model. Kaggle lists the dataset as subject to its competition rules; check those rules before redistributing data or relying on the model outside permitted use.

Yield is displayed in kg/ha and production in metric tonnes. Comparisons are descriptive and use only positive reported estimates. The cassava model uses MobileNetV3-Small ImageNet visual features and a trained five-class classifier. A stratified 15% holdout (3,210 images) was used for a single validation split.

**Cassava model holdout accuracy: 70.6%.** Per-class accuracy was CBB 39.9%, CBSD 68.6%, CGM 41.6%, CMD 82.4%, and healthy 52.2%. A visual-similarity cutoff flags many unrelated/unclear uploads, but the dataset contains no non-cassava negative class, so it cannot reliably reject every other crop. Results are experimental screening outputs, not diagnoses.

## 3. Technology and innovation

The prototype combines Python, Streamlit, Pandas, Plotly, GeoJSON, and PyTorch. Its main innovation is putting three-year, season-matched official district statistics and a spatial comparison in one interface, alongside a clearly separated cassava image-screening experiment. The UI supports English and Kinyarwanda and adapts to light or dark display mode.

## 4. Usability and design

The sidebar groups language, theme, page navigation, and crop/season/year/district filters. The main pages provide a three-year trend, a choropleth district map, a district comparison, a cassava image uploader, and a data/method page. The map tooltip provides the selected estimate; the AI page shows a caution and validation results.

## 5. Potential tangible impact

Agricultural planners and extension staff can use persistent district gaps to prioritize field follow-up with farmers and investigate planting dates, input access, soil and water conditions, and post-harvest constraints. Farmers can explore official estimates and use the cassava page as a demonstration of a future phone-photo screening workflow.

The impact is potential, not yet measured. Before operational use, the team should pilot the dashboard with Rwandan extension staff, validate the image model on locally collected and permissioned cassava photos, measure class-wise error and non-cassava rejection, and revise or withhold disease outputs that fail those checks.

## Sources

- [NISR 2023 Seasonal Agricultural Survey](https://alpha.statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2023)
- [NISR 2024 Seasonal Agricultural Survey](https://www.statistics.gov.rw/statistical-publications/agriculture-environment/seasonal-agricultural-survey-2024-annual-report-2024)
- [NISR 2025 Seasonal Agricultural Survey](https://statistics.gov.rw/data-sources/surveys/Seasonal-Agricultural-Survey/Seasonal-Agricultural-Survey/seasonal-agricultural-survey-2025-)
- [NISR National Strategy for the Development of Statistics 2024–2029, Annex 21](https://www.statistics.gov.rw/sites/default/files/documents/2025-05/NSDS4%20%282024_2029%29.pdf)
- [NISR district boundary layer](https://gis.naeb.gov.rw/server/rest/services/Hosted/Administrative_Boundaries_WFL1/FeatureServer/0)
- [Makerere University AI Lab / Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

## AI assistance and participant review

AI tools assisted with code drafting, translation scaffolding, and project documentation. The team must review the code, verify every claim and source, and disclose AI assistance as required. The participants remain responsible for the submission and must ensure it is their original work. NISR's entry conditions also state that entering transfers the submitted work's intellectual property rights to NISR; both team members should read and understand those conditions before submitting.

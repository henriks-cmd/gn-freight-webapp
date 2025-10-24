# GN Freight WebApp

**Version:** 2025-10-24.5  
**Stack:** Streamlit (Python), NumPy, Pandas (+ XlsxWriter/openpyxl för Excel)

## Funktioner
- Calculator: per-kg ladder (breaks + min), per-pallet, per-FLM, **Auto-lowest**
- Curve fit: **POWER** och **EXP**, visar R²
- Tillägg: fuel %, MARPOL %, extra %, per LDM, per kg, flat + FR roadfee-hjälp
- **LDM Scaler:** 1–13 LDM från P13 (POWER/EXP eller modell)
- **Weight Scaler:** retiering (INTEGRATED / POINT) + **Anchor** (matcha total vid vikt)
- **Containers:** 1–19 från FTL (=20)
- **Export:** Excel med **vertikal** och **horisontell** layout på alla flikar
- Admin: lanes, settings, import/export (JSON)

## Snabbstart (lokalt)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> Excel-export: `pip install XlsxWriter openpyxl` (finns i requirements.txt).

## Deployment (Streamlit Community Cloud)
- Repo: `henriks-cmd/gn-freight-webapp`
- Branch: `main`
- File path: `app.py`

## Struktur
```
Webapp/
├─ app.py
├─ requirements.txt
├─ README.md
└─ docs/
   ├─ ROADMAP.md
   └─ CHANGELOG.md
```

# GN Freight WebApp

**Version:** 2025-10-24.5  
**Stack:** Streamlit (Python), NumPy, Pandas (+ xlsxwriter/openpyxl för Excel)

## Funktioner
- Calculator: per-kg ladder (breaks + min), per-pallet, per-FLM, **Auto-lowest**
- Curve fit: **POWER** och **EXP**, visar R²
- Tillägg: fuel %, MARPOL %, extra %, per LDM, per kg, flat + FR roadfee-hjälp
- **LDM Scaler:** genererar 1–13 LDM från P13 via POWER/EXP eller modell
- **Weight Scaler:** retiering till nya vikter (INTEGRATED / POINT) + **Anchor** (matcha total vid vikt)
- **Containers:** 1–19 skalas från FTL (=20)
- **Export:** Excel med både **vertikal** och **horisontell** layout på alla flikar
- Admin: lanes, settings, import/export (JSON)

## Snabbstart (lokalt)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> Om Excel-export klagar:  
> `pip install xlsxwriter openpyxl` och lägg till dessa i `requirements.txt`.

## Deployment (Streamlit Community Cloud)
- Repo: `henriks-cmd/gn-freight-webapp`
- Branch: `main`
- File path: `app.py`
- Om appen inte uppdateras: hård-reload (Cmd+Shift+R) eller “Rerun” / “Clear cache”.

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

## Tips
- Använd unika `key=` för Streamlit-widgets (undviker DuplicateElementId)
- Säkerhetskopiera tariffer via “Download/Upload lanes.json” i sidopanelen

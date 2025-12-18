# GN Freight WebApp – Streamlit (single-file)
# -------------------------------------------------------------
# Run locally:
#   pip install streamlit numpy pandas XlsxWriter openpyxl
#   streamlit run app.py
#
# Features (kort):
# - Calculator (per-kg/per-pallet/per-FLM + surcharges + FR roadfee helper)
# - Fit POWER/EXP från kg-ladder (R²)
# - LDM Scaler (från P13 eller fit)
# - Weight Scaler (retier till nya brytvikter via INTEGRATED/POINT + anchor)
# - Weight from FTL (skala vikter från exakt FTL-pris) — med manuella intervall/ranges
# - Containers (1–19 från FTL=20)
# - Pallet Scaler (modell eller band; EUR/SEK)
# - Admin (lanes + settings)
# - Excel export: V+H på skalers

import io
import json
import math
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GN Freight WebApp", layout="wide")
__version__ = "2025-12-02.2"  # bump on release

# ------------------------ Helpers ------------------------

def clamp_number(v):
    try:
        if v is None or v == "":
            return None
        n = float(v)
        if math.isfinite(n):
            return n
    except Exception:
        pass
    return None


def round_to_step(value: float, step: float) -> float:
    s = step or 1.0
    return round(value / s) * s


def lin_reg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, r2 for simple linear regression."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    return slope, intercept, r2


def fit_power_model(rows: List[Dict]):
    x, y = [], []
    for r in rows:
        kg = clamp_number(r.get("breakToKg"))
        rate = clamp_number(r.get("ratePerKg"))
        if kg and rate and kg > 0 and rate > 0:
            x.append(math.log(kg))
            y.append(math.log(rate))
    if len(x) < 2:
        return dict(b=float("nan"), c=float("nan"), A=float("nan"), r2=float("nan"))
    x = np.array(x); y = np.array(y)
    b, c, r2 = lin_reg(x, y)
    A = math.exp(c)
    return dict(b=b, c=c, A=A, r2=r2)


def fit_exp_model(rows: List[Dict]):
    x, y = [], []
    for r in rows:
        kg = clamp_number(r.get("breakToKg"))
        rate = clamp_number(r.get("ratePerKg"))
        if kg and rate and kg > 0 and rate > 0:
            x.append(kg)
            y.append(math.log(rate))
    if len(x) < 2:
        return dict(d=float("nan"), c=float("nan"), A=float("nan"), r2=float("nan"))
    x = np.array(x); y = np.array(y)
    d, c, r2 = lin_reg(x, y)
    A = math.exp(c)
    return dict(d=d, c=c, A=A, r2=r2)


def price_from_ladder(weight_kg: float, ladder: List[Dict]):
    if not ladder or not weight_kg:
        return dict(total=0.0, rate=0.0, min=0.0, row=None)
    rows = sorted(ladder, key=lambda r: r.get("breakToKg", 0))
    chosen = rows[-1]
    for r in rows:
        if weight_kg <= (r.get("breakToKg") or 0):
            chosen = r
            break
    rate = float(chosen.get("ratePerKg") or 0.0)
    min_eur = float(chosen.get("minEUR") or 0.0)
    total = max(weight_kg * rate, min_eur)
    return dict(total=total, rate=rate, min=min_eur, row=chosen)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write multiple DataFrames to an in-memory .xlsx (tries xlsxwriter then openpyxl)."""
    last_err = None
    for eng in ("xlsxwriter", "openpyxl"):
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine=eng) as w:
                for name, df in sheets.items():
                    df.to_excel(w, sheet_name=name, index=False)
            return buf.getvalue()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Excel export needs xlsxwriter or openpyxl. Last error: {last_err}")


# ---- Robust parser for numbers & ranges ----
# Accepts tokens like:
#   500                     → [500]
#   500, 1000 2000          → [500, 1000, 2000]
#   500-2500:500            → 500,1000,1500,2000,2500 (inclusive)
#   500-2500@5              → 5 evenly spaced points between 500 and 2500 (inclusive)
#   1_000–2_500:250         → same (underscores allowed, en-dash/em-dash ok)
# Whitespace/comma/semicolon separated. Duplicates removed, sorted ascending.

def parse_numbers_and_ranges(text: str) -> List[float]:
    if not text:
        return []
    # normalize
    t = (text or "").strip()
    t = t.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash → hyphen
    t = t.replace(";", " ").replace(",", " ")
    tokens = [tok for tok in re.split(r"\s+", t) if tok]

    out: List[float] = []
    for tok in tokens:
        tok = tok.replace("_", "")  # allow 1_000 style
        m = re.fullmatch(r"(?i)\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)(?::([0-9]+(?:\.[0-9]+)?))?(?:@([0-9]+))?\s*", tok)
        if m:
            a = float(m.group(1)); b = float(m.group(2))
            step_s = m.group(3); count_s = m.group(4)
            if count_s and step_s:
                count_s = None  # explicit step wins
            if step_s:
                step = float(step_s)
                if step <= 0:
                    continue
                n = int(math.floor((b - a) / step + 0.0000001))
                vals = [a + i * step for i in range(0, n + 1)]
                if len(vals) == 0 or vals[-1] < b - 1e-9:
                    vals.append(b)
            elif count_s:
                cnt = max(2, int(count_s))
                vals = list(np.linspace(a, b, cnt))
            else:
                vals = [a, b]
            out.extend(vals)
        else:
            n = clamp_number(tok)
            if n is not None:
                out.append(float(n))
    # dedupe + sort
    out = sorted({round(v, 6) for v in out})
    return out

# ------------------------ Defaults & State ------------------------

DEFAULT_STATE = {
    "settings": {
        "kgPerFLM": 1850.0,
        "flmPerPallet": 0.4,
        "roundingStep": 1.0,
        "kgPerContainer": 1000.0,
    },
    "lanes": {
        "SE->FR": {
            "perKg": [
                {"breakToKg": 500, "ratePerKg": 0.45, "minEUR": 120},
                {"breakToKg": 1000, "ratePerKg": 0.38, "minEUR": 150},
                {"breakToKg": 2500, "ratePerKg": 0.28, "minEUR": 220},
                {"breakToKg": 5000, "ratePerKg": 0.22, "minEUR": 330},
                {"breakToKg": 10000, "ratePerKg": 0.18, "minEUR": 480},
                {"breakToKg": 25160, "ratePerKg": 0.16, "minEUR": 650},
            ],
            "perPallet": {"rate": 95.0, "min": 150.0},
            "perFLM": {"rate": 310.0, "min": 500.0},
        },
        "SE->PT": {
            "perKg": [
                {"breakToKg": 500, "ratePerKg": 0.50, "minEUR": 140},
                {"breakToKg": 1000, "ratePerKg": 0.42, "minEUR": 170},
                {"breakToKg": 2500, "ratePerKg": 0.32, "minEUR": 250},
                {"breakToKg": 5000, "ratePerKg": 0.26, "minEUR": 360},
                {"breakToKg": 10000, "ratePerKg": 0.21, "minEUR": 520},
                {"breakToKg": 25160, "ratePerKg": 0.18, "minEUR": 720},
            ],
            "perPallet": {"rate": 120.0, "min": 180.0},
            "perFLM": {"rate": 360.0, "min": 560.0},
        },
    },
    "surcharges": {
        "fuelPct": 0.0,
        "marpolPct": 0.0,
        "extraPct": 0.0,
        "perLdmEUR": 0.0,
        "perKgEUR": 0.0,
        "flatEUR": 0.0,
        "frClass": 4,
        "frKm": 0.0,
        "frEurPerKmClass3": 0.27,
        "frEurPerKmClass4": 0.33,
        "frSpecialEUR": 0.0,
        "includeFRinFlat": True,
    },
}

if "app_state" not in st.session_state:
    st.session_state.app_state = DEFAULT_STATE

state = st.session_state.app_state
state["settings"].setdefault("kgPerContainer", 1000.0)

# ------------------------ Sidebar ------------------------

st.sidebar.title("Settings")
st.sidebar.caption(f"Version: {__version__}")
state["settings"]["kgPerFLM"] = st.sidebar.number_input(
    "kg per FLM", value=state["settings"]["kgPerFLM"], step=10.0, key="sidebar_kgperflm"
)
state["settings"]["flmPerPallet"] = st.sidebar.number_input(
    "FLM per pallet", value=state["settings"]["flmPerPallet"], step=0.01, format="%.2f", key="sidebar_flmperpallet"
)
state["settings"]["roundingStep"] = st.sidebar.number_input(
    "Rounding step (€)", value=state["settings"]["roundingStep"], step=1.0, key="sidebar_round"
)
state["settings"]["kgPerContainer"] = st.sidebar.number_input(
    "kg per Container", value=state["settings"]["kgPerContainer"], step=10.0, key="sidebar_kgpercontainer"
)

with st.sidebar.expander("Global surcharges (defaults)", expanded=False):
    s = state["surcharges"]
    s["fuelPct"]   = st.number_input("Fuel % (e.g. 0.12 = 12%)", value=s["fuelPct"], step=0.01, format="%.4f", key="sidebar_fuel")
    s["marpolPct"] = st.number_input("MARPOL %", value=s["marpolPct"], step=0.01, format="%.4f", key="sidebar_marpol")
    s["extraPct"]  = st.number_input("Extra %", value=s["extraPct"], step=0.01, format="%.4f", key="sidebar_extra")
    s["perLdmEUR"] = st.number_input("Per LDM €", value=s["perLdmEUR"], step=1.0, key="sidebar_perldm")
    s["perKgEUR"]  = st.number_input("Per kg €", value=s["perKgEUR"], step=0.001, format="%.4f", key="sidebar_perkg")
    s["flatEUR"]   = st.number_input("Flat €", value=s["flatEUR"], step=1.0, key="sidebar_flat")
    st.write("**FR roadfee helper**")
    s["frClass"]   = int(st.selectbox("FR Class", options=[3,4], index=1 if s["frClass"]==4 else 0, key="sidebar_fr_class"))
    s["frKm"]      = st.number_input("FR toll km", value=s["frKm"], step=10.0, key="sidebar_fr_km")
    s["frEurPerKmClass3"] = st.number_input("€/km Class 3", value=s["frEurPerKmClass3"], step=0.01, key="sidebar_fr_c3")
    s["frEurPerKmClass4"] = st.number_input("€/km Class 4", value=s["frEurPerKmClass4"], step=0.01, key="sidebar_fr_c4")
    s["frSpecialEUR"] = st.number_input("Special points €", value=s["frSpecialEUR"], step=1.0, key="sidebar_fr_special")
    s["includeFRinFlat"] = st.checkbox("Include FR roadfee in Flat", value=s["includeFRinFlat"], key="sidebar_fr_include")

with st.sidebar.expander("Data: lanes import/export", expanded=False):
    lanes_json = json.dumps(state["lanes"], indent=2)
    st.download_button("Download lanes.json", data=lanes_json, file_name="lanes.json", key="sidebar_dl_lanes")
    up = st.file_uploader("Upload lanes.json", type=["json"], key="sidebar_upload_lanes")
    if up is not None:
        try:
            loaded = json.load(up)
            assert isinstance(loaded, dict)
            state["lanes"] = loaded
            st.success("Lanes loaded.")
        except Exception as e:
            st.error(f"Failed to load: {e}")

# ------------------------ Tabs ------------------------

tabs = st.tabs([
    "Calculator",
    "LDM Scaler",
    "Weight Scaler",
    "Weight from FTL",
    "Containers",
    "Pallet Scaler",
    "Admin",
])

# ------------------------ Calculator ------------------------
with tabs[0]:
    st.subheader("Calculator")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        origin = st.text_input("Origin", value="SE", key="calc_origin")
    with colB:
        dest = st.text_input("Destination", value="FR", key="calc_dest")
    with colC:
        pallets = st.number_input("Pallets", value=10, step=1, key="calc_pallets")
    with colD:
        weight = st.number_input("Weight (kg)", value=3500, step=50, key="calc_weight")

    colE, colF, colG, colH = st.columns(4)
    with colE:
        ldm_manual = st.text_input("LDM (optional)", value="", key="calc_ldm")
    with colF:
        tariff = st.selectbox("Tariff", options=["auto","per_kg","per_pallet","per_flm"], index=0, key="calc_tariff")
    with colG:
        kg_per_flm = state["settings"]["kgPerFLM"]
        st.metric("kg/FLM", kg_per_flm)
    with colH:
        flm_per_pallet = state["settings"]["flmPerPallet"]
        st.metric("FLM/pallet", flm_per_pallet)

    lane_id = f"{origin}->{dest}"
    lane = state["lanes"].get(lane_id)
    if not lane:
        st.warning(f"Lane {lane_id} not found. Add it in Admin tab.")
    else:
        ldm_from_pallets = pallets * flm_per_pallet
        ldm_val = clamp_number(ldm_manual) if ldm_manual != "" else ldm_from_pallets
        ldm_val = ldm_val or 0
        kg_by_ldm = ldm_val * kg_per_flm
        chargeable_kg = max(kg_by_ldm, weight)

        perkg = price_from_ladder(chargeable_kg, lane.get("perKg", []))
        rate_pal = lane.get("perPallet", {}).get("rate", 0.0)
        min_pal = lane.get("perPallet", {}).get("min", 0.0)
        total_pal = max(pallets * rate_pal, min_pal)
        rate_flm = lane.get("perFLM", {}).get("rate", 0.0)
        min_flm = lane.get("perFLM", {}).get("min", 0.0)
        total_flm = max(ldm_val * rate_flm, min_flm)

        if tariff == "per_kg":
            base = ("per_kg", perkg["total"])
        elif tariff == "per_pallet":
            base = ("per_pallet", total_pal)
        elif tariff == "per_flm":
            base = ("per_flm", total_flm)
        else:
            candidates = [("per_kg", perkg["total"]), ("per_pallet", total_pal), ("per_flm", total_flm)]
            base = min(candidates, key=lambda t: t[1])

        st.markdown("### Base totals")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Per kg", f"€ {perkg['total']:.2f}")
        c2.metric("Per pallet", f"€ {total_pal:.2f}")
        c3.metric("Per FLM", f"€ {total_flm:.2f}")
        c4.metric("Chosen", f"{base[0]} → € {base[1]:.2f}")

        # Fit från ladder
        rows_for_fit = [
            dict(breakToKg=float(r.get("breakToKg")), ratePerKg=float(r.get("ratePerKg")))
            for r in lane.get("perKg", []) if clamp_number(r.get("breakToKg")) and clamp_number(r.get("ratePerKg"))
        ]
        pow_fit = fit_power_model(rows_for_fit)
        exp_fit = fit_exp_model(rows_for_fit)
        chosen_model = "EXP" if (exp_fit.get("r2") or 0) >= (pow_fit.get("r2") or 0) else "POWER"

        st.markdown("### Surcharges")
        s_local = {}
        cA, cB, cC, cD = st.columns(4)
        with cA:
            s_local["fuelPct"] = st.number_input("Fuel % (decimal)", value=state["surcharges"]["fuelPct"], step=0.01, format="%.4f", key="calc_fuel")
        with cB:
            s_local["marpolPct"] = st.number_input("MARPOL % (decimal)", value=state["surcharges"]["marpolPct"], step=0.01, format="%.4f", key="calc_marpol")
        with cC:
            s_local["extraPct"] = st.number_input("Extra % (decimal)", value=state["surcharges"]["extraPct"], step=0.01, format="%.4f", key="calc_extra")
        with cD:
            rounding_step = st.number_input("Rounding step €", value=state["settings"]["roundingStep"], step=1.0, key="calc_round")

        cE, cF, cG, cH = st.columns(4)
        with cE:
            s_local["perLdmEUR"] = st.number_input("Per LDM €", value=state["surcharges"]["perLdmEUR"], step=1.0, key="calc_perldm")
        with cF:
            s_local["perKgEUR"] = st.number_input("Per kg €", value=state["surcharges"]["perKgEUR"], step=0.001, format="%.4f", key="calc_perkg")
        with cG:
            s_local["flatEUR"] = st.number_input("Flat €", value=state["surcharges"]["flatEUR"], step=1.0, key="calc_flat")
        with cH:
            st.metric("Model (R²)", f"{chosen_model} (P {pow_fit['r2']:.3f} / E {exp_fit['r2']:.3f})")

        st.info("FR roadfee helper below adds into Flat if toggled.")
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            fr_class = int(st.selectbox("FR Class", options=[3,4], index=1 if state["surcharges"]["frClass"]==4 else 0, key="calc_fr_class"))
        with r2:
            fr_km = st.number_input("FR toll km", value=state["surcharges"]["frKm"], step=10.0, key="calc_fr_km")
        with r3:
            fr_c3 = st.number_input("€/km Class 3", value=state["surcharges"]["frEurPerKmClass3"], step=0.01, key="calc_fr_c3")
        with r4:
            fr_c4 = st.number_input("€/km Class 4", value=state["surcharges"]["frEurPerKmClass4"], step=0.01, key="calc_fr_c4")
        with r5:
            fr_special = st.number_input("Special €", value=state["surcharges"]["frSpecialEUR"], step=1.0, key="calc_fr_special")
        include_fr = st.checkbox("Include FR in Flat", value=state["surcharges"]["includeFRinFlat"], key="calc_fr_include")

        per_km = fr_c3 if fr_class == 3 else fr_c4
        fr_fee = fr_km * per_km + fr_special

        percent_eur = base[1] * ((s_local["fuelPct"] or 0) + (s_local["marpolPct"] or 0) + (s_local["extraPct"] or 0))
        per_ldm_eur = ldm_val * (s_local["perLdmEUR"] or 0)
        per_kg_eur = chargeable_kg * (s_local["perKgEUR"] or 0)
        flat_eur = (s_local["flatEUR"] or 0) + (fr_fee if include_fr else 0)

        total_raw = base[1] + percent_eur + per_ldm_eur + per_kg_eur + flat_eur
        total_rounded = round_to_step(total_raw, rounding_step)

        st.markdown("### Totals")
        t1, t2 = st.columns(2)
        with t1:
            df_totals = pd.DataFrame({
                "component": ["base", "percent", "per LDM", "per kg", "flat (incl FR)", "total_raw", "total_rounded"],
                "EUR": [base[1], percent_eur, per_ldm_eur, per_kg_eur, flat_eur, total_raw, total_rounded],
            })
            st.dataframe(df_totals.style.format({"EUR": "€ {:.2f}"}), width='stretch')
        with t2:
            st.metric("Total (rounded)", f"€ {total_rounded:.2f}", help=f"raw € {total_raw:.2f}")

        # Export V+H
        df_v = df_totals.copy()
        df_h = pd.DataFrame([{
            "lane": lane_id,
            "origin": origin,
            "dest": dest,
            "pallets": pallets,
            "ldm": ldm_val,
            "weight_kg": weight,
            "tariff_chosen": base[0],
            "per_kg": perkg["total"],
            "per_pallet": total_pal,
            "per_flm": total_flm,
            "base": base[1],
            "percent": percent_eur,
            "per_LDM": per_ldm_eur,
            "per_kg_add": per_kg_eur,
            "flat_incl_FR": flat_eur,
            "total_raw": total_raw,
            "total_rounded": total_rounded,
        }])
        try:
            xlsx = to_excel_bytes({"Totals_V": df_v, "Totals_H": df_h})
            st.download_button(
                "Export to Excel (V+H)",
                data=xlsx,
                file_name="calculator_totals.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="calc_export_xlsx",
            )
        except Exception as e:
            st.error(str(e))

# ------------------------ LDM Scaler ------------------------
with tabs[1]:
    st.subheader("LDM Scaler – from full truck price or from fitted model")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mode = st.selectbox("Mode", options=["FROM_P13","FROM_FIT"], index=0, key="ldm_mode")
    with col2:
        p13 = st.number_input("P13 (EUR)", value=2000.0, step=10.0, key="ldm_p13")
    with col3:
        model_choice = st.selectbox("Model", options=["AUTO","POWER","EXP"], index=0, key="ldm_model_choice")

    lane_for_fit = state["lanes"].get("SE->FR")
    rows_for_fit = [
        dict(breakToKg=float(r.get("breakToKg")), ratePerKg=float(r.get("ratePerKg")))
        for r in (lane_for_fit.get("perKg", []) if lane_for_fit else [])
        if clamp_number(r.get("breakToKg")) and clamp_number(r.get("ratePerKg"))
    ]
    pow_fit = fit_power_model(rows_for_fit)
    exp_fit = fit_exp_model(rows_for_fit)

    with col4:
        b = st.number_input("POWER: b", value=float(0 if math.isnan(pow_fit.get("b", float("nan"))) else pow_fit.get("b")), step=0.01, format="%.5f", key="ldm_b")
    with col5:
        d = st.number_input("EXP: d", value=float(0 if math.isnan(exp_fit.get("d", float("nan"))) else exp_fit.get("d")), step=0.00001, format="%.6f", key="ldm_d")

    kg_per_flm = state["settings"]["kgPerFLM"]
    kg13 = 13 * kg_per_flm

    ldm_rows = []
    chosen = model_choice
    if chosen == "AUTO":
        chosen = "EXP" if (exp_fit.get("r2") or 0) >= (pow_fit.get("r2") or 0) else "POWER"

    for L in range(1, 14):
        kgL = L * kg_per_flm
        if mode == "FROM_P13":
            if chosen == "POWER":
                price = p13 * ((L/13) ** (b + 1))
            else:
                price = p13 * ((kgL / kg13) * math.exp(d * (kgL - kg13))) if kg13 else float("nan")
        else:
            if chosen == "POWER":
                if abs(b + 1) < 1e-12:
                    price = float("nan")
                else:
                    rawL = (kgL ** (b + 1)) / (b + 1)
                    raw13 = (kg13 ** (b + 1)) / (b + 1)
                    price = p13 * (rawL / raw13)
            else:
                if abs(d) < 1e-12:
                    rawL = kgL; raw13 = kg13
                else:
                    rawL = (math.exp(d * kgL) - 1) / d
                    raw13 = (math.exp(d * kg13) - 1) / d
                price = p13 * (rawL / raw13)
        ldm_rows.append({"LDM": L, "kg": kgL, "Price EUR": price})

    df_ldm = pd.DataFrame(ldm_rows)
    st.dataframe(df_ldm.style.format({"kg": "{:.0f}", "Price EUR": "€ {:.2f}"}), width='stretch')

    # Export V+H
    df_v = df_ldm.copy()
    row = {"kg_per_flm": kg_per_flm, "mode": mode, "model": chosen, "p13": p13, "b": b, "d": d}
    for _, r in df_ldm.iterrows():
        row[f"LDM_{int(r['LDM'])}"] = r["Price EUR"]
    df_h = pd.DataFrame([row])
    try:
        xlsx = to_excel_bytes({"LDM_V": df_v, "LDM_H": df_h})
        st.download_button(
            "Export LDM (V+H)",
            data=xlsx,
            file_name="ldm_scaler.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ldm_export_xlsx",
        )
    except Exception as e:
        st.error(str(e))

# # ------------------------ Weight Scaler (retier) ------------------------
with tabs[2]:
    st.subheader("Weight Scaler – retier to new breaks")

    st.info(
        "Vad visar vad:\n"
        "• **Band Total €** = bara priset för det aktuella intervallet (bandet).\n"
        "• **Cumulative Total €** = pris från 0 kg fram till brytviktens slut (stigande, lätt att sanity-checka).\n"
        "• **Avg €/kg** = bandets genomsnittliga €/kg (användbar som fast band-tariff)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        lane_id = st.text_input("Lane ID", value="SE->FR", key="weight_lane")
    with c2:
        model_sel = st.selectbox("Model", options=["AUTO","POWER","EXP"], index=0, key="weight_model_sel")
    with c3:
        method = st.selectbox("Method", options=["INTEGRATED","POINT"], index=0, key="weight_method_sel")

    lane = state["lanes"].get(lane_id)
    if not lane:
        st.warning("Lane not found. Add it under Admin.")
    else:
        rows_for_fit = [
            dict(breakToKg=float(r.get("breakToKg")), ratePerKg=float(r.get("ratePerKg")))
            for r in lane.get("perKg", []) if clamp_number(r.get("breakToKg")) and clamp_number(r.get("ratePerKg"))
        ]
        pow_fit = fit_power_model(rows_for_fit)
        exp_fit = fit_exp_model(rows_for_fit)
        chosen = model_sel
        if chosen == "AUTO":
            chosen = "EXP" if (exp_fit.get("r2") or 0) >= (pow_fit.get("r2") or 0) else "POWER"

        st.markdown("#### Optional anchor (normalize totals)")
        a1, a2, a3 = st.columns(3)
        with a1:
            anchor_mode = st.selectbox("Anchor type", ["NONE","MATCH_TOTAL_AT_KG"], index=0, key="weight_anchor_mode")
        with a2:
            anchor_kg = st.number_input("Anchor weight (kg)", value=10000.0, step=50.0, key="weight_anchor_kg")
        with a3:
            anchor_total = st.number_input("Anchor total € at weight", value=0.0, step=10.0, key="weight_anchor_total")

        st.caption("Enter breaks as numbers and/or ranges. Examples: '500, 1000-2500:250, 5000, 10000, 25160' or '500-2500@5'.")
        breaks_text = st.text_area(
            "New breaks (kg)",
            value="500, 1000-2500:250, 5000, 10000, 25160",
            key="weight_breaks",
        )

        # Robust parser för tal och intervall (samma som i övriga flikar)
        def parse_numbers_and_ranges(text: str) -> List[float]:
            import re, numpy as np, math
            if not text:
                return []
            t = (text or "").strip()
            t = t.replace("\u2013", "-").replace("\u2014", "-")
            t = t.replace(";", " ").replace(",", " ")
            tokens = [tok for tok in re.split(r"\s+", t) if tok]
            out = []
            for tok in tokens:
                tok = tok.replace("_", "")
                m = re.fullmatch(r"(?i)\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)(?::([0-9]+(?:\.[0-9]+)?))?(?:@([0-9]+))?\s*", tok)
                if m:
                    a = float(m.group(1)); b = float(m.group(2))
                    step_s = m.group(3); count_s = m.group(4)
                    if count_s and step_s:
                        count_s = None
                    if step_s:
                        step = float(step_s)
                        if step <= 0:
                            continue
                        n = int(math.floor((b - a) / step + 1e-9))
                        vals = [a + i * step for i in range(0, n + 1)]
                        if len(vals) == 0 or vals[-1] < b - 1e-9:
                            vals.append(b)
                    elif count_s:
                        cnt = max(2, int(count_s))
                        vals = list(np.linspace(a, b, cnt))
                    else:
                        vals = [a, b]
                    out.extend(vals)
                else:
                    try:
                        n = float(tok)
                        if math.isfinite(n):
                            out.append(float(n))
                    except Exception:
                        pass
            out = sorted({round(v, 6) for v in out})
            return out

        new_breaks = parse_numbers_and_ranges(breaks_text)

        # --- Modellfunktioner (oskalade) ---
        def rate_at_raw(kg: float) -> float:
            if chosen == "EXP":
                A, d = exp_fit["A"], exp_fit["d"]
                return (A or 0) * math.exp((d or 0) * kg)
            else:
                A, b = pow_fit["A"], pow_fit["b"]
                return (A or 0) * (kg ** (b or 0))

        def integral_to_raw(kg: float) -> float:
            if kg <= 0:
                return 0.0
            if chosen == "EXP":
                A, d = exp_fit["A"], exp_fit["d"]
                if abs(d or 0) < 1e-12:
                    return (A or 0) * kg
                return (A / d) * (math.exp(d * kg) - 1)
            else:
                A, b = pow_fit["A"], pow_fit["b"]
                if abs((b or 0) + 1) < 1e-12:
                    return (A or 0) * math.log(max(kg, 1e-9))
                return (A / (b + 1)) * (kg ** (b + 1))

        # --- Normalisering via anchor ---
        scale = 1.0
        if anchor_mode == "MATCH_TOTAL_AT_KG" and (anchor_total or 0) > 0 and (anchor_kg or 0) > 0:
            model_total = integral_to_raw(anchor_kg)
            if model_total and model_total > 0:
                scale = anchor_total / model_total
        st.caption(f"Anchor scale factor: {scale:.6f}")

        def rate_at(kg: float) -> float:
            return scale * rate_at_raw(kg)

        def integral_to(kg: float) -> float:
            return scale * integral_to_raw(kg)

        # --- Beräkna band + kumulativt ---
        out_rows = []
        start = 0.0
        cum_total = 0.0
        for brk in new_breaks:
            width = brk - start
            r_break = rate_at(brk)
            if method == "INTEGRATED":
                band_total = integral_to(brk) - integral_to(start)
            else:
                band_total = r_break * width
            avg_eur_per_kg = band_total / width if width > 0 else float("nan")
            cum_total += band_total
            out_rows.append({
                "BreakToKg": brk,
                "BandStartKg": start,
                "WidthKg": width,
                "Rate@break €/kg": r_break,
                "Band Total €": band_total,
                "Cumulative Total €": cum_total,
                "Avg €/kg": avg_eur_per_kg,
            })
            start = brk

        df_out = pd.DataFrame(out_rows)
        st.dataframe(df_out.style.format({
            "BandStartKg": "{:.0f}",
            "BreakToKg": "{:.0f}",
            "WidthKg": "{:.0f}",
            "Rate@break €/kg": "{:.4f}",
            "Band Total €": "€ {:.2f}",
            "Cumulative Total €": "€ {:.2f}",
            "Avg €/kg": "{:.4f}",
        }), width='stretch')

        # Export CSV + (om du har Excel-funktionen) V+H
        st.download_button(
            "Download new ladder (CSV)",
            data=df_out.to_csv(index=False),
            file_name=f"{lane_id.replace('>','-')}-retiered.csv",
            key="weight_download"
        )

        # Om din app-version har to_excel_bytes, exportera även V+H:
        try:
            df_v = df_out.copy()
            row = {"lane": lane_id, "model": chosen, "method": method,
                   "anchor_mode": anchor_mode, "anchor_kg": anchor_kg,
                   "anchor_total": anchor_total, "scale": scale}
            for _, r in df_out.iterrows():
                brk = int(r["BreakToKg"])
                row[f"Band_≤{brk}_Total€"] = r["Band Total €"]
                row[f"Cumulative_≤{brk}_Total€"] = r["Cumulative Total €"]
                row[f"Avg_≤{brk}_€/kg"] = r["Avg €/kg"]
            df_h = pd.DataFrame([row])

            xlsx = to_excel_bytes({"Retier_V": df_v, "Retier_H": df_h})
            st.download_button(
                "Export Weight Scaler (Excel V+H)",
                data=xlsx,
                file_name="weight_scaler.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="weight_export_xlsx",
            )
        except Exception:
            pass

# ------------------------ Weight from FTL (manual weights/ranges) ------------------------
with tabs[3]:
    st.subheader("Weight from FTL – scale totals by weight from an exact FTL price")

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        lane_id = st.text_input("Lane ID (for optional fit)", value="SE->FR", key="wftl_lane")
    with c1:
        ftl_kg_default = 13 * state["settings"]["kgPerFLM"]
        anchor_kg = st.number_input("FTL kg (anchor weight)", value=float(ftl_kg_default), step=100.0, key="wftl_anchor_kg")
    with c2:
        ftl_price = st.number_input("FTL price (€)", value=2000.0, step=10.0, key="wftl_ftl_price")
    with c3:
        rounding_step = st.number_input("Rounding step €", value=state["settings"]["roundingStep"], step=1.0, key="wftl_round")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        model_sel = st.selectbox("Model", options=["AUTO","POWER","EXP","MANUAL_POWER","MANUAL_EXP"], index=0, key="wftl_model")
    with m2:
        b_par = st.number_input("POWER: b", value=-0.25, step=0.01, format="%.5f", key="wftl_b")
    with m3:
        d_par = st.number_input("EXP: d", value=-0.00002, step=0.00001, format="%.6f", key="wftl_d")
    with m4:
        st.caption("AUTO väljer bästa fit från lane om möjligt; annars används b/d ovan.")

    st.caption("Ange vikter som lista och/eller ranges. Exempel: '500 750 1000-2500:250 10000 25160' eller '500-2500@5'.")
    breaks_text = st.text_area(
        "Weights to price (kg)",
        value="500, 1000-2500:250, 5000, 10000, 25160",
        key="wftl_breaks",
    )
    weight_list = parse_numbers_and_ranges(breaks_text)

    lane = state["lanes"].get(lane_id)
    rows_for_fit = []
    if lane:
        rows_for_fit = [
            dict(breakToKg=float(r.get("breakToKg")), ratePerKg=float(r.get("ratePerKg")))
            for r in lane.get("perKg", []) if clamp_number(r.get("breakToKg")) and clamp_number(r.get("ratePerKg"))
        ]
    pow_fit = fit_power_model(rows_for_fit) if rows_for_fit else dict(b=b_par, A=1.0, r2=float("nan"))
    exp_fit = fit_exp_model(rows_for_fit) if rows_for_fit else dict(d=d_par, A=1.0, r2=float("nan"))

    chosen = model_sel
    if chosen == "AUTO":
        if rows_for_fit and (exp_fit.get("r2") or 0) >= (pow_fit.get("r2") or 0):
            chosen = "EXP"
        else:
            chosen = "POWER"

    if chosen in ("POWER","MANUAL_POWER"):
        b = (pow_fit.get("b") if chosen == "POWER" else b_par) or 0.0
        A_raw = pow_fit.get("A", 1.0) if chosen == "POWER" else 1.0
        def integral_raw(kg: float) -> float:
            if kg <= 0: return 0.0
            if abs(b + 1) < 1e-12:
                return A_raw * math.log(max(kg, 1e-9))
            return (A_raw / (b + 1)) * (kg ** (b + 1))
    else:
        d = (exp_fit.get("d") if chosen == "EXP" else d_par) or 0.0
        A_raw = exp_fit.get("A", 1.0) if chosen == "EXP" else 1.0
        def integral_raw(kg: float) -> float:
            if kg <= 0: return 0.0
            if abs(d) < 1e-12:
                return A_raw * kg
            return (A_raw / d) * (math.exp(d * kg) - 1)

    base = integral_raw(anchor_kg)
    scale = (ftl_price / base) if base and base > 0 else 0.0
    st.caption(f"Normalization scale = {scale:.6f} (so total at {anchor_kg:.0f} kg = € {ftl_price:.2f})")

    rows = []
    for kg in weight_list:
        total = scale * integral_raw(kg)
        rounded = round_to_step(total, rounding_step)
        avg_per_kg = total / kg if kg > 0 else float("nan")
        rows.append({
            "kg": kg,
            "Total EUR": total,
            "Rounded EUR": rounded,
            "Avg €/kg": avg_per_kg,
        })
    df_wftl = pd.DataFrame(rows)
    st.dataframe(df_wftl.style.format({
        "kg": "{:.0f}",
        "Total EUR": "€ {:.2f}",
        "Rounded EUR": "€ {:.2f}",
        "Avg €/kg": "{:.4f}",
    }), width='stretch')

    # Export V+H
    df_v = df_wftl.copy()
    row = {
        "lane": lane_id,
        "model": chosen,
        "anchor_kg": anchor_kg,
        "ftl_price": ftl_price,
        "round_step": rounding_step,
        "scale": scale,
    }
    for _, r in df_wftl.iterrows():
        k = int(r["kg"])
        row[f"KG_{k}_Total€"] = r["Total EUR"]
        row[f"KG_{k}_Rounded€"] = r["Rounded EUR"]
    df_h = pd.DataFrame([row])
    try:
        xlsx = to_excel_bytes({"WFTL_V": df_v, "WFTL_H": df_h})
        st.download_button(
            "Export Weight-from-FTL (V+H)",
            data=xlsx,
            file_name="weight_from_ftl.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="wftl_export_xlsx",
        )
    except Exception as e:
        st.error(str(e))

# ------------------------ Containers Scaler ------------------------
with tabs[4]:
    st.subheader("Containers – scale 1–19 from FTL (20 containers)")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        p20 = st.number_input("FTL = 20 containers (EUR)", value=2000.0, step=10.0, key="cont_p20")
    with col2:
        model_choice = st.selectbox("Model", options=["POWER","EXP","AUTO"], index=0, key="containers_model_choice")
    with col3:
        b_cont = st.number_input("POWER: b", value=-0.25, step=0.01, format="%.5f", key="cont_b")
    with col4:
        d_cont = st.number_input("EXP: d", value=-0.00002, step=0.00001, format="%.6f", key="cont_d")
    with col5:
        kg_per_container = st.number_input("kg per Container", value=float(state["settings"].get("kgPerContainer", 1000.0)), step=10.0, key="cont_kg")
    with col6:
        rounding_step = st.number_input("Rounding step €", value=state["settings"].get("roundingStep", 1.0), step=1.0, key="cont_round")

    kg20 = 20 * kg_per_container
    chosen = model_choice if model_choice != "AUTO" else "POWER"

    rows = []
    for n in range(1, 20):
        kg_n = n * kg_per_container
        if chosen == "POWER":
            price = p20 * ((n/20) ** (b_cont + 1))
        else:
            price = p20 * ((kg_n / kg20) * math.exp(d_cont * (kg_n - kg20))) if kg20 else float("nan")
        rounded = round_to_step(price, rounding_step)
        rows.append({
            "Containers": n,
            "kg (est)": kg_n,
            "Total EUR": price,
            "Rounded EUR": rounded,
            "EUR/container": price / n if n>0 else float("nan"),
        })
    rows.append({
        "Containers": 20,
        "kg (est)": kg20,
        "Total EUR": p20,
        "Rounded EUR": round_to_step(p20, rounding_step),
        "EUR/container": p20/20,
    })

    df_cont = pd.DataFrame(rows)
    st.dataframe(df_cont.style.format({
        "kg (est)": "{:.0f}",
        "Total EUR": "€ {:.2f}",
        "Rounded EUR": "€ {:.2f}",
        "EUR/container": "€ {:.2f}"
    }), width='stretch')

    # Export V+H
    df_v = df_cont.copy()
    row = {"kg_per_container": kg_per_container, "model": chosen, "p20": p20, "b": b_cont, "d": d_cont, "rounding_step": rounding_step}
    for _, r in df_cont.iterrows():
        n = int(r["Containers"])
        row[f"N{n}_Total€"] = r["Total EUR"]
        row[f"N{n}_Rounded€"] = r["Rounded EUR"]
    df_h = pd.DataFrame([row])
    try:
        xlsx = to_excel_bytes({"Containers_V": df_v, "Containers_H": df_h})
        st.download_button(
            "Export Containers (V+H)",
            data=xlsx,
            file_name="containers_scaler.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="cont_export_xlsx",
        )
    except Exception as e:
        st.error(str(e))

# ------------------------ Pallet Scaler ------------------------
with tabs[5]:
    st.subheader("Pallet Scaler – model or bands (EUR/SEK)")

    c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    with c0:
        currency = st.selectbox("Currency", options=["EUR","SEK"], index=0, key="pallet_currency")
    with c1:
        fx = st.number_input("EUR→SEK", value=11.00, step=0.10, format="%.2f", key="pallet_fx")
    with c2:
        rounding_step_disp = st.number_input("Rounding step (in chosen currency)", value=1.0, step=1.0, key="pallet_round_step")
    with c3:
        mode = st.selectbox("Mode", options=["MODEL","BANDS"], index=0, key="pallet_mode")

    def to_eur(amount_in_display: float) -> float:
        if currency == "EUR":
            return amount_in_display or 0.0
        return (amount_in_display or 0.0) / (fx or 1.0)

    def to_disp(amount_eur: float) -> float:
        if currency == "EUR":
            return amount_eur or 0.0
        return (amount_eur or 0.0) * (fx or 1.0)

    sym = "€" if currency == "EUR" else "kr"

    b1, b2 = st.columns(2)
    with b1:
        ftl_pallets = int(st.number_input("FTL pallets (e.g. 33 or 34)", value=34, step=1, min_value=1, max_value=120, key="pallet_ftl_count"))
    with b2:
        ftl_price_disp = st.number_input(f"FTL total ({sym})", value=2000.0 if currency=="EUR" else 2000.0*fx, step=10.0, key="pallet_ftl_price")
    ftl_price_eur = to_eur(ftl_price_disp)

    if mode == "MODEL":
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            model_choice = st.selectbox("Model", options=["AUTO","POWER","EXP"], index=0, key="pallet_model_choice")
        with m2:
            b_par = st.number_input("POWER: b", value=-0.25, step=0.01, format="%.5f", key="pallet_power_b")
        with m3:
            d_par = st.number_input("EXP: d", value=-0.00002, step=0.00001, format="%.6f", key="pallet_exp_d")
        with m4:
            ensure_match_ftl = st.checkbox("Normalize so N=FTL equals FTL total", value=True, key="pallet_norm_to_ftl")

        chosen = model_choice
        if chosen == "AUTO":
            chosen = "POWER"

        rows = []
        for n in range(1, ftl_pallets + 1):
            if chosen == "POWER":
                total_eur = ftl_price_eur * ((n / ftl_pallets) ** (b_par + 1))
            else:
                if ftl_pallets > 0:
                    total_eur = ftl_price_eur * ((n / ftl_pallets) * math.exp(d_par * (n - ftl_pallets)))
                else:
                    total_eur = float("nan")
            rows.append({"Pallets": n, "Total (EUR)": total_eur})

        df = pd.DataFrame(rows)
        if ensure_match_ftl and ftl_pallets > 0:
            last = df.loc[df["Pallets"] == ftl_pallets, "Total (EUR)"].values
            if len(last) and last[0]:
                scale = ftl_price_eur / last[0]
                df["Total (EUR)"] = df["Total (EUR)"] * scale

        df["Total (disp)"] = df["Total (EUR)"].apply(to_disp)
        df["Rounded (disp)"] = df["Total (disp)"] .apply(lambda x: round_to_step(x, rounding_step_disp))
        df[f"{sym}/pallet (rounded)"] = df.apply(lambda r: (r["Rounded (disp)"] / r["Pallets"]) if r["Pallets"]>0 else float("nan"), axis=1)

        st.dataframe(
            df[["Pallets","Total (disp)","Rounded (disp)",f"{sym}/pallet (rounded)"]].rename(
                columns={"Total (disp)": f"Total ({sym})", "Rounded (disp)": f"Rounded ({sym})"}
            ).style.format({f"Total ({sym})": f"{sym} " + "{:.2f}", f"Rounded ({sym})": f"{sym} " + "{:.2f}", f"{sym}/pallet (rounded)": f"{sym} " + "{:.2f}"}),
            width='stretch'
        )

        df_v = df[["Pallets","Total (EUR)","Total (disp)","Rounded (disp)",f"{sym}/pallet (rounded)"]].rename(
            columns={"Total (disp)": f"Total ({sym})", "Rounded (disp)": f"Rounded ({sym})"}
        ).copy()
        row = {"currency": currency, "fx_EUR_to_SEK": fx, "round_step_disp": rounding_step_disp,
               "ftl_pallets": ftl_pallets, "ftl_total_disp": ftl_price_disp, "model": chosen, "b": b_par, "d": d_par}
        for _, r in df.iterrows():
            n = int(r["Pallets"]); row[f"N{n}_Rounded({sym})"] = r["Rounded (disp)"]
        df_h = pd.DataFrame([row])

        try:
            xlsx = to_excel_bytes({"Pallet_V": df_v, "Pallet_H": df_h})
            st.download_button(
                "Export Pallet Scaler (V+H)",
                data=xlsx,
                file_name="pallet_scaler.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="pallet_export_xlsx",
            )
        except Exception as e:
            st.error(str(e))

    else:
        st.markdown("#### Bands (intervals)")
        st.caption("Ex: 1–3 pall = X/pall, 4–6 = Y/pall, etc. Priser i vald valuta.")

        default_bands = pd.DataFrame([
            {"From": 1, "To": 3, "Rate_per_pallet": 150 if currency=="EUR" else 150*fx, "Min_total": 0},
            {"From": 4, "To": 6, "Rate_per_pallet": 120 if currency=="EUR" else 120*fx, "Min_total": 0},
            {"From": 7, "To": 10, "Rate_per_pallet": 100 if currency=="EUR" else 100*fx, "Min_total": 0},
            {"From": 11, "To": 34, "Rate_per_pallet": 90 if currency=="EUR" else 90*fx, "Min_total": 0},
        ])

        df_bands = st.data_editor(
            default_bands,
            num_rows="dynamic",
            width='stretch',
            key="pallet_bands_editor"
        )

        normalize_to_ftl = st.checkbox("Normalize totals so N=FTL equals FTL total", value=True, key="pallet_bands_norm")

        def band_for_n(n: int):
            for _, r in df_bands.iterrows():
                f = int(clamp_number(r.get("From")) or 0)
                t = int(clamp_number(r.get("To")) or 0)
                if f <= n <= t:
                    rate_disp = clamp_number(r.get("Rate_per_pallet")) or 0.0
                    min_total_disp = clamp_number(r.get("Min_total")) or 0.0
                    return rate_disp, min_total_disp
            if len(df_bands) > 0:
                r = df_bands.iloc[-1]
                return clamp_number(r.get("Rate_per_pallet")) or 0.0, clamp_number(r.get("Min_total")) or 0.0
            return 0.0, 0.0

        rows = []
        for n in range(1, ftl_pallets + 1):
            rate_disp, min_total_disp = band_for_n(n)
            total_disp = max(rate_disp * n, min_total_disp)
            rows.append({"Pallets": n, "Total_disp": total_disp})

        df = pd.DataFrame(rows)

        if normalize_to_ftl and ftl_pallets > 0:
            last = df.loc[df["Pallets"] == ftl_pallets, "Total_disp"].values
            if len(last) and last[0]:
                scale = (ftl_price_disp or 0.0) / last[0]
                df["Total_disp"] = df["Total_disp"] * scale

        df["Rounded_disp"] = df["Total_disp"].apply(lambda x: round_to_step(x, rounding_step_disp))
        df[f"{sym}/pallet (rounded)"] = df.apply(lambda r: (r["Rounded_disp"] / r["Pallets"]) if r["Pallets"]>0 else float("nan"), axis=1)

        st.dataframe(
            df[["Pallets","Total_disp","Rounded_disp",f"{sym}/pallet (rounded)"]].rename(
                columns={"Total_disp": f"Total ({sym})", "Rounded_disp": f"Rounded ({sym})"}
            ).style.format({f"Total ({sym})": f"{sym} " + "{:.2f}", f"Rounded ({sym})": f"{sym} " + "{:.2f}", f"{sym}/pallet (rounded)": f"{sym} " + "{:.2f}"}),
            width='stretch'
        )

        df_v = df[["Pallets","Total_disp","Rounded_disp",f"{sym}/pallet (rounded)"]].rename(
            columns={"Total_disp": f"Total ({sym})", "Rounded_disp": f"Rounded ({sym})"}
        ).copy()
        row = {"currency": currency, "fx_EUR_to_SEK": fx, "round_step_disp": rounding_step_disp,
               "ftl_pallets": ftl_pallets, "ftl_total_disp": ftl_price_disp, "mode": "BANDS"}
        for _, r in df.iterrows():
            n = int(r["Pallets"]); row[f"N{n}_Rounded({sym})"] = r["Rounded_disp"]
        df_h = pd.DataFrame([row])

        try:
            xlsx = to_excel_bytes({"Pallet_V": df_v, "Pallet_H": df_h})
            st.download_button(
                "Export Pallet Scaler (V+H)",
                data=xlsx,
                file_name="pallet_scaler.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="pallet_export_xlsx_bands",
            )
        except Exception as e:
            st.error(str(e))

# ------------------------ Admin ------------------------
with tabs[6]:
    st.subheader("Admin – lanes & settings")
    st.info("Use the JSON export/import in the sidebar for backups. Data persists in your browser session.")

    lane_ids = sorted(state["lanes"].keys())
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        selected_lane = st.selectbox("Select lane", options=lane_ids, index=0 if lane_ids else None, key="admin_select_lane")
    with col2:
        new_from = st.text_input("New lane: origin", value="SE", key="admin_new_from")
    with col3:
        new_to = st.text_input("New lane: dest", value="FR", key="admin_new_to")

    if st.button("Add lane", key="admin_add_lane"):
        new_id = f"{new_from}->{new_to}"
        if new_id in state["lanes"]:
            st.warning("Lane already exists.")
        else:
            state["lanes"][new_id] = {"perKg": [], "perPallet": {"rate": 0, "min": 0}, "perFLM": {"rate": 0, "min": 0}}
            st.success(f"Added {new_id}")

    if selected_lane:
        lane = state["lanes"][selected_lane]
        st.markdown(f"### Edit lane {selected_lane}")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Per-kg ladder (breaks ascending)")
            df = pd.DataFrame(lane.get("perKg", []))
            df = st.data_editor(df, num_rows="dynamic", width='stretch', key=f"edit_perkg_{selected_lane}")
            lane["perKg"] = [
                {"breakToKg": clamp_number(r.get("breakToKg")) or 0,
                 "ratePerKg": clamp_number(r.get("ratePerKg")) or 0,
                 "minEUR": clamp_number(r.get("minEUR")) or 0}
                for _, r in df.fillna(0).iterrows()
            ]
        with c2:
            st.write("Per pallet")
            lane.setdefault("perPallet", {})
            lane["perPallet"]["rate"] = st.number_input("€/pallet", value=float(lane["perPallet"].get("rate", 0)), step=1.0, key="admin_perpallet_rate")
            lane["perPallet"]["min"]  = st.number_input("Min € (pallet)", value=float(lane["perPallet"].get("min", 0)), step=1.0, key="admin_perpallet_min")
            st.write("Per FLM")
            lane.setdefault("perFLM", {})
            lane["perFLM"]["rate"] = st.number_input("€/FLM", value=float(lane["perFLM"].get("rate", 0)), step=1.0, key="admin_perflm_rate")
            lane["perFLM"]["min"]  = st.number_input("Min € (FLM)", value=float(lane["perFLM"].get("min", 0)), step=1.0, key="admin_perflm_min")

    st.divider()
    if st.button("Reset to defaults", key="admin_reset_btn"):
        st.session_state.app_state = DEFAULT_STATE
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
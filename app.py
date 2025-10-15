# GN Freight WebApp – Streamlit MVP (single-file)
# -------------------------------------------------------------
# Run locally:
#   pip install streamlit numpy pandas
#   streamlit run app.py
#
# Features:
# - Calculator: per-kg ladder (breaks+min), per-pallet, per-FLM, Auto-lowest
# - Curve fit: POWER & EXP from kg ladder (shows R²)
# - Surcharges: fuel %, MARPOL %, extra %, per-LDM, per-kg, flat; FR roadfee helper
# - LDM Scaler: 1–13 LDM from P13 (POWER/EXP) or from fitted model
# - Weight Scaler: retier to new weight breaks (INTEGRATED exact / POINT approx)
# - Admin: manage lanes, settings, import/export JSON

import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GN Freight WebApp", layout="wide")

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
    x = np.array(x)
    y = np.array(y)
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
    x = np.array(x)
    y = np.array(y)
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


# ------------------------ Data Model & Defaults ------------------------
DEFAULT_STATE = {
    "settings": {
        "kgPerFLM": 1850.0,
        "flmPerPallet": 0.4,
        "roundingStep": 1.0,
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
        # FR roadfee helper
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

# ------------------------ Sidebar: Settings & Data ------------------------
st.sidebar.title("Settings")
state["settings"]["kgPerFLM"] = st.sidebar.number_input("kg per FLM", value=state["settings"]["kgPerFLM"], step=10.0)
state["settings"]["flmPerPallet"] = st.sidebar.number_input("FLM per pallet", value=state["settings"]["flmPerPallet"], step=0.01, format="%.2f")
state["settings"]["roundingStep"] = st.sidebar.number_input("Rounding step (€)", value=state["settings"]["roundingStep"], step=1.0)

with st.sidebar.expander("Global surcharges (defaults)"):
    s = state["surcharges"]
    s["fuelPct"] = st.number_input("Fuel % (e.g. 0.12 = 12%)", value=s["fuelPct"], step=0.01, format="%.4f")
    s["marpolPct"] = st.number_input("MARPOL %", value=s["marpolPct"], step=0.01, format="%.4f")
    s["extraPct"] = st.number_input("Extra %", value=s["extraPct"], step=0.01, format="%.4f")
    s["perLdmEUR"] = st.number_input("Per LDM €", value=s["perLdmEUR"], step=1.0)
    s["perKgEUR"] = st.number_input("Per kg €", value=s["perKgEUR"], step=0.001, format="%.4f")
    s["flatEUR"] = st.number_input("Flat €", value=s["flatEUR"], step=1.0)
    st.write("**FR roadfee helper**")
    s["frClass"] = int(st.selectbox("FR Class", options=[3,4], index=1 if s["frClass"]==4 else 0))
    s["frKm"] = st.number_input("FR toll km", value=s["frKm"], step=10.0)
    s["frEurPerKmClass3"] = st.number_input("€/km Class 3", value=s["frEurPerKmClass3"], step=0.01)
    s["frEurPerKmClass4"] = st.number_input("€/km Class 4", value=s["frEurPerKmClass4"], step=0.01)
    s["frSpecialEUR"] = st.number_input("Special points €", value=s["frSpecialEUR"], step=1.0)
    s["includeFRinFlat"] = st.checkbox("Include FR roadfee in Flat", value=s["includeFRinFlat"])

with st.sidebar.expander("Data: lanes import/export"):
    lanes_json = json.dumps(state["lanes"], indent=2)
    st.download_button("Download lanes.json", data=lanes_json, file_name="lanes.json")
    up = st.file_uploader("Upload lanes.json", type=["json"])
    if up is not None:
        try:
            loaded = json.load(up)
            assert isinstance(loaded, dict)
            state["lanes"] = loaded
            st.success("Lanes loaded.")
        except Exception as e:
            st.error(f"Failed to load: {e}")

# ------------------------ Tabs ------------------------
tabs = st.tabs(["Calculator", "LDM Scaler", "Weight Scaler", "Admin"])  # type: ignore

# ------------------------ Calculator ------------------------
with tabs[0]:
    st.subheader("Calculator")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        origin = st.text_input("Origin", value="SE")
    with colB:
        dest = st.text_input("Destination", value="FR")
    with colC:
        pallets = st.number_input("Pallets", value=10, step=1)
    with colD:
        weight = st.number_input("Weight (kg)", value=3500, step=50)

    colE, colF, colG, colH = st.columns(4)
    with colE:
        ldm_manual = st.text_input("LDM (optional)", value="")
    with colF:
        tariff = st.selectbox("Tariff", options=["auto","per_kg","per_pallet","per_flm"], index=0)
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
            # auto choose lowest
            candidates = [("per_kg", perkg["total"]), ("per_pallet", total_pal), ("per_flm", total_flm)]
            base = min(candidates, key=lambda t: t[1])

        st.markdown("### Base totals")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Per kg", f"€ {perkg['total']:.2f}")
        c2.metric("Per pallet", f"€ {total_pal:.2f}")
        c3.metric("Per FLM", f"€ {total_flm:.2f}")
        c4.metric("Chosen", f"{base[0]} → € {base[1]:.2f}")

        # fit from ladder
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
            s_local["fuelPct"] = st.number_input("Fuel % (decimal)", value=state["surcharges"]["fuelPct"], step=0.01, format="%.4f")
        with cB:
            s_local["marpolPct"] = st.number_input("MARPOL % (decimal)", value=state["surcharges"]["marpolPct"], step=0.01, format="%.4f")
        with cC:
            s_local["extraPct"] = st.number_input("Extra % (decimal)", value=state["surcharges"]["extraPct"], step=0.01, format="%.4f")
        with cD:
            rounding_step = st.number_input("Rounding step €", value=state["settings"]["roundingStep"], step=1.0)

        cE, cF, cG, cH = st.columns(4)
        with cE:
            s_local["perLdmEUR"] = st.number_input("Per LDM €", value=state["surcharges"]["perLdmEUR"], step=1.0)
        with cF:
            s_local["perKgEUR"] = st.number_input("Per kg €", value=state["surcharges"]["perKgEUR"], step=0.001, format="%.4f")
        with cG:
            s_local["flatEUR"] = st.number_input("Flat €", value=state["surcharges"]["flatEUR"], step=1.0)
        with cH:
            st.metric("Model (R²)", f"{chosen_model} (P {pow_fit['r2']:.3f} / E {exp_fit['r2']:.3f})")

        st.info("FR roadfee helper below adds into Flat if toggled.")
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            fr_class = int(st.selectbox("FR Class", options=[3,4], index=1 if state["surcharges"]["frClass"]==4 else 0))
        with r2:
            fr_km = st.number_input("FR toll km", value=state["surcharges"]["frKm"], step=10.0)
        with r3:
            fr_c3 = st.number_input("€/km Class 3", value=state["surcharges"]["frEurPerKmClass3"], step=0.01)
        with r4:
            fr_c4 = st.number_input("€/km Class 4", value=state["surcharges"]["frEurPerKmClass4"], step=0.01)
        with r5:
            fr_special = st.number_input("Special €", value=state["surcharges"]["frSpecialEUR"], step=1.0)
        include_fr = st.checkbox("Include FR in Flat", value=state["surcharges"]["includeFRinFlat"])

        per_km = fr_c3 if fr_class == 3 else fr_c4
        fr_fee = fr_km * per_km + fr_special

        percent_eur = base[1] * ( (s_local["fuelPct"] or 0) + (s_local["marpolPct"] or 0) + (s_local["extraPct"] or 0) )
        per_ldm_eur = ldm_val * (s_local["perLdmEUR"] or 0)
        per_kg_eur = chargeable_kg * (s_local["perKgEUR"] or 0)
        flat_eur = (s_local["flatEUR"] or 0) + (fr_fee if include_fr else 0)

        total_raw = base[1] + percent_eur + per_ldm_eur + per_kg_eur + flat_eur
        total_rounded = round_to_step(total_raw, rounding_step)

        st.markdown("### Totals")
        t1, t2 = st.columns(2)
        with t1:
            st.write(pd.DataFrame({
                "component": ["base", "percent", "per LDM", "per kg", "flat (incl FR)"],
                "EUR": [base[1], percent_eur, per_ldm_eur, per_kg_eur, flat_eur],
            }).style.format({"EUR": "€ {:.2f}"}))
        with t2:
            st.metric("Total (rounded)", f"€ {total_rounded:.2f}", help=f"raw € {total_raw:.2f}")

# ------------------------ LDM Scaler ------------------------
with tabs[1]:
    st.subheader("LDM Scaler – from full truck price or from fitted model")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mode = st.selectbox("Mode", options=["FROM_P13","FROM_FIT"], index=0)
    with col2:
        p13 = st.number_input("P13 (EUR)", value=2000.0, step=10.0)
    with col3:
        model_choice = st.selectbox("Model", options=["AUTO","POWER","EXP"], index=0)
    # Fit defaults based on lane SE->FR (you can change below)
    lane = state["lanes"].get("SE->FR")
    rows_for_fit = [
        dict(breakToKg=float(r.get("breakToKg")), ratePerKg=float(r.get("ratePerKg")))
        for r in lane.get("perKg", []) if clamp_number(r.get("breakToKg")) and clamp_number(r.get("ratePerKg"))
    ] if lane else []
    pow_fit = fit_power_model(rows_for_fit)
    exp_fit = fit_exp_model(rows_for_fit)

    with col4:
        b = st.number_input("POWER: b", value=float(0 if math.isnan(pow_fit.get("b", float("nan"))) else pow_fit.get("b")), step=0.01, format="%.5f")
    with col5:
        d = st.number_input("EXP: d", value=float(0 if math.isnan(exp_fit.get("d", float("nan"))) else exp_fit.get("d")), step=0.00001, format="%.6f")

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
                price = p13 * ((kgL / kg13) * math.exp(d * (kgL - kg13)))
        else:
            if chosen == "POWER":
                if abs(b + 1) < 1e-12:
                    price = float("nan")
                else:
                    # scale-free; anchor by setting FTL to p13
                    rawL = (kgL ** (b + 1)) / (b + 1)
                    raw13 = (kg13 ** (b + 1)) / (b + 1)
                    price = p13 * (rawL / raw13)
            else:
                if abs(d) < 1e-12:
                    rawL = kgL
                    raw13 = kg13
                else:
                    rawL = (math.exp(d * kgL) - 1) / d
                    raw13 = (math.exp(d * kg13) - 1) / d
                price = p13 * (rawL / raw13)
        ldm_rows.append({"LDM": L, "kg": kgL, "Price EUR": price})

    df_ldm = pd.DataFrame(ldm_rows)
    st.dataframe(df_ldm.style.format({"kg": "{:.0f}", "Price EUR": "€ {:.2f}"}), use_container_width=True)
    st.download_button("Download LDM table (CSV)", data=df_ldm.to_csv(index=False), file_name="ldm_prices.csv")

# ------------------------ Weight Scaler ------------------------
with tabs[2]:
    st.subheader("Weight Scaler – retier to new breaks")

    c1, c2, c3 = st.columns(3)
    with c1:
        lane_id = st.text_input("Lane ID", value="SE->FR")
    with c2:
        model_sel = st.selectbox("Model", options=["AUTO","POWER","EXP"], index=0)
    with c3:
        method = st.selectbox("Method", options=["INTEGRATED","POINT"], index=0)

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

        breaks_text = st.text_area("New breaks (kg), comma/space separated", value="500, 1000, 2000, 5000, 10000, 25160")
        new_breaks = []
        for tok in [t.strip() for t in breaks_text.replace(";", ",").replace("\n", ",").split(",")]:
            n = clamp_number(tok)
            if n and n > 0:
                new_breaks.append(float(n))
        new_breaks = sorted(set(new_breaks))

        def rate_at(kg: float) -> float:
            if chosen == "EXP":
                A, d = exp_fit["A"], exp_fit["d"]
                return (A or 0) * math.exp((d or 0) * kg)
            else:
                A, b = pow_fit["A"], pow_fit["b"]
                return (A or 0) * (kg ** (b or 0))

        def integral_to(kg: float) -> float:
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
                    return (A or 0) * math.log(kg)
                return (A / (b + 1)) * (kg ** (b + 1))

        out_rows = []
        start = 0.0
        for brk in new_breaks:
            width = brk - start
            r_break = rate_at(brk)
            if method == "INTEGRATED":
                total = integral_to(brk) - integral_to(start)
            else:
                total = r_break * width  # point approximation
            avg_eur_per_kg = total / width if width > 0 else float("nan")
            out_rows.append({
                "BreakToKg": brk,
                "BandStartKg": start,
                "WidthKg": width,
                "Rate@break €/kg": r_break,
                "Band Total €": total,
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
            "Avg €/kg": "{:.4f}",
        }), use_container_width=True)
        st.download_button("Download new ladder (CSV)", data=df_out.to_csv(index=False), file_name=f"{lane_id.replace('>','-')}-retiered.csv")

# ------------------------ Admin ------------------------
with tabs[3]:
    st.subheader("Admin – lanes & settings")
    st.info("Use the JSON export/import in the sidebar for backups. Data persists in your browser session.")

    lane_ids = sorted(state["lanes"].keys())
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        selected_lane = st.selectbox("Select lane", options=lane_ids, index=0 if lane_ids else None)
    with col2:
        new_from = st.text_input("New lane: origin", value="SE")
    with col3:
        new_to = st.text_input("New lane: dest", value="FR")

    if st.button("Add lane"):
        lane_id = f"{new_from}->{new_to}"
        if lane_id in state["lanes"]:
            st.warning("Lane already exists.")
        else:
            state["lanes"][lane_id] = {"perKg": [], "perPallet": {"rate": 0, "min": 0}, "perFLM": {"rate": 0, "min": 0}}
            st.success(f"Added {lane_id}")

    if selected_lane:
        lane = state["lanes"][selected_lane]
        st.markdown(f"### Edit lane {selected_lane}")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Per-kg ladder (breaks ascending)")
            df = pd.DataFrame(lane.get("perKg", []))
            df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"edit_{selected_lane}")
            # Save back
            lane["perKg"] = [
                {"breakToKg": clamp_number(r.get("breakToKg")) or 0,
                 "ratePerKg": clamp_number(r.get("ratePerKg")) or 0,
                 "minEUR": clamp_number(r.get("minEUR")) or 0}
                for _, r in df.fillna(0).iterrows()
            ]
        with c2:
            st.write("Per pallet")
            lane.setdefault("perPallet", {})
            lane["perPallet"]["rate"] = st.number_input("€/pallet", value=float(lane["perPallet"].get("rate", 0)), step=1.0)
            lane["perPallet"]["min"] = st.number_input("Min € (pallet)", value=float(lane["perPallet"].get("min", 0)), step=1.0)
            st.write("Per FLM")
            lane.setdefault("perFLM", {})
            lane["perFLM"]["rate"] = st.number_input("€/FLM", value=float(lane["perFLM"].get("rate", 0)), step=1.0)
            lane["perFLM"]["min"] = st.number_input("Min € (FLM)", value=float(lane["perFLM"].get("min", 0)), step=1.0)

    st.divider()
    if st.button("Reset to defaults"):
        st.session_state.app_state = DEFAULT_STATE
        st.experimental_rerun()


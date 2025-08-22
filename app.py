# -*- coding: utf-8 -*-
# EcoSwitch — Démo v11.4r (UI Streamlit) — patch Streamlit Cloud

import os, io, sys, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EcoSwitch — Démo v11.4r", page_icon="🌿", layout="wide")

# Fonts + Theme CSS
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# 🔧 TROUVE LE BON "PROJECT ROOT" (que app.py soit à la racine ou dans ui/)
HERE = Path(__file__).parent

def _find_project_root() -> Path:
    cands = [HERE, HERE.parent]
    for c in cands:
        if (c / "requirements.txt").exists() or (c / "modules").exists() or (c / "data").exists():
            return c
    return HERE

PROJECT = _find_project_root()

# Thème + logo (facultatif)
ASSETS = PROJECT / "assets"
THEME_CSS = ASSETS / "theme.css"
if THEME_CSS.exists():
    st.markdown(f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
if (ASSETS / "logo.svg").exists():
    st.image(str(ASSETS / "logo.svg"), width=28)

# 📁 Dossiers de données (toujours créés dans le repo)
DATA_DIR = PROJECT / "data"
METEO_DIR = DATA_DIR / "meteo"
TARIFS_DIR = DATA_DIR / "tarifs"
for d in (METEO_DIR, TARIFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ✅ Import du moteur : modules/ prioritaire, sinon core.py (secours)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
try:
    from modules import Audit, simulate, load_meteo, log_run
except Exception:
    try:
        from core import Audit, simulate, load_meteo, log_run
    except Exception as e:
        st.error("Moteur introuvable. Crée soit `modules/__init__.py` (qui fait `from core import *`), soit place `core.py` à la racine.")
        st.stop()

# (le reste de ton fichier inchangé à partir d’ici)


# Header
colH1, colH2 = st.columns([3, 1])
with colH1:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.75rem">
          <img src="app://local/ui/assets/logo.svg" width="28" height="28" alt="EcoSwitch logo"/>
          <div>
            <h1 style="margin:0">EcoSwitch — Démo PAC & Hybride</h1>
            <div style="opacity:.75">Alignée landing • typographie Inter • palette verts/bleus</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colH2:
    st.markdown(
        "<div style='display:flex;justify-content:flex-end;align-items:center;height:100%'>"
        "<a href='#' target='_blank' class='eos-cta'>Retour landing</a>"
        "</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    _dark = st.toggle("Thème sombre (beta)", value=False)
if _dark:
    st.markdown(
        """
        <style>
          :root{
            --bg:#0b1220; --surface:#0f172a; --ink:#e5e7eb; --muted:#94a3b8;
            --eco:#22c55e; --blue:#3b82f6; --ring:rgba(34,197,94,.25)
          }
          html,body{background:var(--bg)!important;color:var(--ink)}
          .stApp header, .st-emotion-cache-18ni7ap{background:var(--bg)!important}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.info("Démo indicative • ISO/EN simplifié • Incertitude ±15% • Wholesale (≈0,07 €/kWh) ≠ Retail (≈0,16–0,21 €/kWh).", icon="ℹ️")
_badge_sources()

# Sidebar inputs
with st.sidebar:
    st.header("Profil logement")
    zone = st.selectbox("Zone climatique", ["H1", "H2", "H3"], index=1)
    surface = st.slider("Surface (m²)", 40, 250, 100, 5)
    isolation = st.selectbox("Isolation", ["faible", "moyenne", "bonne"], index=1)
    emetteurs = st.selectbox("Émetteurs", ["acier", "alu", "plancher"], index=0)
    t_depart = st.slider("Température de départ (°C)", 30, 70, 45, 1)
    chaudiere_age = st.slider("Âge chaudière (ans)", 0, 30, 10)
    pac_age = st.slider("Âge PAC (ans)", 0, 20, 0)
    climate_shift = st.slider("Sensibilité climat 2050 (+°C)", 0.0, 3.0, 0.0, 0.1)

    st.header("Tarifs énergie")
    pricing_mode = st.radio("Mode tarifs", ["wholesale", "retail"], horizontal=True, index=1)
    retail_profile = st.selectbox("Profil retail", ["Base", "HPHC", "Tempo"], index=1)
    st.caption("Utilise **retail** pour simuler une facture ménage (Base/HPHC/Tempo).")

    st.divider()
    st.header("Aides & ROI (optionnel)")
    apply_aides_to = st.selectbox("Appliquer aides à", ["Hybride", "PAC", "Les deux", "Aucune"], index=0)
    mpr = st.slider("MaPrimeRénov’ (€/logement)", 0, 5000, 3000, 500)
    cee = st.slider("Prime CEE (€/logement)", 0, 2000, 1000, 100)
    capex_pac = st.number_input("CAPEX PAC (€)", min_value=0, value=12000, step=500)
    capex_hybr = st.number_input("CAPEX Hybride (€)", min_value=0, value=15000, step=500)
    sans_aides = st.checkbox("Afficher ROI **sans** aides", value=False)

# Importers
with st.expander("Importer des données (EPW / Tarifs)"):
    st.caption("Charge EPW/ZIP OneBuilding et CSV pour tarifs (Ember EUR/MWh ou TRV base/hp/hc).")
    col_epw, col_tar = st.columns(2)
    with col_epw:
        st.subheader("EPW (météo)")
        zone_choice = st.selectbox("Associer à la zone", ["H1 (Paris-Orly)", "H2 (Lyon-Bron)", "H3 (Marseille-Marignane)"], index=1, key="epw_zone_sel")
        to_zone = {"H1 (Paris-Orly)": "H1", "H2 (Lyon-Bron)": "H2", "H3 (Marseille-Marignane)": "H3"}[zone_choice]
        target_name = {"H1": "paris_orly.epw", "H2": "lyon_bron.epw", "H3": "marseille_marignane.epw"}[to_zone]

        up_epw = st.file_uploader("Fichier EPW ou ZIP (.epw/.zip)", type=["epw", "zip"], key="epw_uploader")
        if up_epw:
            try:
                content = up_epw.read()
                if up_epw.name.lower().endswith(".zip"):
                    z = zipfile.ZipFile(io.BytesIO(content))
                    members = [n for n in z.namelist() if n.lower().endswith(".epw")]
                    if not members:
                        raise ValueError("ZIP sans fichier .epw")
                    content = z.read(members[0])
                out = METEO_DIR / target_name
                out.write_bytes(content)
                st.success(f"EPW importé pour {to_zone} → {out.name} (sha256:{hashlib.sha256(content).hexdigest()[:12]})")
                if st.button("Recharger l'app", key="reload_epw"):
                    st.rerun()
            except Exception as e:
                st.error(f"Échec import EPW: {e}")

        quick = st.selectbox("Source rapide EPW (ZIP OneBuilding)", ["—", "Paris-Orly", "Lyon-Bron", "Marseille-Marignane"], index=1)
        urls = {
            "Paris-Orly": "https://climate.onebuilding.org/WMO_Region_6_Europe/FRA_Ile-de-France_Paris_Paris-Orly_AP_FRA_EPWTMYx.2009-2023.zip",
            "Lyon-Bron": "https://climate.onebuilding.org/WMO_Region_6_Europe/FRA_Rhone-Alpes_Lyon_Bron_AP_FRA_EPWTMYx.2009-2023.zip",
            "Marseille-Marignane": "https://climate.onebuilding.org/WMO_Region_6_Europe/FRA_Provence-Alpes-Cote_dAzur_Marseille_Marignane_AP_FRA_EPWTMYx.2009-2023.zip",
        }
        url_box = st.text_input("URL EPW/ZIP", value=urls.get(quick, ""))
        if st.button("Télécharger EPW depuis URL", key="dl_epw_btn"):
            try:
                u = url_box.strip()
                if not u:
                    raise ValueError("URL vide")
                resp = requests.get(u, timeout=35)
                resp.raise_for_status()
                content = resp.content
                try:
                    if u.lower().endswith(".zip"):
                        z = zipfile.ZipFile(io.BytesIO(content))
                        members = [n for n in z.namelist() if n.lower().endswith(".epw")]
                        if not members:
                            raise ValueError("ZIP sans .epw")
                        content = z.read(members[0])
                except zipfile.BadZipFile:
                    pass
                out = METEO_DIR / target_name
                out.write_bytes(content)
                st.success(f"EPW téléchargé → {out.name} (sha256:{hashlib.sha256(content).hexdigest()[:12]})")
            except Exception as e:
                st.error(f"Échec téléchargement EPW: {e}")

    with col_tar:
        st.subheader("Tarifs (CSV)")
        st.caption("Wholesale: `elec_spot_2025.csv` (colonne price en EUR/MWh). Retail TRV: `retail_trv.csv` (base,hp,hc en €/kWh TTC).")

        up_wh = st.file_uploader("Wholesale CSV (EUR/MWh)", type=["csv"], key="wh_csv")
        if up_wh:
            try:
                p = TARIFS_DIR / "elec_spot_2025.csv"
                p.write_bytes(up_wh.read())
                st.success(f"Wholesale mis à jour → {p.name} (sha256:{_sha8(p)})")
            except Exception as e:
                st.error(f"Échec import wholesale: {e}")

        up_rt = st.file_uploader("Retail TRV CSV (base,hp,hc)", type=["csv"], key="rt_csv")
        if up_rt:
            try:
                p = TARIFS_DIR / "retail_trv.csv"
                p.write_bytes(up_rt.read())
                st.success(f"Retail TRV mis à jour → {p.name} (sha256:{_sha8(p)})")
            except Exception as e:
                st.error(f"Échec import retail: {e}")

        c1, c2 = st.columns(2)
        if c1.button("Préremplir URL ZIP Ember EU"):
            st.session_state["wh_url"] = "https://ember-energy.org/media/site/edgenl0l/european-wholesale-electricity-price-data.zip"
        wh_url = st.text_input("URL Wholesale CSV/ZIP (EUR/MWh)", value=st.session_state.get("wh_url", ""))
        if c1.button("Télécharger Wholesale"):
            try:
                u = wh_url.strip()
                if not u:
                    raise ValueError("URL vide")
                r = requests.get(u, timeout=35)
                r.raise_for_status()
                content = r.content
                try:
                    if u.lower().endswith(".zip"):
                        z = zipfile.ZipFile(io.BytesIO(content))
                        names = z.namelist()
                        cand = [n for n in names if n.lower().endswith(".csv")]
                        fr = [n for n in cand if n.lower().endswith("fr.csv") or "_fr.csv" in n.lower() or "/fr.csv" in n.lower()]
                        pick = fr[0] if fr else (cand[0] if cand else None)
                        if not pick:
                            raise ValueError("ZIP sans CSV exploitable")
                        content = z.read(pick)
                except zipfile.BadZipFile:
                    pass
                p = TARIFS_DIR / "elec_spot_2025.csv"
                p.write_bytes(content)
                st.success(f"Wholesale téléchargé → {p.name} (sha256:{_sha8(p)})")
            except Exception as e:
                st.error(f"Échec téléchargement wholesale: {e}")

        rt_url = st.text_input("URL Retail TRV CSV (base,hp,hc)")
        if c2.button("Télécharger Retail TRV"):
            try:
                u = rt_url.strip()
                if not u:
                    raise ValueError("URL vide")
                r = requests.get(u, timeout=20)
                r.raise_for_status()
                p = TARIFS_DIR / "retail_trv.csv"
                p.write_bytes(r.content)
                st.success(f"Retail TRV téléchargé → {p.name} (sha256:{_sha8(p)})")
            except Exception as e:
                st.error(f"Échec téléchargement retail: {e}")

        st.divider()
        gen_trv = st.button("Générer TRV par défaut (base=0.1952 / hp=0.2081 / hc=0.1600)")
        if gen_trv:
            try:
                (TARIFS_DIR / "retail_trv.csv").write_text("base,hp,hc\n0.1952,0.2081,0.1600\n", encoding="utf-8")
                st.success("TRV créé.")
            except Exception as e:
                st.error(f"Échec génération TRV: {e}")

        HC_start = st.number_input("Heure début Heures Creuses (Tempo/HPHC)", min_value=0, max_value=23, value=22, step=1)
        HC_hours = st.number_input("Durée HC (heures)", min_value=1, max_value=12, value=8, step=1)
        gen_tempo = st.button("Générer calendrier Tempo 2025 (horaire)")
        if gen_tempo:
            try:
                start = pd.Timestamp("2025-01-01 00:00:00")
                hours = pd.date_range(start, periods=8760, freq="H")
                rng = np.random.default_rng(42)
                days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
                colors = []
                for d in days:
                    if d.month in (12, 1, 2):
                        pprob = [0.70, 0.20, 0.10]
                    else:
                        pprob = [0.80, 0.20, 0.00]
                    colors.append(rng.choice(["bleu", "blanc", "rouge"], p=pprob))
                day_color = pd.Series(colors, index=days)
                df = pd.DataFrame({"datetime": hours})
                df["date"] = df["datetime"].dt.date
                df["hour"] = df["datetime"].dt.hour
                df["color"] = df["date"].map(day_color.astype(str))
                hc_end = (HC_start + HC_hours) % 24
                if HC_start < hc_end:
                    mask_hc = (df["hour"] >= HC_start) & (df["hour"] < hc_end)
                else:
                    mask_hc = (df["hour"] >= HC_start) | (df["hour"] < hc_end)
                df["period"] = np.where(mask_hc, "HC", "HP")
                price_map = {
                    ("bleu", "HP"): 0.19, ("bleu", "HC"): 0.15,
                    ("blanc", "HP"): 0.23, ("blanc", "HC"): 0.18,
                    ("rouge", "HP"): 0.55, ("rouge", "HC"): 0.40,
                }
                df["price_eur_per_kwh"] = [price_map[(c, p)] for c, p in zip(df["color"], df["period"])]
                out = TARIFS_DIR / "tempo_2025.csv"
                df[["datetime", "color", "period", "price_eur_per_kwh"]].to_csv(out, index=False)
                st.success(f"Tempo 2025 généré → {out.name}")
            except Exception as e:
                st.error(f"Échec génération Tempo: {e}")

# Build Audit
audit = Audit(
    surface_m2=surface,
    zone=zone,
    isolation=isolation,
    emetteurs=emetteurs,
    t_depart=t_depart,
    chaudiere_age=chaudiere_age,
    pac_age_years=pac_age,
)

# Meteo
try:
    dfm = load_meteo(zone, climate_shift)
except TypeError:
    dfm = load_meteo(zone)

meteo_ok = (len(dfm) == 8760) and dfm["temp_ext"].between(-30, 45).all()
st.caption(f"Météo: {'✅ EPW' if meteo_ok else '⚠️ Fallback'} • Tmin={dfm['temp_ext'].min():.1f}°C • Tmax={dfm['temp_ext'].max():.1f}°C • len={len(dfm)}")

# Simulate
sim_kwargs = dict(pricing_mode=pricing_mode, retail_profile=retail_profile, climate_shift=climate_shift)
try:
    res = simulate(audit, **sim_kwargs)
except TypeError:
    try:
        res = simulate(audit, climate_shift=climate_shift)
    except TypeError:
        res = simulate(audit)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Coût Gaz réf.", f"{res.get('cost_ref', 0):,.0f} €".replace(",", " "))
k2.metric("Coût PAC", f"{res.get('cost_pac', 0):,.0f} €".replace(",", " "))
k3.metric("Coût Hybride", f"{res.get('cost_hybr', 0):,.0f} €".replace(",", " "))
k4.metric("% heures PAC (hybride)", f"{int(res.get('pac_share', 0)*100)} %")
st.caption("Astuce : si **% heures PAC = 100%**, l’hybride utilise la PAC toute l’année ⇒ **Coût Hybride = Coût PAC**.")

# Charts
df_bar = pd.DataFrame({
    "Scenario": ["Gaz", "PAC", "Hybride"],
    "Coût (€)": [res.get("cost_ref", 0), res.get("cost_pac", 0), res.get("cost_hybr", 0)],
    "CO2 (t)": [res.get("co2_ref", 0)/1000, res.get("co2_pac", 0)/1000, res.get("co2_hybr", 0)/1000],
})
st.subheader("Comparaison coûts & CO₂ (incertitude ±15%)")
fig_cost = px.bar(df_bar, x="Scenario", y="Coût (€)", title="Coûts annuels par scénario")
fig_cost.update_traces(error_y=dict(type="percent", value=15, visible=True))
st.plotly_chart(fig_cost, use_container_width=True)
fig_co2 = px.bar(df_bar, x="Scenario", y="CO2 (t)", title="Émissions CO₂ annuelles par scénario")
fig_co2.update_traces(error_y=dict(type="percent", value=15, visible=True))
st.plotly_chart(fig_co2, use_container_width=True)

# Daily T
df_daily = dfm.copy()
df_daily["date"] = pd.to_datetime(df_daily["timestamp"]).dt.date
df_daily = df_daily.groupby("date", as_index=False)["temp_ext"].mean()
fig_T = px.line(df_daily, x="date", y="temp_ext", title="Température extérieure — moyenne journalière")
fig_T.update_traces(error_y=dict(type="percent", value=15, visible=True))
st.plotly_chart(fig_T, use_container_width=True)

# ROI
def _safe_roi(capex, annual_save):
    if annual_save <= 0:
        return None
    return capex / annual_save

save_pac = max(res.get("cost_ref", 0) - res.get("cost_pac", 0), 0)
save_hyb = max(res.get("cost_ref", 0) - res.get("cost_hybr", 0), 0)

aides_pac = 0
aides_hyb = 0
if "PAC" in ("PAC",) or ("Les deux" in ("Les deux",) and False):
    pass  # placeholder
if apply_aides_to in ("PAC", "Les deux"):
    aides_pac = (0 if sans_aides else (mpr + cee))
if apply_aides_to in ("Hybride", "Les deux"):
    aides_hyb = (0 if sans_aides else (mpr + cee))

roi_pac_avant = _safe_roi(capex_pac, save_pac)
roi_hyb_avant = _safe_roi(capex_hybr, save_hyb)
roi_pac_apres = _safe_roi(max(capex_pac - aides_pac, 0), save_pac)
roi_hyb_apres = _safe_roi(max(capex_hybr - aides_hyb, 0), save_hyb)

st.subheader("ROI — avant / après aides")
c1, c2 = st.columns(2)
c1.metric("PAC — ROI avant aides", "—" if roi_pac_avant is None else f"{roi_pac_avant:.1f} ans")
c1.metric("PAC — ROI après aides", "—" if roi_pac_apres is None else f"{roi_pac_apres:.1f} ans")
c2.metric("Hybride — ROI avant aides", "—" if roi_hyb_avant is None else f"{roi_hyb_avant:.1f} ans")
c2.metric("Hybride — ROI après aides", "—" if roi_hyb_apres is None else f"{roi_hyb_apres:.1f} ans")
st.caption("Aides indicatives (MPR/CEE). Plafonds cumulatifs typiques ≈ 20 k€ / logement. Mode **Sans aides** disponible.")

# KPI demo (H1/H2/H3) with same params
with st.expander("KPI démo (H1/H2/H3 — mêmes paramètres)"):
    vals = []
    for z in ["H1", "H2", "H3"]:
        a2 = Audit(surface_m2=surface, zone=z, isolation=isolation, emetteurs=emetteurs,
                   t_depart=t_depart, chaudiere_age=chaudiere_age, pac_age_years=pac_age)
        try:
            r2 = simulate(a2, **sim_kwargs)
        except TypeError:
            try:
                r2 = simulate(a2, climate_shift=climate_shift)
            except TypeError:
                r2 = simulate(a2)
        vals.append(r2)
    med_save = np.median([max(v.get("cost_ref", 0) - v.get("cost_hybr", 0), 0) for v in vals])
    med_share = np.median([v.get("pac_share", 0) for v in vals])
    kA, kB = st.columns(2)
    kA.metric("Économie médiane (Hybride vs Gaz)", f"{med_save:,.0f} €".replace(",", " "))
    kB.metric("% médian d'heures PAC (Hybride)", f"{int(med_share*100)} %")

# Export JSON
def _hash_or_none(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if _file_exists(p) else None

export = {
    "audit": {
        "surface_m2": surface, "zone": zone, "isolation": isolation, "emetteurs": emetteurs,
        "t_depart": t_depart, "chaudiere_age": chaudiere_age, "pac_age_years": pac_age
    },
    "params": sim_kwargs | {
        "capex_pac": capex_pac, "capex_hybr": capex_hybr,
        "mpr": mpr, "cee": cee, "apply_aides_to": apply_aides_to, "sans_aides": sans_aides
    },
    "results": res,
    "hashes": {
        "epw_H1": _hash_or_none(METEO_DIR / "paris_orly.epw"),
        "epw_H2": _hash_or_none(METEO_DIR / "lyon_bron.epw"),
        "epw_H3": _hash_or_none(METEO_DIR / "marseille_marignane.epw"),
        "wholesale": _hash_or_none(TARIFS_DIR / "elec_spot_2025.csv"),
        "retail_trv": _hash_or_none(TARIFS_DIR / "retail_trv.csv"),
        "tempo_2025": _hash_or_none(TARIFS_DIR / "tempo_2025.csv"),
    }
}
st.download_button(
    "📥 Exporter les résultats (JSON)",
    data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="ecoswitch_v11_4r_export.json",
    mime="application/json",
)

# Logging
try:
    log_run(audit, sim_kwargs, res)
except Exception:
    pass

st.caption("© EcoSwitch — v11.4r • Wholesale ≠ Retail • ISO/EN simplifié • Incertitude ±15%.")

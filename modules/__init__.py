# modules/__init__.py
from __future__ import annotations

import os, json, hashlib
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

# pvlib est optionnel (EPW réel si dispo, sinon fallback synthétique)
try:
    import pvlib  # type: ignore
except Exception:  # pragma: no cover
    pvlib = None

# --------------------------------------------------------------------------------------
# Chemins & dossiers (créés automatiquement)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
METEO_DIR = DATA_DIR / "meteo"
TARIFS_DIR = DATA_DIR / "tarifs"
LOGS_DIR = ROOT / "logs"

for _d in (DATA_DIR, METEO_DIR, TARIFS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Mode strict : si TRUE et EPW invalide → on lève une erreur (sinon fallback)
STRICT_MODE = os.getenv("STRICT_MODE", "false").lower() == "true"

# --------------------------------------------------------------------------------------
# Modèle d'audit & auxiliaires physiques
# --------------------------------------------------------------------------------------
@dataclass
class Audit:
    surface_m2: float = 100.0
    zone: Literal["H1", "H2", "H3"] = "H2"
    isolation: Literal["faible", "moyenne", "bonne"] = "moyenne"
    emetteurs: Literal["acier", "alu", "plancher"] = "acier"
    t_depart: float = 45.0
    chaudiere_age: int = 10
    pac_age_years: int = 0
    temp_int: float = 20.0  # °C

def UA_from_audit(a: Audit) -> float:
    """UA global (W/K) simplifié, calibré par zone et émetteurs."""
    base_u = {"faible": 2.5, "moyenne": 1.8, "bonne": 1.2}[a.isolation]
    em_mult = {"acier": 1.00, "alu": 0.95, "plancher": 0.85}[a.emetteurs]
    UA = a.surface_m2 * base_u * 0.5 * em_mult  # 0.5 ~ ratio enveloppe/Surf.
    calib = {"H1": 1.10, "H2": 1.00, "H3": 0.90}.get(a.zone, 1.0)
    return UA * calib

def cop_pac(T_ext: np.ndarray, t_depart: float, isolation: str, age: int = 0) -> np.ndarray:
    """Courbe COP ~ quad. + pénalités TD & isolation & vieillissement."""
    T = np.asarray(T_ext, dtype=float)
    cop = 3.2 + 0.06*T - 0.001*(T**2)                # loi simple
    cop -= 0.01 * max(0.0, t_depart - 35.0)          # -1%/K au-dessus de 35°C
    cop *= {"faible": 0.95, "moyenne": 1.0, "bonne": 1.03}.get(isolation, 1.0)
    cop *= max(0.0, 1.0 - 0.005*age)                 # -0,5%/an
    return np.clip(cop, 1.1, 6.5)

def eta_chaudiere(t_depart: float, age: int = 0) -> float:
    """Rendement chaudière simple avec effet TD & âge."""
    eta = 0.94 - 0.001*max(0.0, t_depart - 35.0)
    eta *= max(0.0, 1.0 - 0.005*age)
    return float(np.clip(eta, 0.85, 0.98))

# --------------------------------------------------------------------------------------
# Météo (EPW si dispo, sinon fallback synthétique)
# --------------------------------------------------------------------------------------
def _epw_path(zone: str) -> Path:
    mapping = {"H1": "paris_orly.epw", "H2": "lyon_bron.epw", "H3": "marseille_marignane.epw"}
    return METEO_DIR / mapping.get(zone, "lyon_bron.epw")

def load_meteo(zone: str, climate_shift: float = 0.0) -> pd.DataFrame:
    """Retourne DataFrame 8760h avec colonnes: timestamp, temp_ext (°C)."""
    epw = _epw_path(zone)
    if pvlib is not None and epw.exists():
        try:
            weather, _meta = pvlib.iotools.read_epw(epw.as_posix())
            idx = weather.index
            if getattr(idx, "tz", None) is not None:  # supprime timezone si présente
                idx = idx.tz_localize(None)
            temp = pd.to_numeric(weather["temp_air"], errors="coerce")
            temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
            df = pd.DataFrame({"timestamp": idx[:len(temp)], "temp_ext": temp.values + climate_shift})
            if len(df) == 8760 and df["temp_ext"].between(-30, 45).all():
                return df.reset_index(drop=True)
        except Exception as e:
            if STRICT_MODE:
                raise RuntimeError(f"EPW invalide ({epw}): {e}")

    # --- fallback synthétique (saison + journalier + bruit)
    n = 8760
    t = pd.date_range("2025-01-01", periods=n, freq="H")
    means = {"H1": 8.0, "H2": 10.5, "H3": 12.0}
    amp   = {"H1": 16.0, "H2": 14.0, "H3": 12.0}
    seasonal = means.get(zone, 10.0) + (amp.get(zone, 14.0) / 2.0) * np.sin(2*np.pi*(np.arange(n)/n) - np.pi/2)
    daily    = 2.0 * np.sin(2*np.pi*(np.arange(n)/24.0))
    noise    = np.random.normal(0, 3.0, n)
    temp_ext = seasonal + daily + noise + climate_shift
    df = pd.DataFrame({"timestamp": t, "temp_ext": temp_ext})
    if STRICT_MODE:
        raise RuntimeError("STRICT_MODE: EPW requis, fallback désactivé.")
    return df

# --------------------------------------------------------------------------------------
# Tarifs (wholesale Ember ou retail TRV/Tempo)
# --------------------------------------------------------------------------------------
def _ember_wholesale(n_hours: int) -> np.ndarray:
    """Lit elec_spot_2025.csv (EUR/MWh) → €/kWh, sinon génère un profil réaliste."""
    path = TARIFS_DIR / "elec_spot_2025.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if "price" in df.columns:
                arr = pd.to_numeric(df["price"], errors="coerce").values  # EUR/MWh
                arr = arr[~np.isnan(arr)]
                elec = (arr[:n_hours].astype(float) / 1000.0)             # €/kWh
                if len(elec) < n_hours:
                    elec = np.resize(elec, n_hours)
                if 0.0 < float(np.nanmean(elec)) < 0.40:
                    return elec
        except Exception:
            pass
    # synthèse
    h = np.arange(n_hours)
    base = 0.066
    seasonal = base + 0.02*np.sin(2*np.pi*(h/n_hours))
    daily    = base + 0.02*np.sin(2*np.pi*(h/24.0))
    noise    = np.random.normal(0, 0.005, n_hours)
    elec = 0.5*(seasonal + daily) + noise
    return np.clip(elec, 0.03, 0.30)

def _retail_from_csv() -> Optional[Tuple[float, float, float]]:
    """Lit retail_trv.csv → (base, hp, hc) en €/kWh si dispo."""
    path = TARIFS_DIR / "retail_trv.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict()
            return float(row.get("base", np.nan)), float(row.get("hp", np.nan)), float(row.get("hc", np.nan))
        except Exception:
            return None
    return None

def _retail_vector(
    n_hours: int,
    profile: Literal["Base", "HPHC", "Tempo"] = "Base",
    base: float = 0.1952, hp: float = 0.2081, hc: float = 0.1600
) -> np.ndarray:
    """Construit un vecteur horaire €/kWh pour Base/HPHC/Tempo (synthetic)."""
    h = np.arange(n_hours)
    if profile == "Base":
        return np.full(n_hours, base, dtype=float)
    if profile == "HPHC":
        hour = h % 24
        return np.where(((hour >= 22) | (hour < 7)), hc, hp).astype(float)
    # Tempo synthétique (si pas d'override CSV)
    days = np.array(["bleu"]*274 + ["blanc"]*73 + ["rouge"]*18)
    rng = np.random.default_rng(42)
    rng.shuffle(days)
    day_index = (h // 24).astype(int)
    hour = h % 24
    vec = np.zeros(n_hours, dtype=float)
    for i in range(n_hours):
        d = days[day_index[i]]
        if d == "bleu":
            vec[i] = hc if (hour[i] < 7 or hour[i] >= 22) else base
        elif d == "blanc":
            v_hp = max(hp, base*1.10); v_hc = max(hc, base*0.95)
            vec[i] = v_hc if (hour[i] < 7 or hour[i] >= 22) else v_hp
        else:
            v_hp = max(0.55, hp*2.5); v_hc = max(0.40, hc*2.0)
            vec[i] = v_hc if (hour[i] < 7 or hour[i] >= 22) else v_hp
    return vec

def _load_tempo_csv(tarifs_dir: Path) -> Optional[np.ndarray]:
    f = tarifs_dir / "tempo_2025.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if "price_eur_per_kwh" not in df.columns:
        return None
    vals = pd.to_numeric(df["price_eur_per_kwh"], errors="coerce").to_numpy()
    if len(vals) == 8760:
        return vals.astype(float)
    return np.resize(vals.astype(float), 8760)

def load_tarifs(
    n_hours: int,
    pricing_mode: Literal["wholesale", "retail"] = "wholesale",
    retail_profile: Literal["Base", "HPHC", "Tempo"] = "Base"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Retourne (elec_vec, gaz_vec, info)."""
    info: Dict[str, Any] = {}
    if pricing_mode == "retail":
        csv_vals = _retail_from_csv()
        if csv_vals is not None:
            base, hp, hc = csv_vals
            info["retail_source"] = "retail_trv.csv"
        else:
            base, hp, hc = 0.1952, 0.2081, 0.1600
            info["retail_source"] = "ui_defaults"
        elec = _retail_vector(n_hours, retail_profile, base, hp, hc)
        if retail_profile == "Tempo":  # override si CSV horaire dispo
            try:
                tempo_vec = _load_tempo_csv(TARIFS_DIR)
                if tempo_vec is not None:
                    elec = tempo_vec
                    info["retail_source"] = "tempo_2025.csv"
            except Exception:
                pass
    else:
        elec = _ember_wholesale(n_hours)

    gaz = np.full(n_hours, 0.11, dtype=float)  # €/kWh
    info["elec_mean"] = float(np.nanmean(elec))
    info["pricing_mode"] = pricing_mode
    info["retail_profile"] = retail_profile if pricing_mode == "retail" else None
    return elec, gaz, info

def _co2_vectors(n_hours: int) -> Tuple[np.ndarray, np.ndarray]:
    """Facteurs CO₂ (kg/kWh) simples FR mix."""
    return np.full(n_hours, 0.05), np.full(n_hours, 0.227)

# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
def simulate(
    audit: Audit,
    pricing_mode: Literal["wholesale", "retail"] = "wholesale",
    retail_profile: Literal["Base", "HPHC", "Tempo"] = "Base",
    climate_shift: float = 0.0
) -> Dict[str, Any]:
    n = 8760
    dfm = load_meteo(audit.zone, climate_shift)
    if len(dfm) != n:
        raise RuntimeError("Meteo invalide (len != 8760)")

    T = dfm["temp_ext"].to_numpy()
    price_elec, price_gaz, price_info = load_tarifs(n, pricing_mode, retail_profile)

    UA = UA_from_audit(audit)
    load_kw = np.maximum(0.0, audit.temp_int - T) * UA / 1000.0  # kW utiles
    cop = cop_pac(T, audit.t_depart, audit.isolation, audit.pac_age_years)
    eta = eta_chaudiere(audit.t_depart, audit.chaudiere_age)

    # Gaz (réf)
    e_gaz_ref = load_kw / eta                    # kWh PCS
    cost_ref = float(np.sum(e_gaz_ref * price_gaz))

    # PAC seule
    e_elec_pac = load_kw / cop
    cost_pac = float(np.sum(e_elec_pac * price_elec))

    # Hybride : dispatch heure par heure selon coût unitaire
    cost_u_pac = price_elec / np.maximum(cop, 1e-6)
    cost_u_gaz = price_gaz / eta
    use_pac = cost_u_pac < cost_u_gaz            # bool vector
    load_pac = load_kw * use_pac
    load_gaz = load_kw - load_pac
    e_elec_h = load_pac / cop
    e_gaz_h  = load_gaz / eta
    cost_hybr = float(np.sum(e_elec_h * price_elec) + np.sum(e_gaz_h * price_gaz))
    pac_share = float(np.mean(use_pac))

    # CO2
    co2_elec, co2_gaz = _co2_vectors(n)
    co2_ref  = float(np.sum(e_gaz_ref * co2_gaz))
    co2_pac  = float(np.sum(e_elec_pac * co2_elec))
    co2_hybr = float(np.sum(e_elec_h * co2_elec) + np.sum(e_gaz_h * co2_gaz))

    return {
        "cost_ref": cost_ref,
        "cost_pac": cost_pac,
        "cost_hybr": cost_hybr,
        "co2_ref": co2_ref,
        "co2_pac": co2_pac,
        "co2_hybr": co2_hybr,
        "pac_share": pac_share,
        "UA": UA,
        "price_info": price_info,
        "meteo_source": "epw" if (_epw_path(audit.zone).exists() and pvlib is not None) else "fallback",
    }

# --------------------------------------------------------------------------------------
# Log exécutions (best-effort, JSONL)
# --------------------------------------------------------------------------------------
def _sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def log_run(audit: Audit, params: Dict[str, Any], results: Dict[str, Any]) -> None:
    epw = _epw_path(audit.zone)
    elec_csv = TARIFS_DIR / "elec_spot_2025.csv"
    retail_csv = TARIFS_DIR / "retail_trv.csv"
    payload = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "audit": audit.__dict__,
        "params": params,
        "results": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in results.items()
                    if k.startswith(("cost", "co2", "pac_share", "UA"))},
        "hashes": {
            "epw": _sha256(epw) if epw.exists() else None,
            "elec_spot_2025.csv": _sha256(elec_csv) if elec_csv.exists() else None,
            "retail_trv.csv": _sha256(retail_csv) if retail_csv.exists() else None,
        },
    }
    try:
        with open(LOGS_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

__all__ = [
    "Audit",
    "UA_from_audit",
    "cop_pac",
    "eta_chaudiere",
    "load_meteo",
    "load_tarifs",
    "simulate",
    "log_run",
]

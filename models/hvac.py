"""
Shared HVAC data loading and physics used by both the ILM and NILM predictors.

Formulas:
  Sensible heat : Q [BTU/hr] = 1.08 x CFM x dT          (ASHRAE Fundamentals)
  Chilled water : Q [BTU/hr] = 500 x GPM x dT           (ASHRAE HVAC Systems & Equipment)
  1 ton refrigeration = 12,000 BTU/hr
"""

from __future__ import annotations
import pandas as pd
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from buildings.building_data import VAV, group_vavs_by_ahu

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SENSIBLE_HEAT_CONSTANT = 1.08   # BTU / (hr x cfm x F)
WATER_HEAT_CONSTANT = 500       # BTU / (hr x gpm x F)
MIN_DELTA_T = 0.5               # F, noise floor
KWH_TO_KBTU = 3.412
TONS_TO_KBTU_HR = 12.0
HP_TO_KW = 0.7457
TIMEZONE = "America/Los_Angeles"

# If True, NILM allocates the HVAC sub-meter instead of the whole-building meter (when the file has one).
USE_HVAC_ELECTRIC_METER = False

# AHU ref -> fans (supply + return) whose electric power is attributed to that AHU's VAVs.
# Keyed by the column prefix found in the csv ("S&BS Gateway", "Bren Hall", ...).
#
# !!! The Bren Hall entry is a PLACEHOLDER GUESS (SF/RF 1-2 on AH-01, 3-4 on AH-02). The csv cannot
# !!! tell which fan serves which AHU, so please correct it from the building's drawings.
FAN_MAP: dict[str, dict[str, list[str]]] = {
    "S&BS Gateway": {
        "A01": ["A01.SF1A.VFD", "A01.SF1B.VFD", "A01.RF1A.VFD", "A01.RF1B.VFD"],
        "A02": ["A02.SF1A.VFD", "A02.SF1B.VFD", "A02.SF1C.VFD", "A02.RF1A.VFD", "A02.RF1B.VFD"],
    },
    "Bren Hall": {
        "AH-01": ["SF-1", "SF-2", "RF-1", "RF-2"],
        "AH-02": ["SF-3", "SF-4", "RF-3", "RF-4"],
    },
}

# Chilled-water meter entity differs between buildings: "S&BS Gateway chw btu meter chw flow" vs "Bren Hall CHW chw flow"
_CHW_ENTITIES = ("chw btu meter", "CHW")

# Only columns ending in one of these (case-insensitive) are read from the (very wide) csv.
_KEEP_SUFFIXES = (
    "zone temp", "supply air temp", "supply air flow cfm", "supply air flow.1",
    "elec power", "elec power (na removed)", "hvac elec power",
    "hhw htg power (calculated)",
    "chw clg power", "chw flow", "chw supply temp", "chw return temp",
)

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _norm(name: str) -> str:
    """Collapse the double spaces the BMS export puts in some column names."""
    return re.sub(r"\s+", " ", str(name)).strip()


# ─── Loaded data container ────────────────────────────────────────────────────
@dataclass
class HvacData:
    raw: pd.DataFrame                 # selected columns, still strings with units, row-aligned to `timestamps`
    timestamps: pd.Series             # naive local time, one per row
    interval_hours: float
    prefix: str                       # e.g. "Bren Hall"
    _names: dict[str, str] = field(default_factory=dict)       # normalised name -> real column name
    _num_cache: dict[str, pd.Series] = field(default_factory=dict)
    _thermal_cache: dict[tuple[str, str], "VavThermal | None"] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set)
    _vav_ids: set[str] | None = None

    @property
    def n_rows(self) -> int:
        return len(self.raw)

    def zeros(self) -> pd.Series:
        return pd.Series(0.0, index=self.raw.index)

    def find(self, entities, suffixes) -> str | None:
        """Return the real column name for `<prefix> <entity> <suffix>` (first hit), or None."""
        entities = (entities,) if isinstance(entities, str) else entities
        suffixes = (suffixes,) if isinstance(suffixes, str) else suffixes
        for entity in entities:
            for suffix in suffixes:
                hit = self._names.get(_norm(f"{self.prefix} {entity} {suffix}"))
                if hit is not None:
                    return hit
        return None

    def find_usable(self, entities, suffixes) -> str | None:
        """Like find(), but skips columns that exist and are empty (all NaN / unparseable)."""
        entities = (entities,) if isinstance(entities, str) else entities
        suffixes = (suffixes,) if isinstance(suffixes, str) else suffixes
        for entity in entities:
            for suffix in suffixes:
                col = self.find(entity, suffix)
                if col is not None and self.numeric(col).notna().any():
                    return col
        return None

    @property
    def all_vav_ids(self) -> set[str]:
        """Every VAV that appears in the file (by its airflow column), mapped or not."""
        if self._vav_ids is None:
            ids = set()
            for name in self._names:
                if name.endswith(("supply air flow cfm", "supply air flow.1")):
                    token = name[len(self.prefix) + 1:].split(" ")[0]
                    if token.startswith("VAV"):
                        ids.add(token)
            self._vav_ids = ids
        return self._vav_ids

    def warn_once(self, key: str, message: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message, *args)

    def numeric(self, col: str) -> pd.Series:
        """Column as floats with units stripped ('69.75°F' -> 69.75). Unparseable -> NaN."""
        if col not in self._num_cache:
            s = self.raw[col]
            if not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s.astype("string").str.extract(f"({_NUMBER.pattern})")[0], errors="coerce")
            self._num_cache[col] = s.astype(float)
        return self._num_cache[col]

    def power_kw(self, col: str) -> pd.Series:
        """Electric power in kW, converting from hp / W if that is what the column reports."""
        values = self.numeric(col)
        sample = self.raw[col].dropna()
        unit = re.sub(_NUMBER.pattern, "", str(sample.iloc[0])).strip().lower() if len(sample) else "kw"
        if unit == "hp":
            return values * HP_TO_KW
        if unit == "w":
            return values / 1000
        return values


# ─── Loading ──────────────────────────────────────────────────────────────────
def _detect_prefix(columns: list[str]) -> str:
    for c in columns:
        m = re.match(r"^(.*?)\s+Whole Building Electric Meter\b", _norm(c))
        if m:
            return m.group(1)
    # fallback: most common first two words
    counts = pd.Series([" ".join(_norm(c).split()[:2]) for c in columns if c != "Timestamp"]).value_counts()
    return counts.index[0]


@lru_cache(maxsize=2)
def _load_cached(path: str, mtime: float) -> HvacData:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    keep = [c for c in header
            if c == "Timestamp" or _norm(c).lower().endswith(_KEEP_SUFFIXES)]
    raw = pd.read_csv(path, usecols=keep, dtype=str, low_memory=False)

    # Timestamps look like '2024-12-29T00:00:00-08:00 Los_Angeles'. Drop the trailing zone name, parse as UTC.
    clean = raw["Timestamp"].str.strip().str.replace(r"\s+[A-Za-z_/]+$", "", regex=True)
    utc = pd.to_datetime(clean, utc=True, errors="coerce")
    ok = utc.notna()
    raw, utc = raw[ok], utc[ok]
    # De-duplicate / sort on the unambiguous UTC instant (so the repeated DST hour is not thrown away).
    order = utc.drop_duplicates().sort_values().index
    raw, utc = raw.loc[order], utc.loc[order]

    interval_hours = utc.diff().dropna().median().total_seconds() / 3600
    timestamps = utc.dt.tz_convert(TIMEZONE).dt.tz_localize(None).reset_index(drop=True)
    raw = raw.drop(columns="Timestamp").reset_index(drop=True)

    return HvacData(
        raw=raw,
        timestamps=timestamps,
        interval_hours=interval_hours,
        prefix=_detect_prefix(header),
        _names={_norm(c): c for c in raw.columns},
    )


def load_hvac_data(hvac_file: Path) -> HvacData:
    """Load (and cache) the combined HVAC csv. Calling this from both predictors only reads the file once."""
    p = Path(hvac_file)
    return _load_cached(str(p.resolve()), p.stat().st_mtime)


def get_timestamps(hvac_file: Path) -> list[datetime]:
    """Timestamp for each element of the lists returned by the predictors (naive local time)."""
    return load_hvac_data(hvac_file).timestamps.dt.to_pydatetime().tolist()


# ─── Fraction of the building the passed VAVs represent ───────────────────────
def vav_fraction(data: HvacData, vavs: list[VAV]) -> float:
    """
    Share of the building's VAVs that were passed in (passed VAVs found in the file / all VAVs in the file).
    Metered building energy and AHU fan energy are scaled by this, so passing only part of the building
    is not allocated the whole building's energy.
    """
    total = len(data.all_vav_ids)
    if total == 0:
        return 1.0
    passed = sum(1 for v in vavs if v.vav_name in data.all_vav_ids)
    return min(1.0, passed / total)


# ─── Per-VAV thermal calculation ──────────────────────────────────────────────
@dataclass
class VavThermal:
    saf: pd.Series              # supply airflow, cfm
    q_cool_kbtu: pd.Series      # sensible cooling per interval (zeros if it cannot be computed)
    q_reheat_kbtu: pd.Series    # reheat per interval (zeros if it cannot be computed)
    thermal_signal: pd.Series   # cfm x (dT_cool + dT_reheat), the NILM weighting
    has_cooling: bool
    has_reheat: bool


def _ahu_supply_temp(data: HvacData, ahu_ref: str) -> pd.Series | None:
    col = data.find_usable(ahu_ref, "supply air temp")
    if col is None:
        data.warn_once(f"ahu-sat:{ahu_ref}",
                       "No usable '%s supply air temp' column (check the AHU ref in the mapping); "
                       "reheat cannot be computed for VAVs on this AHU and is treated as 0", ahu_ref)
        return None
    return data.numeric(col).ffill()


def _above_floor(dt: pd.Series) -> pd.Series:
    """Clip negatives, zero the sub-noise-floor readings, treat missing as 0."""
    dt = dt.clip(lower=0).fillna(0)
    return dt.where(dt >= MIN_DELTA_T, 0.0)


def vav_thermal(data: HvacData, vav: VAV) -> VavThermal | None:
    """
    Cooling/reheat energy for one VAV from whatever sensors it actually has:
      cooling needs  airflow + VAV supply temp + zone temp
      reheat  needs  airflow + VAV supply temp + the AHU's supply temp
    Returns None if neither can be computed.
    """
    key = (vav.vav_name, vav.ahu_ref)
    if key in data._thermal_cache:
        return data._thermal_cache[key]

    saf_col = data.find_usable(vav.vav_name, ("supply air flow.1", "supply air flow cfm"))   # '.1' quirk from SBSG export
    sat_col = data.find_usable(vav.vav_name, "supply air temp")
    zt_col = data.find_usable(vav.vav_name, "zone temp")

    result = None
    if saf_col and sat_col:
        saf = data.numeric(saf_col).ffill().fillna(0).clip(lower=0)
        vav_sat = data.numeric(sat_col).ffill()
        h = data.interval_hours

        d_cool, d_reheat = data.zeros(), data.zeros()
        if zt_col:
            d_cool = _above_floor(data.numeric(zt_col).ffill() - vav_sat)
        ahu_sat = _ahu_supply_temp(data, vav.ahu_ref)
        if ahu_sat is not None:
            d_reheat = _above_floor(vav_sat - ahu_sat)

        if zt_col or ahu_sat is not None:
            result = VavThermal(
                saf=saf,
                q_cool_kbtu=SENSIBLE_HEAT_CONSTANT * saf * d_cool * h / 1000,
                q_reheat_kbtu=SENSIBLE_HEAT_CONSTANT * saf * d_reheat * h / 1000,
                thermal_signal=saf * (d_cool + d_reheat),
                has_cooling=bool(zt_col),
                has_reheat=ahu_sat is not None,
            )

    if result is None:
        log.warning("Not enough sensor data for %s; its consumption will be all zeros", vav.vav_name)
    data._thermal_cache[key] = result
    return result


# ─── AHU fans ─────────────────────────────────────────────────────────────────
def ahu_fan_kbtu(data: HvacData, ahu_ref: str) -> pd.Series:
    """Supply + return fan energy for one AHU per interval, in kBTU (0 if fans are unmapped / unmetered)."""
    fans = FAN_MAP.get(data.prefix, {}).get(ahu_ref)
    if fans is None:
        data.warn_once(f"fan-map:{ahu_ref}", "No fan mapping for %s / %s in FAN_MAP; fan energy set to 0",
                       data.prefix, ahu_ref)
        return data.zeros()
    total_kw = data.zeros()
    for fan in fans:
        col = data.find_usable(fan, "elec power")
        if col is None:
            data.warn_once(f"fan:{fan}", "Fan %s has no usable 'elec power' column; ignored", fan)
            continue
        total_kw += data.power_kw(col).fillna(0).clip(lower=0)
    return total_kw * data.interval_hours * KWH_TO_KBTU


# ─── Building-level metered energy (used by NILM) ─────────────────────────────
def building_energy_kbtu(data: HvacData, vav_reheat_kbtu: pd.Series | None, fraction: float = 1.0) -> pd.Series:
    """
    Energy to disaggregate, per interval, in kBTU. Built only from the sources this building actually has;
    each missing source is logged and left out. Raises ValueError if there is nothing at all.

      electric : whole-building (or HVAC) electric meter
      cooling  : chilled-water tonnage and/or flow x dT (averaged when both exist)
      heating  : sum of per-VAV 'hhw htg power (calculated)' columns if present; otherwise estimated
                 as the summed reheat of the passed VAVs (`vav_reheat_kbtu`, None if not computable)

    Metered sources are multiplied by `fraction` (the share of the building's VAVs being predicted).
    The reheat estimate is already a sum over the passed VAVs only, so it is NOT scaled again.
    """
    h = data.interval_hours
    parts: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}

    # 1. Electricity
    elec_col = data.find_usable("HVAC Meter", "hvac elec power") if USE_HVAC_ELECTRIC_METER else None
    if elec_col is None:
        elec_col = data.find_usable("Whole Building Electric Meter", ("elec power", "elec power (NA removed)"))
    if elec_col:
        parts["electric"] = data.power_kw(elec_col).fillna(0).clip(lower=0) * h * KWH_TO_KBTU * fraction
        sources["electric"] = elec_col

    # 2. Chilled water
    tons_col = data.find_usable(_CHW_ENTITIES, "chw clg power")
    flow_col = data.find_usable(_CHW_ENTITIES, "chw flow")
    sup_col = data.find_usable(_CHW_ENTITIES, "chw supply temp")
    ret_col = data.find_usable(_CHW_ENTITIES, "chw return temp")

    chw_tonnage = chw_flow = None
    if tons_col:
        chw_tonnage = data.numeric(tons_col).fillna(0).clip(lower=0) * TONS_TO_KBTU_HR * h
    if flow_col and sup_col and ret_col:
        gpm = data.numeric(flow_col).fillna(0).clip(lower=0)
        d_t = (data.numeric(ret_col).ffill() - data.numeric(sup_col).ffill()).clip(lower=0).fillna(0)
        d_t = d_t.where(d_t >= MIN_DELTA_T, 0.0)
        chw_flow = WATER_HEAT_CONSTANT * gpm * d_t * h / 1000

    if chw_tonnage is not None and chw_flow is not None:
        cooling = (chw_tonnage + chw_flow) / 2 if chw_tonnage.sum() > 0 else chw_flow
        sources["cooling"] = "chilled-water tonnage + flow/dT" if chw_tonnage.sum() > 0 else "chilled-water flow/dT"
    elif chw_tonnage is not None:
        cooling, sources["cooling"] = chw_tonnage, "chilled-water tonnage"
    elif chw_flow is not None:
        cooling, sources["cooling"] = chw_flow, "chilled-water flow/dT"
    else:
        cooling = None
    if cooling is not None:
        parts["cooling"] = cooling * fraction

    # 3. Heating
    hhw_cols = [c for c in data.raw.columns
                if _norm(c).endswith("hhw htg power (calculated)") and data.numeric(c).notna().any()]
    if hhw_cols:
        hhw = sum(data.numeric(c).fillna(0).clip(lower=0) * h for c in hhw_cols)
        parts["heating"] = hhw * fraction
        sources["heating"] = f"{len(hhw_cols)} per-VAV HHW power columns"
    elif vav_reheat_kbtu is not None:
        parts["heating"] = vav_reheat_kbtu
        sources["heating"] = "estimated from summed VAV reheat (no HHW meter data)"

    missing = [c for c in ("electric", "cooling", "heating") if c not in parts]
    if not parts:
        raise ValueError("No electric, cooling or heating data could be found in the HVAC file, "
                         "so there is nothing to disaggregate")
    log.info("NILM energy sources: %s", sources)
    if missing:
        log.warning("NILM is running WITHOUT: %s (not available in this file)", ", ".join(missing))

    return sum(parts.values())


# ─── Convenience for the predictors ───────────────────────────────────────────
def collect_vav_thermals(data: HvacData, vavs: list[VAV]) -> dict[VAV, VavThermal | None]:
    return {vav: vav_thermal(data, vav) for vav in vavs}


def vavs_by_ahu(vavs: list[VAV]) -> dict[str, list[VAV]]:
    return group_vavs_by_ahu(vavs)
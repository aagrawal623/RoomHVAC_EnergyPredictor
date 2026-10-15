from pathlib import Path

from buildings.building_data import VAV
from models.hvac import ahu_fan_kbtu, collect_vav_thermals, load_hvac_data, vav_fraction, vavs_by_ahu


def predict_ilm_consumption(vavs: list[VAV], hvac_file: Path) -> dict[VAV, list[float]]:
    """
    Intrusive Load Monitoring: physics-based energy per VAV from its own sensors.

        cooling = 1.08 x CFM x max(T_zone - T_vav_supply, 0)
        reheat  = 1.08 x CFM x max(T_vav_supply - T_ahu_supply, 0)
        fan     = the AHU's fan energy, split between its VAVs by share of airflow

    Uses whatever sensors exist: cooling needs zone temp, reheat needs the AHU supply temp, fan needs the
    fan power columns; anything unavailable contributes 0 (and is logged) rather than failing.

    Fan energy is split by airflow among the VAVs passed in, then scaled by the fraction of the building's
    VAVs that were passed (see hvac_common.vav_fraction) so a partial list is not given the whole fan load.

    Returns {VAV: [kBTU for each interval]}, one value per row of the HVAC file (15 min each).
    VAVs with no usable sensor data get a list of zeros. Use models.get_timestamps(hvac_file) for the time axis.
    """
    data = load_hvac_data(hvac_file)
    thermals = collect_vav_thermals(data, vavs)
    fraction = vav_fraction(data, vavs)
    result = {vav: [0.0] * data.n_rows for vav in vavs}

    for ahu_ref, ahu_vavs in vavs_by_ahu(vavs).items():
        usable = [(v, thermals[v]) for v in ahu_vavs if thermals[v] is not None]
        if not usable:
            continue

        system_saf = sum(t.saf for _, t in usable).replace(0, float("nan"))
        fan_kbtu = ahu_fan_kbtu(data, ahu_ref) * fraction

        for vav, t in usable:
            saf_share = (t.saf / system_saf).fillna(0).clip(0, 1)
            total = t.q_cool_kbtu + t.q_reheat_kbtu + saf_share * fan_kbtu
            result[vav] = total.tolist()

    return result
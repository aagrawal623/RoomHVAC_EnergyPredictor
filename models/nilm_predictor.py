from pathlib import Path

from buildings.building_data import VAV
from models.hvac import building_energy_kbtu, collect_vav_thermals, load_hvac_data, vav_fraction


#Make algorithm go line by line

def predict_nilm_consumption(vavs: list[VAV], hvac_file: Path) -> dict[VAV, list[float]]:
    """
    Non-Intrusive Load Monitoring: disaggregate the building's metered energy (electric + chilled water
    + hot water) down to VAVs, in proportion to each VAV's thermal signal  CFM x (dT_cool + dT_reheat).

    Only the energy sources the building actually has are used (electric / cooling / heating; missing ones are
    logged and left out; ValueError if none exist). If there is no hot-water data, heating is estimated from
    the VAVs' reheat.

    Each VAV's share is relative to the VAVs passed in, and the metered energy is scaled by the fraction of
    the building's VAVs that were passed (see hvac_common.vav_fraction), so passing part of the building
    allocates only that part's share of the total.

    Returns {VAV: [kBTU for each interval]}, one value per row of the HVAC file (15 min each).
    VAVs with no usable sensor data get a list of zeros.
    """
    data = load_hvac_data(hvac_file)
    thermals = collect_vav_thermals(data, vavs)
    usable = {vav: t for vav, t in thermals.items() if t is not None}

    result = {vav: [0.0] * data.n_rows for vav in vavs}
    if not usable:
        return result

    total_signal = sum(t.thermal_signal for t in usable.values()).replace(0, float("nan"))
    reheat = [t.q_reheat_kbtu for t in usable.values() if t.has_reheat]
    building_kbtu = building_energy_kbtu(data, sum(reheat) if reheat else None,
                                         vav_fraction(data, vavs))

    for vav, t in usable.items():
        share = (t.thermal_signal / total_signal).fillna(0).clip(0, 1)
        result[vav] = (share * building_kbtu).tolist()

    return result
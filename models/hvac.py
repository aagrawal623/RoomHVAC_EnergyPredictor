"""
Shared HVAC Data Models and Loaders.

This module provides data structures and a memory-efficient generator to parse HVAC CSV data
line-by-line. It defines the physical constants and equipment mappings required for
energy consumption calculations.
"""

import csv
import re
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Generator, Any, Union, Set, Iterable
from dateutil import parser


# Physical and Conversion Constants
SENSIBLE_HEAT_CONSTANT = 1.08    # BTU / (hr * cfm * °F)
WATER_HEAT_CONSTANT = 500        # BTU / (hr * gpm * °F)
MIN_DELTA_T = 0.5                # °F, noise floor for temperature differentials
KWH_TO_KBTU = 3.412              # kBTU per kWh
TONS_TO_KBTU_HR = 12.0           # kBTU/hr per Ton of refrigeration


# Equipment Mapping: Maps Building -> AHU -> List of Associated Fans
FAN_MAP: Dict[str, Dict[str, List[str]]] = {
    "Bren Hall": {
        "AH-01": ["SF-1", "SF-2", "RF-1", "RF-2"],
        "AH-02": ["SF-3", "SF-4", "RF-3", "RF-4"],
    }
}


@dataclass
class HVACRecord:
    """
    Represents a stream for a single timestamp's HVAC data across the entire building.

    Attributes:
        timestamp (datetime): The naive datetime of the current record.
        interval_hours (float): The duration of this reading in hours (e.g., 0.25 for 15 mins).
        vav_data (Dict[str, Dict[str, float]]): Data keyed by VAV name (e.g., airflow, temps).
        ahu_data (Dict[str, Dict[str, float]]): Data keyed by AHU reference (e.g., fan power).
        building_data (Dict[str, float]): Building-level metered data (e.g., electric, chilled water).
    """
    timestamp: datetime
    interval_hours: float
    vav_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ahu_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    building_data: Dict[str, float] = field(default_factory=dict)

    @property
    def day_of_week(self) -> int:
        """Returns the day of the week as an integer (0 = Monday, 6 = Sunday)."""
        return self.timestamp.weekday()

    @property
    def hour(self) -> int:
        """Returns the hour of the day (0-23)."""
        return self.timestamp.hour


class EnergyAccumulator:
    """
    Accumulates calculated energy consumption over time. Groups calculated ILM and NILM energy
    values by day of the week and hour for reporting or downstream visualization.
    """

    def __init__(self) -> None:
        """Dictionary structure: consumption[day_of_week, hour][vav_name] = [ilm_energy, nilm_energy]"""
        self.consumption = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    def add_record(self, record: HVACRecord, ilm_results: Dict[Any, float], nilm_results: Dict[Any, float]) -> None:
        """
        Adds a single timestep's calculations to the running total given the HVACRecord, ilm_results,
        and nilm_results per VAV.
        """
        key = (record.day_of_week, record.hour)
        for vav, energy in ilm_results.items():
            self.consumption[key][vav.vav_name][0] += energy
        for vav, energy in nilm_results.items():
            self.consumption[key][vav.vav_name][1] += energy


def _safe_float(val: Any) -> float:
    """
    Safely casts a raw CSV string to a float and returns it, stripping units if present.
    Returns 0.0 if parsing fails or value is None.
    """
    if not val:
        return 0.0
    try:
        cleaned = re.sub(r"[^\d\.\-]", "", str(val))
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def hvac_record_generator(filepath: Union[str, Path], active_vavs: Iterable[Any], prefix: str = "Bren Hall",
                          interval_hours: float = 0.25) -> Generator[HVACRecord, None, None]:
    """
    A memory-efficient generator that parses HVAC data line-by-line to yield an HVACRecord of
    all the necessary sensor data for the current timestamp.

    Args:
        filepath (str): The file path to the CSV data.
        active_vavs (List[VAV]): List of VAV objects to parse data for.
        prefix (str): The building prefix used in the CSV column headers.
        interval_hours (float): The time interval of each row (default is 15 minutes / 0.25 hours).
    """
    vav_names = [v.vav_name for v in active_vavs]
    ahu_names = list({v.ahu_ref for v in active_vavs})
    building_fans = FAN_MAP.get(prefix, {})

    with open(filepath, 'r', encoding = 'utf-8') as f:
        reader = csv.DictReader(f)

        header_map = {}
        for k in reader.fieldnames or []:
            if k:
                clean_k = re.sub(r'\s+', ' ', str(k).strip().lower())
                header_map[clean_k] = k

        def get_val(row_dict: dict, expected_col: str) -> Any:
            """Fetches a value ignoring case and extra spacing."""
            clean_expected = re.sub(r'\s+', ' ', expected_col.strip().lower())
            actual_col = header_map.get(clean_expected)
            return row_dict.get(actual_col, 0) if actual_col else 0

        for row in reader:
            actual_ts_col = header_map.get('timestamp')
            if not actual_ts_col or not row.get(actual_ts_col):
                continue

            dt = parser.parse(row[actual_ts_col].split()[0]).replace(tzinfo = None)
            record = HVACRecord(timestamp = dt, interval_hours = interval_hours)

            # Parse VAV Data
            for vav in vav_names:
                record.vav_data[vav] = {
                    'saf': _safe_float(get_val(row, f"{prefix} {vav} supply air flow cfm")),
                    'sat': _safe_float(get_val(row, f"{prefix} {vav} supply air temp")),
                    'zt': _safe_float(get_val(row, f"{prefix} {vav} zone temp")),
                    'hhw': _safe_float(get_val(row, f"{prefix} {vav} hhw htg power (calculated)"))
                }

            # Parse AHU Data
            for ahu in ahu_names:
                fan_power_kw = sum(
                    _safe_float(get_val(row, f"{prefix} {fan} elec power"))
                    for fan in building_fans.get(ahu, [])
                )
                record.ahu_data[ahu] = {
                    'sat': _safe_float(get_val(row, f"{prefix} {ahu} supply air temp")),
                    'fan_kw': fan_power_kw
                }

            # Parse Building Data
            record.building_data['electric_kw'] = _safe_float(get_val(row, f"{prefix} Whole Building Electric Meter total elec power"))
            record.building_data['chw_tons'] = _safe_float(get_val(row, f"{prefix} CHW chw clg power"))
            record.building_data['chw_flow'] = _safe_float(get_val(row, f"{prefix} CHW chw flow"))
            record.building_data['chw_return'] = _safe_float(get_val(row, f"{prefix} CHW chw return temp"))
            record.building_data['chw_supply'] = _safe_float(get_val(row, f"{prefix} CHW chw supply temp"))

            yield record

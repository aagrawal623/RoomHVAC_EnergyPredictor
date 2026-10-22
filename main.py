"""
Author: Aadi Agrawal
Purpose: A program to predict/plot room energy consumption for a smart building from HVAC data
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, Union, Optional

import buildings
import models
from buildings.building_data import VAV


# ─── Configuration ─────────────────────────────────────────────────────────────────────
"""
If chosen to not use the command line, can configure the program using these global variables
"""
BUILDING = 'BrenHall'    # the smart building in which room energy consumption is investigated
PURPOSE = 'predict'       # 0 = predict, 1 = plot
ROOMS = None              # None = all rooms, INT = specific room, STR = room type
# ───────────────────────────────────────────────────────────────────────────────────────


def _get_rooms_to_predict(vavs: Set[VAV], rooms: Union[None, int, str]) -> Set[int]:
    if rooms is None:
        return {room.room_num for vav in vavs for room in vav.rooms_associated}
    if isinstance(rooms, str):
        return {room.room_num for vav in vavs for room in vav.rooms_associated if room.room_type == rooms}
    return {rooms}


def main(building: str, purpose: int, rooms: None | int | str) -> None:
    # Get Building VAV information
    vav_room_mapping = Path(f'data/{building}_vav_room_mapping.csv')
    hvac_file = Path(f'data/{building}2025_combined.csv')

    if not vav_room_mapping.is_file():
        print(f'Mising {vav_room_mapping} file')
        return
    if not hvac_file.is_file():
        print(f'Missing {hvac_file} file')
        return

    try:
        building_vavs = buildings.get_building_vavs(vav_room_mapping)
    except ValueError as e:
        print(e)
        return

    target_rooms = _get_rooms_to_predict(building_vavs, rooms)

    accumulator = models.EnergyAccumulator()
    prefix = "Bren Hall" if building in ("Bren Hall", "BrenHall") else building

    generator = models.hvac_record_generator(hvac_file, building_vavs, prefix = prefix)
    for record in generator:
        ilm_results = models.predict_ilm_consumption(building_vavs, record)
        nilm_results = models.predict_nilm_consumption(building_vavs, record)

        accumulator.add_record(record, ilm_results, nilm_results)

    room_consumption = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    for time_key, vav_data in accumulator.consumption.items():
        for vav in building_vavs:
            if vav.vav_name not in vav_data:
                continue

            ilm_energy, nilm_energy = vav_data[vav.vav_name]
            num_rooms = len(vav.rooms_associated)
            if num_rooms == 0:
                continue

            for room in vav.rooms_associated:
                if room.room_num in target_rooms:
                    room_consumption[time_key][room.room_num][0] += (ilm_energy / num_rooms)
                    room_consumption[time_key][room.room_num][1] += (nilm_energy / num_rooms)

    if purpose == 0:
        for time_key, rooms_data in sorted(room_consumption.items()):
            day, hr = time_key
            print(f"--- Day {day}, Hour {hr:02d} ---")
            for rm, energy in rooms_data.items():
                print(f" Room {rm}: ILM = {energy[0]:.2f} kBTU, NILM = {energy[1]:.2f} kBTU")
    else:
        print("Plotting functionality not yet implemented.")


if __name__ == '__main__':
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        BUILDING = sys.argv[1]
        PURPOSE = sys.argv[2]
        try:
            ROOMS = int(sys.argv[3]) if len(sys.argv) == 4 else None
        except ValueError:
            ROOMS = sys.argv[3]

    elif len(sys.argv) != 1:
        print("Usage: main.py building purpose rooms")
        exit(1)

    if BUILDING != 'BrenHall' and BUILDING != 'SBSGateway':
        print("BUILDING must be either BrenHall or SBSGateway")
        exit(1)

    if PURPOSE != 'predict' and PURPOSE != 'plot':
        print("PURPOSE must be either predict or plot")
        exit(1)
    else:
        PURPOSE = 0 if PURPOSE == 'predict' else 1

    main(BUILDING, PURPOSE, ROOMS)

"""
Author: Aadi Agrawal
Purpose: A program to predict/plot room energy consumption for a smart building from HVAC data
"""

import sys
from pathlib import Path

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


def _get_rooms_to_predict(vavs: list[VAV], rooms: None | int | str) -> set[int]:
    if rooms is None:
        return set(room.room_num for vav in vavs for room in vav.rooms_associated if room.room_num not in rooms)
    if type(rooms) is str:
        return set(room.room_num for vav in vavs for room in vav.rooms_associated if room.room_type == rooms)
    return set([rooms])


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

    building_vavs = buildings.get_building_vavs(vav_room_mapping)

    # Calculate ILM and NILM energy consumption associated with each VAV
    vav_ilm = models.predict_ilm_consumption(building_vavs, hvac_file)
    vav_nilm = models.predict_nilm_consumption(building_vavs, hvac_file)

    print(vav_ilm)
    print(vav_nilm)

    # Transform from VAV energy consumption to room energy consumption
    rooms = _get_rooms_to_predict(building_vavs, rooms)


    # fix display output and how to handle different rooms
    # room_consumption = dict(zip(rooms, [0, 0] * len(rooms)))
    # for vav in vav_ilm:
    #     for room in vav.rooms_associated:
    #         room_consumption[room][0] += vav_ilm[vav] / len(vav.rooms_associated)
    # for vav in vav_nilm:
    #     for room in vav.rooms_associated:
    #         room_consumption[room][1] += vav_nilm[vav] / len(vav.rooms_associated)
    #
    # # Print predictions or plot them
    # if purpose == 0:
    #     print(room_consumption)
    # else:
    #     pass


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

    if BUILDING != 'Bren Hall' and BUILDING != 'SBSGateway':
        print("BUILDING must be either Bren Hall or SBSGateway")
        exit(1)

    if PURPOSE != 'predict' and PURPOSE != 'plot':
        print("PURPOSE must be either predict or plot")
        exit(1)
    else:
        PURPOSE = 0 if PURPOSE == 'predict' else 1

    main(BUILDING, PURPOSE, ROOMS)

import pandas as pd
from pathlib import Path


class Room:
    def __init__(self, room_num: int, room_type: str, room_capacity: int, room_bucket: str):
        self.room_num = room_num
        self.room_type = room_type
        self.room_capacity = room_capacity
        self.room_bucket = room_bucket

    def __repr__(self) -> str:
        return f"Room({self.room_num}, {self.room_type!r}, cap={self.room_capacity}, {self.room_bucket!r})"


class VAV:
    def __init__(self, vav_name: str, ahu_ref: str):
        self.vav_name = vav_name
        self.ahu_ref = ahu_ref
        self.rooms_associated = []

    def add_room(self, room: Room):
        self.rooms_associated.append(room)

    def __repr__(self) -> str:
        return f"VAV({self.vav_name!r}, ahu={self.ahu_ref!r}, rooms={len(self.rooms_associated)})"


# Accepted header names(case - insensitive) for each field in the mapping file.
_COLUMN_ALIASES = {
    "vav_name": ("vavname", "vav_name", "vav", "vav_id"),
    "ahu_ref": ("ahuref", "ahu_ref", "ahu", "ahu_id", "ahu_system"),
    "room_num": ("room_number", "room_num", "room"),
    "room_type": ("room_type", "type"),
    "room_capacity": ("room_capacity", "capacity"),
    "room_bucket": ("room_bucket", "bucket"),
}


def _resolve_columns(columns: list[str]) -> dict[str, str | None]:
    lowered = {c.strip().lower(): c for c in columns}
    resolved = {}
    for field, aliases in _COLUMN_ALIASES.items():
        resolved[field] = next((lowered[a] for a in aliases if a in lowered), None)
    for required in ("vav_name", "ahu_ref"):
        if resolved[required] is None:
            raise ValueError(
                f"Mapping file needs a '{required}' column (accepted names: {_COLUMN_ALIASES[required]}); "
                f"found {columns}"
            )
    return resolved


def _short_ahu_ref(ref: str) -> str:
    """'Bren Hall AH-01' -> 'AH-01' (the id the HVAC csv uses for the AHU, without the building name)."""
    parts = ref.split()
    return parts[-1] if parts else ""


def get_building_vavs(file: Path) -> list[VAV]:
    """
    Parse the VAV/room mapping file (e.g. BrenHall_vav_room_type_confirmed.csv) into a list of VAVs.

    One row per VAV. Columns used (extra columns, including the duplicated table on the right of the
    Bren Hall file, are ignored):

        vavName, ahuRef, room_number, room_type, room_capacity, room_bucket

    - The AHU for each VAV comes from `ahuRef`, with the building name stripped ('Bren Hall AH-01' -> 'AH-01').
    - Blank rows are skipped; blank capacity becomes 0 and blank bucket/type become ''.
    - Every VAV gets its own Room object(s). If a room is served by several VAVs, each of those VAVs lists it
      separately (same room_num); combining them is left to whoever totals up room energy.
    - Order of first appearance in the file is preserved.
    """
    df = pd.read_csv(file, dtype = str, keep_default_na = False)
    cols = _resolve_columns(list(df.columns))

    def cell(row, field: str) -> str:
        col = cols[field]
        return row[col].strip() if col else ""

    vavs: dict[str, VAV] = {}
    for line_no, row in df.iterrows():
        vav_name = cell(row, "vav_name")
        if not vav_name:
            continue

        ahu_ref = _short_ahu_ref(cell(row, "ahu_ref"))
        vav = vavs.get(vav_name)
        if vav is None:
            vav = vavs[vav_name] = VAV(vav_name, ahu_ref)
        elif ahu_ref and vav.ahu_ref != ahu_ref:
            raise ValueError(
                f"{file}: VAV {vav_name!r} maps to two AHUs ({vav.ahu_ref!r} and {ahu_ref!r})")

        room_num = cell(row, "room_num")
        if not room_num:
            continue
        try:
            vav.add_room(Room(
                room_num = int(float(room_num)),
                room_type = cell(row, "room_type"),
                room_capacity = int(float(cell(row, "room_capacity") or 0)),
                room_bucket = cell(row, "room_bucket"),
            ))
        except ValueError as e:
            raise ValueError(f"{file}: bad room data on row {line_no + 2}: {e}") from e

    return list(vavs.values())


def group_vavs_by_ahu(vavs: list[VAV]) -> dict[str, list[VAV]]:
    """VAV -> AHU lookup in the other direction: AHU ref -> the VAVs it serves."""
    groups: dict[str, list[VAV]] = {}
    for vav in vavs:
        groups.setdefault(vav.ahu_ref, []).append(vav)
    return groups

"""
merge_hotel_sheets.py — combine two BIM JSON exports from separate Revit sheets.

When hotel rooms are spread across two sheets (e.g. bedrooms on sheet 1,
bathrooms/closets on sheet 2), run extract_revit_hotel.py on each sheet
separately with KEY_METHOD = "grid_bay", then use this script to merge
the two exports into one complete BIM JSON per floor.

Matching is done by key_id — both exports must use the same grid bay labels,
which is guaranteed when KEY_METHOD = "grid_bay" is used in Revit.

Usage
-----
  python merge_hotel_sheets.py sheet1.json sheet2.json --out merged.json
  python merge_hotel_sheets.py sheet1.json sheet2.json  # writes sheet1_merged.json

Keys present in only one sheet are kept as-is (no rooms dropped).
Keys present in both sheets have their room lists merged and door lists combined,
with room IDs renumbered to be contiguous from 0.
"""

import argparse
import json
import sys
from pathlib import Path


def _load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _merge_keys(key_a, key_b):
    """
    Merge two key records that share the same key_id.
    Rooms from key_b are appended after key_a's rooms with renumbered IDs.
    Doors from both sides are remapped to the new IDs.
    key_configuration and unit_type are taken from whichever key has the
    richer classification (suite > deluxe > standard).
    """
    TYPE_RANK = {'suite': 2, 'deluxe': 1, 'standard': 0, 'unknown': -1}
    CFG_RANK  = {'suite': 1, 'standard': 0}

    rooms_a = key_a.get('rooms', [])
    rooms_b = key_b.get('rooms', [])
    doors_a = key_a.get('doors', [])
    doors_b = key_b.get('doors', [])

    offset = len(rooms_a)

    # Renumber rooms_b starting from offset
    id_map_b = {}
    new_rooms_b = []
    for room in rooms_b:
        old_id = room['id']
        new_id = old_id + offset
        id_map_b[old_id] = new_id
        r = dict(room)
        r['id'] = new_id
        new_rooms_b.append(r)

    # Remap doors_b to new IDs
    new_doors_b = []
    for d in doors_b:
        f = id_map_b.get(d['from'], d['from'] + offset)
        t = id_map_b.get(d['to'],   d['to']   + offset) if d['to'] != -1 else -1
        new_doors_b.append({'from': f, 'to': t})

    # Pick the richer classification
    utype = key_a['unit_type'] if (
        TYPE_RANK.get(key_a['unit_type'], -1) >= TYPE_RANK.get(key_b['unit_type'], -1)
    ) else key_b['unit_type']

    cfg = key_a['key_configuration'] if (
        CFG_RANK.get(key_a['key_configuration'], 0) >= CFG_RANK.get(key_b['key_configuration'], 0)
    ) else key_b['key_configuration']

    return {
        'key_id':            key_a['key_id'],
        'unit_type':         utype,
        'key_configuration': cfg,
        'rooms':             rooms_a + new_rooms_b,
        'doors':             doors_a + new_doors_b,
    }


def merge(data_a, data_b):
    """
    Merge two BIM JSON dicts.  Returns a new dict with combined keys.
    Metadata (star_rating, floor) is taken from data_a.
    """
    index_a = {k['key_id']: k for k in data_a.get('keys', [])}
    index_b = {k['key_id']: k for k in data_b.get('keys', [])}

    all_ids = sorted(set(list(index_a.keys()) + list(index_b.keys())))

    merged_keys = []
    only_a = only_b = both = 0

    for kid in all_ids:
        in_a = kid in index_a
        in_b = kid in index_b
        if in_a and in_b:
            merged_keys.append(_merge_keys(index_a[kid], index_b[kid]))
            both += 1
        elif in_a:
            merged_keys.append(index_a[kid])
            only_a += 1
        else:
            merged_keys.append(index_b[kid])
            only_b += 1

    print("  Keys in both sheets : {0}".format(both))
    print("  Keys only in sheet 1: {0}".format(only_a))
    print("  Keys only in sheet 2: {0}".format(only_b))
    print("  Total keys           : {0}".format(len(merged_keys)))

    return {
        'star_rating': data_a.get('star_rating', 0),
        'floor':       data_a.get('floor', ''),
        'keys':        merged_keys,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Merge two BIM JSON exports from separate Revit sheets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python merge_hotel_sheets.py bedrooms.json bathrooms.json --out floor.json\n"
            "\nAfter merging, continue with:\n"
            "  python extract_hotel_keys.py floor.json --out_dir extracted_keys/"
        )
    )
    parser.add_argument('sheet1', help='First BIM JSON (e.g. bedrooms sheet)')
    parser.add_argument('sheet2', help='Second BIM JSON (e.g. bathrooms/closets sheet)')
    parser.add_argument('--out', default=None,
                        help='Output path (default: <sheet1 stem>_merged.json)')
    args = parser.parse_args()

    path_a = Path(args.sheet1)
    path_b = Path(args.sheet2)
    out_path = Path(args.out) if args.out else path_a.with_name(path_a.stem + '_merged.json')

    print("Hotel sheet merge")
    print("=" * 40)
    print("  Sheet 1 : {0}".format(path_a))
    print("  Sheet 2 : {0}".format(path_b))
    print("  Output  : {0}".format(out_path))
    print("")

    data_a = _load(path_a)
    data_b = _load(path_b)

    result = merge(data_a, data_b)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print("")
    print("Written to {0}".format(out_path))
    print("Next: python extract_hotel_keys.py \"{0}\" --out_dir extracted_keys/".format(out_path))


if __name__ == '__main__':
    main()

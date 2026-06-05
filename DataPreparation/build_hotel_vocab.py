"""
Stage 2 of the hotel data pipeline — vocabulary builder.

Scans all unit JSON files produced by extract_hotel_keys.py, collects every
unique room name string, and writes a hotel_vocab.json mapping file.

Before running Stage 3 (convert_hotel_to_mat.py), review hotel_vocab.json and:
  - Merge synonyms  e.g. rename "WC" and "Guest WC" to the same integer
  - Remove types you don't want the model to learn  (delete the entry)
  - Check that integers remain unique and contiguous from 0

Usage
-----
  python build_hotel_vocab.py extracted_keys/
  python build_hotel_vocab.py extracted_keys/ --out hotel_vocab.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# Room types that typically live at floor level (corridor, lobby, etc.) rather
# than inside a key unit.  They will not appear in extracted unit files but
# are listed here as a reminder to add them manually to hotel_vocab.json if
# your dataset includes full-floor layouts.
FLOOR_LEVEL_TYPES = [
    'Corridor',
    'Entrance',
    'Lobby',
    'ElevatorLobby',
    'ServiceRoom',
    'LinenRoom',
    'StaffArea',
]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan(extracted_dir):
    """
    Read every unit JSON in extracted_dir and collect room name statistics.

    Returns
    -------
    name_counter      : Counter  name → total occurrences across all units
    by_unit_type      : dict     unit_type → Counter of name → occurrences
    by_star           : dict     star_rating → Counter of name → occurrences
    by_key_config     : dict     key_configuration → set of names seen
    total_units       : int
    skipped           : int
    """
    extracted_dir = Path(extracted_dir)
    files = sorted(extracted_dir.glob('*.json'))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in '{extracted_dir}'.\n"
            "Run extract_hotel_keys.py first."
        )

    name_counter  = Counter()
    by_unit_type  = defaultdict(Counter)
    by_star       = defaultdict(Counter)
    by_key_config = defaultdict(set)
    total_units   = 0
    skipped       = 0

    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                unit = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {fpath.name}: {e}")
            skipped += 1
            continue

        rooms = unit.get('rooms', [])
        if not rooms:
            skipped += 1
            continue

        total_units   += 1
        utype  = unit.get('unit_type', 'unknown')
        star   = unit.get('star_rating', 0)
        kconfig = unit.get('key_configuration', 'unknown')

        for room in rooms:
            name = room.get('name', 'Unknown').strip()
            if not name:
                name = 'Unknown'
            name_counter[name]       += 1
            by_unit_type[utype][name] += 1
            by_star[star][name]       += 1
            by_key_config[kconfig].add(name)

    return name_counter, by_unit_type, by_star, by_key_config, total_units, skipped


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocab(name_counter):
    """
    Assign a contiguous integer to each unique room name.
    Sorted by descending frequency, then alphabetically to break ties.

    Returns
    -------
    dict  name → int
    """
    sorted_names = sorted(
        name_counter.keys(),
        key=lambda n: (-name_counter[n], n.lower())
    )
    return {name: idx for idx, name in enumerate(sorted_names)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(name_counter, by_unit_type, by_star, by_key_config,
                 total_units, skipped, vocab):
    w = 55
    print()
    print('─' * w)
    print(f"  Units scanned  : {total_units}  (skipped: {skipped})")
    print(f"  Unique names   : {len(vocab)}")
    print('─' * w)
    print(f"  {'Idx':<5}  {'Count':>6}  Name")
    print(f"  {'───':<5}  {'─────':>6}  {'─' * 30}")
    for name, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        print(f"  {idx:<5}  {name_counter[name]:>6}  {name}")

    print()
    print('  Room names by unit type:')
    for utype, counter in sorted(by_unit_type.items()):
        names_str = ', '.join(
            f"{n}({c})" for n, c in counter.most_common()
        )
        print(f"    {utype:<12}: {names_str}")

    print()
    print('  Room names by star rating:')
    for star, counter in sorted(by_star.items()):
        names_str = ', '.join(
            f"{n}({c})" for n, c in counter.most_common()
        )
        print(f"    {star}-star       : {names_str}")

    print()
    print('  Room names by key configuration:')
    for kconfig, names in sorted(by_key_config.items()):
        names_str = ', '.join(sorted(names))
        print(f"    {kconfig:<12}: {names_str}")

    print('─' * w)
    print()


def _names_present_only_in(unit_type, by_unit_type):
    """Return names that appear in unit_type but no other unit type."""
    others = set()
    for utype, counter in by_unit_type.items():
        if utype != unit_type:
            others.update(counter.keys())
    return set(by_unit_type[unit_type].keys()) - others


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: Build hotel room vocabulary from extracted key JSONs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_hotel_vocab.py extracted_keys/\n"
            "  python build_hotel_vocab.py extracted_keys/ --out hotel_vocab.json\n"
            "\nEdit hotel_vocab.json to merge synonyms, then run convert_hotel_to_mat.py."
        )
    )
    parser.add_argument(
        'extracted_dir',
        help='Directory of unit JSON files produced by extract_hotel_keys.py'
    )
    parser.add_argument(
        '--out', default='hotel_vocab.json',
        help='Output vocab JSON file (default: hotel_vocab.json)'
    )
    args = parser.parse_args()

    print("Hotel vocabulary builder — Stage 2")
    print("=" * 40)
    print(f"Scanning {args.extracted_dir} ...")

    try:
        name_counter, by_unit_type, by_star, by_key_config, total_units, skipped = \
            scan(args.extracted_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if not name_counter:
        print("[ERROR] No room names found. Ensure extracted files contain 'rooms'.")
        return

    vocab = build_vocab(name_counter)

    # Highlight names unique to suites — those are the extras we hadn't defined yet
    suite_only = _names_present_only_in('suite', by_unit_type)

    # Write output
    out_path = Path(args.out)
    output = {
        '_instructions': [
            "Review before running Stage 3 (convert_hotel_to_mat.py).",
            "To merge synonyms: set both names to the same integer.",
            "To remove a type: delete its entry.",
            "Integers must be unique and contiguous starting from 0.",
            "Add floor-level types (Corridor, Lobby, etc.) manually if needed.",
        ],
        '_suite_only_names': sorted(suite_only),
        '_floor_level_types_to_add_if_needed': FLOOR_LEVEL_TYPES,
        'vocab': vocab,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print_report(name_counter, by_unit_type, by_star, by_key_config,
                 total_units, skipped, vocab)

    print(f"Wrote {len(vocab)} room type(s) to {out_path}")

    if suite_only:
        print(f"\n  Names found only in suites (review these carefully):")
        for name in sorted(suite_only):
            print(f"    • {name}")

    print(
        f"\nNext steps:\n"
        f"  1. Open {out_path} and review the vocab mapping\n"
        f"  2. Merge any synonyms (e.g. 'WC' and 'Guest Bathroom' → same integer)\n"
        f"  3. Run Stage 3: python convert_hotel_to_mat.py extracted_keys/ "
        f"--vocab {out_path}\n"
    )


if __name__ == '__main__':
    main()

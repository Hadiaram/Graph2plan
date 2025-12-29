"""
Generate mapping between ResPlan plan_XXXXX and Graph2plan numeric IDs

This creates a CSV file mapping:
  plan_00000 → 293
  plan_00001 → 10028
  plan_00538 → 2957
  etc.
"""

import pickle
import json
import csv
from pathlib import Path

def load_mat_data(mat_path):
    """Load data from .mat file"""
    try:
        import scipy.io as sio
        mat_data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        return mat_data['data']
    except Exception as e:
        print(f"Failed to load .mat file: {e}")
        return None

def load_pkl_data(pkl_path):
    """Load data from .pkl file"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Handle different data structures
        if isinstance(data, dict):
            if 'data' in data:
                items = data['data']
                namelist = data.get('nameList', None)
                return items, namelist
            else:
                return data, None
        else:
            return data, None
    except Exception as e:
        print(f"Failed to load .pkl file: {e}")
        return None, None

def extract_id_from_item(item, index):
    """Extract ID from a data item"""
    # Try different ways to get the ID
    if hasattr(item, 'name'):
        return str(item.name).strip()
    elif isinstance(item, dict) and 'name' in item:
        return str(item['name']).strip()
    elif isinstance(item, dict) and 'id' in item:
        return str(item['id']).strip()
    else:
        # Fallback to index
        return f"floorplan_{index:05d}"

def generate_mapping_from_source():
    """Generate mapping from original source data"""
    print("=" * 70)
    print("ResPlan ID Mapping Generator")
    print("=" * 70)

    # Try to find source data in order of preference
    source_paths = [
        Path('./data/data.mat'),
        Path('../Interface/static/Data/data.mat'),
        Path('./data_combined.pkl'),
    ]

    mapping = []

    for source_path in source_paths:
        if source_path.exists():
            print(f"\nFound source data: {source_path}")

            if source_path.suffix == '.mat':
                data = load_mat_data(source_path)
                if data is not None:
                    print(f"Loaded {len(data)} items from .mat file")
                    for i, item in enumerate(data):
                        plan_id = extract_id_from_item(item, i)
                        mapping.append({
                            'index': i,
                            'plan_name': f'plan_{i:05d}',
                            'graph2plan_id': plan_id
                        })
                    break

            elif source_path.suffix == '.pkl':
                items, namelist = load_pkl_data(source_path)
                if items is not None:
                    print(f"Loaded {len(items)} items from .pkl file")
                    for i, item in enumerate(items):
                        plan_id = extract_id_from_item(item, i)
                        mapping.append({
                            'index': i,
                            'plan_name': f'plan_{i:05d}',
                            'graph2plan_id': plan_id
                        })
                    break

    # If no source data found, reconstruct from train+test sets
    if not mapping:
        print("\nNo source data.mat found. Reconstructing from train/test sets...")
        mapping = reconstruct_from_train_test()

    return mapping

def reconstruct_from_train_test():
    """Reconstruct original ordering from train and test sets"""

    # Load both train and test data
    train_path = Path('../Interface/static/Data/data_train_converted.pkl')
    test_path = Path('../Interface/static/Data/data_test_converted.pkl')

    all_items = []
    train_mapping = []
    test_mapping = []

    if train_path.exists():
        train_data, train_namelist = load_pkl_data(train_path)
        if train_data is not None and train_namelist is not None:
            print(f"  Loaded {len(train_namelist)} training items")
            for i, (item, name) in enumerate(zip(train_data, train_namelist)):
                plan_id = str(name).strip()
                all_items.append({
                    'graph2plan_id': plan_id,
                    'source': 'train',
                    'source_index': i
                })

    if test_path.exists():
        with open(test_path, 'rb') as f:
            test_dict = pickle.load(f)
        test_data = test_dict['data']
        test_namelist = test_dict.get('testNameList', test_dict.get('nameList', []))
        if test_data is not None and test_namelist is not None and len(test_namelist) > 0:
            print(f"  Loaded {len(test_namelist)} test items")
            for i, (item, name) in enumerate(zip(test_data, test_namelist)):
                plan_id = str(name).strip()
                all_items.append({
                    'graph2plan_id': plan_id,
                    'source': 'test',
                    'source_index': i
                })

    # Sort by numeric ID to approximate original order
    # (This is an approximation - true order may differ)
    print("\n  WARNING: Reconstructed mapping is approximate!")
    print("  Original sequential order may differ from numeric ID order.")

    all_items.sort(key=lambda x: int(x['graph2plan_id']) if x['graph2plan_id'].isdigit() else 999999)

    combined_mapping = []
    for i, item in enumerate(all_items):
        combined_mapping.append({
            'index': i,
            'plan_name': f'plan_{i:05d}',
            'graph2plan_id': item['graph2plan_id'],
            'source': item['source'],
            'source_index': item['source_index']
        })

    return combined_mapping

def save_mapping(mapping, output_dir):
    """Save mapping to CSV and JSON files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined mapping as CSV
    csv_path = output_dir / 'resplan_id_mapping_all.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if mapping:
            fieldnames = list(mapping[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mapping)
    print(f"\nSaved combined CSV mapping to: {csv_path}")
    print(f"  Total mappings: {len(mapping)}")

    # Split into train and test mappings
    train_mapping = [item for item in mapping if item.get('source') == 'train']
    test_mapping = [item for item in mapping if item.get('source') == 'test']

    # Save train-only mapping
    if train_mapping:
        train_csv_path = output_dir / 'resplan_id_mapping_train.csv'
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(train_mapping[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(train_mapping)
        print(f"Saved train-only CSV to: {train_csv_path}")
        print(f"  Train items: {len(train_mapping)}")

    # Save test-only mapping
    if test_mapping:
        test_csv_path = output_dir / 'resplan_id_mapping_test.csv'
        with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(test_mapping[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_mapping)
        print(f"Saved test-only CSV to: {test_csv_path}")
        print(f"  Test items: {len(test_mapping)}")

    # Save combined JSON for easy programmatic access
    json_path = output_dir / 'resplan_id_mapping_all.json'
    json_data = {
        'plan_to_id': {item['plan_name']: item['graph2plan_id'] for item in mapping},
        'id_to_plan': {item['graph2plan_id']: item['plan_name'] for item in mapping},
        'total_count': len(mapping)
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved combined JSON to: {json_path}")

    # Save train-only JSON
    if train_mapping:
        train_json_path = output_dir / 'resplan_id_mapping_train.json'
        train_json_data = {
            'plan_to_id': {item['plan_name']: item['graph2plan_id'] for item in train_mapping},
            'id_to_plan': {item['graph2plan_id']: item['plan_name'] for item in train_mapping},
            'id_to_train_index': {item['graph2plan_id']: item['source_index'] for item in train_mapping},
            'total_count': len(train_mapping)
        }
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_json_data, f, indent=2)
        print(f"Saved train JSON to: {train_json_path}")

    # Save test-only JSON
    if test_mapping:
        test_json_path = output_dir / 'resplan_id_mapping_test.json'
        test_json_data = {
            'plan_to_id': {item['plan_name']: item['graph2plan_id'] for item in test_mapping},
            'id_to_plan': {item['graph2plan_id']: item['plan_name'] for item in test_mapping},
            'id_to_test_index': {item['graph2plan_id']: item['source_index'] for item in test_mapping},
            'total_count': len(test_mapping)
        }
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_json_data, f, indent=2)
        print(f"Saved test JSON to: {test_json_path}")

    # Print sample mappings
    print(f"\nSample combined mappings (first 10):")
    print(f"  {'Plan Name':<15} -> {'Graph2plan ID':<15} {'Source':<8}")
    print(f"  {'-'*15}    {'-'*15} {'-'*8}")
    for item in mapping[:10]:
        source = item.get('source', 'N/A')
        print(f"  {item['plan_name']:<15} -> {item['graph2plan_id']:<15} {source:<8}")

    if len(mapping) > 10:
        print(f"  ... ({len(mapping) - 10} more)")

    # Print test set sample if available
    if test_mapping:
        print(f"\nSample TEST mappings (first 5):")
        print(f"  {'Plan Name':<15} -> {'Graph2plan ID':<15} {'Test Index':<10}")
        print(f"  {'-'*15}    {'-'*15} {'-'*10}")
        for item in test_mapping[:5]:
            print(f"  {item['plan_name']:<15} -> {item['graph2plan_id']:<15} {item['source_index']:<10}")
        if len(test_mapping) > 5:
            print(f"  ... ({len(test_mapping) - 5} more test items)")

    # Show how to find specific IDs
    print(f"\n" + "="*70)
    print("How to use the mapping:")
    print("="*70)
    print("FILES GENERATED:")
    print("  - resplan_id_mapping_all.csv/.json   (combined train+test)")
    print("  - resplan_id_mapping_train.csv/.json (train set only)")
    print("  - resplan_id_mapping_test.csv/.json  (test set only)")
    print("\n1. Look up by plan name in CSV:")
    print("   Search for 'plan_02420' in CSV to find Graph2plan ID '2957'")
    print("\n2. Programmatic access for TEST set (Python):")
    print("   import json")
    if test_mapping:
        print(f"   test_map = json.load(open('{output_dir}/resplan_id_mapping_test.json'))")
        print("   # Look up Graph2plan ID from plan name")
        print("   id = test_map['plan_to_id']['plan_XXXXX']")
        print("   # Look up plan name from Graph2plan ID")
        print("   plan = test_map['id_to_plan']['12345']")
        print("   # Look up test dataset index from Graph2plan ID")
        print("   test_idx = test_map['id_to_test_index']['12345']")
    print("="*70)

def main():
    # Generate mapping
    mapping = generate_mapping_from_source()

    if not mapping:
        print("\nERROR: Could not generate mapping. No source data found.")
        print("Please ensure one of these files exists:")
        print("  - ./data/data.mat")
        print("  - ../Interface/static/Data/data.mat")
        print("  - ./data_combined.pkl")
        print("  - ../Interface/static/Data/data_train_converted.pkl")
        print("  - ../Interface/static/Data/data_test_converted.pkl")
        return

    # Save mapping
    save_mapping(mapping, output_dir='./mapping_output')

    print("\nMapping generation complete!")

if __name__ == '__main__':
    main()

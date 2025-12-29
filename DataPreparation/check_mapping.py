import pickle
import os

# Skip checking original plan files (they have shapely dependencies)
print("Skipping original plan files (shapely dependencies)\n")
print("="*60 + "\n")

# Check the converted train data
train_path = r"../Interface/static/Data/data_train_converted.pkl"
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

print(f"Converted train data:")
print(f"  Keys: {list(train_data.keys())}")

nameList = train_data.get('nameList', [])
print(f"  NameList length: {len(nameList)}")
print(f"  First 20 names: {[str(n).strip() for n in nameList[:20]]}")

# Check if "2957" is in the list
target = "2957"
if any(str(n).strip() == target for n in nameList):
    idx = [i for i, n in enumerate(nameList) if str(n).strip() == target][0]
    print(f"\n  Found '{target}' at index {idx}")

    # Get the corresponding data item
    data_items = train_data['data']
    if idx < len(data_items):
        item = data_items[idx]
        print(f"  Item name: {item.name if hasattr(item, 'name') else 'N/A'}")

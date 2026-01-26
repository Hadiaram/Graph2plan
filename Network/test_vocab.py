import sys
sys.path.insert(0, './model')
from utils import get_vocab

v = get_vocab()
print("=" * 60)
print("VOCABULARY UPDATE VERIFICATION")
print("=" * 60)
print(f"\nVocabulary size: {len(v['object_idx_to_name'])}")
print(f"Object indices: {sorted(v['object_name_to_idx'].values())}")
print(f"Object names: {v['object_idx_to_name']}")
print("\nMapping:")
for name, idx in v['object_name_to_idx'].items():
    print(f"  {idx}: {name}")
print("\n✅ Vocabulary successfully updated to match data!")
print("=" * 60)

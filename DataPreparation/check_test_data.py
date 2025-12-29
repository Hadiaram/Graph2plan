import pickle

with open('../Interface/static/Data/data_test_converted.pkl', 'rb') as f:
    data = pickle.load(f)

print('Keys:', list(data.keys()))

# Check for different possible name list keys
for key in ['testNameList', 'nameList', 'names']:
    if key in data:
        names = data[key]
        print(f'{key}: type={type(names)}, length={len(names)}')
        print(f'  First 10: {[str(n).strip() for n in names[:10]]}')

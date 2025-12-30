"""
Test edge handling for various edge array shapes
"""
import numpy as np

print("="*70)
print("TESTING EDGE ARRAY HANDLING")
print("="*70)

# Simulate the fix
def handle_edges(rEdge):
    """Test version of edge handling logic"""
    # Convert to numpy array if needed
    if not isinstance(rEdge, np.ndarray):
        rEdge = np.array(rEdge)
    
    # Handle different edge array shapes
    if rEdge.ndim == 0 or (rEdge.ndim == 1 and len(rEdge) == 0):
        # Scalar or empty - no edges
        rEdge = np.array([]).reshape(0, 3)
    elif rEdge.ndim == 1:
        # 1D array - single edge, reshape to (1, 3)
        if len(rEdge) == 3:
            rEdge = rEdge.reshape(1, 3)
        else:
            # Invalid shape, skip edges
            rEdge = np.array([]).reshape(0, 3)
    # If ndim == 2, it's already correct shape
    
    return rEdge

# Test cases
test_cases = [
    ("Normal 2D array (3 edges)", np.array([[0, 1, 0], [1, 2, 1], [2, 3, 0]])),
    ("Single edge as 2D (1, 3)", np.array([[0, 1, 0]])),
    ("Single edge as 1D [3]", np.array([0, 1, 0])),
    ("Empty 2D array", np.array([]).reshape(0, 3)),
    ("Empty 1D array", np.array([])),
    ("Scalar (no edges)", np.int32(0)),
    ("List of edges", [[0, 1, 0], [1, 2, 1]]),
]

print("\nTest Results:")
print("-" * 70)

for name, test_data in test_cases:
    try:
        result = handle_edges(test_data)
        print(f"✅ {name}")
        print(f"   Input shape: {np.array(test_data).shape if isinstance(test_data, np.ndarray) else 'N/A'}")
        print(f"   Output shape: {result.shape}")
        print(f"   Can iterate: ", end="")
        count = 0
        for u, v, t in result:
            count += 1
        print(f"Yes ({count} edges)")
    except Exception as e:
        print(f"❌ {name}")
        print(f"   Error: {e}")
    print()

print("="*70)
print("EDGE HANDLING TEST COMPLETE")
print("="*70)
print("\nAll test cases should pass with ✅")
print("If any fail, the fix needs adjustment.")

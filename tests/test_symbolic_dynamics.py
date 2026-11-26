
# Test block entropy
# Tests:
# 1. Binary digitization
# 2. Non-overlapping block probabilities
# 3. Overlapping block probabilities (n_overlap=2)
# 4. Overlapping block probabilities (n_overlap=1)
# 5. Consistency between methods (n_overlap=0)
# 6. Handling NaN values in digitization and blocks
# 7. Invalid inputs and error handling


import numpy as np

from symbolic_dynamics import (digitization_binary, block_prob_no_overlap, block_prob_overlap)

#------------------------------
def check_error(test_name, func):

    try:
        func()
    except ValueError:
        print(f"{test_name}: Ok")
    else:
        raise AssertionError(f"{test_name}: Problem. Expected ValueError")
#------------------------------


# %% Test 1: Binary digitization

x1 = np.array([-3, 3, 0, 2, np.nan, np.inf, -np.inf, -5, 1])
threshold = 0

# Analytically:
# x <= 0 -> 0
# x > 0  -> 1
# NaN stays NaN
expected1 = np.array([0, 1, 0, 1, np.nan, 1, 0, 0, 1])

result1 = digitization_binary(x1, threshold)

if np.all(np.isclose(result1, expected1, equal_nan=True)):
    print("Test 1 - Digitization: Ok")
else:
    raise AssertionError(f"Expected {expected1}, got {result1}")
    
    
# %% Test 2: Non-overlapping blocks

x2 = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1])
block_length = 3

# Blocks: 010, 101, 001
# P(010) = 1/3
# P(101) = 1/3
# P(001) = 1/3
# All outcomes: 000, 001, 010, 011, 100, 101, 110, 111

p2_a = np.array([0, 1/3, 1/3, 0, 0, 1/3, 0, 0])

p2, bins2, patterns2 = block_prob_no_overlap(x2, block_length)

if np.all(np.isclose(p2, p2_a)):
    print("Test 2 - Non-overlapping blocks: Ok")
else:
    raise AssertionError(f"Expected {p2_a}, got {p2}")

# %% Test 3: Overlapping blocks (n_overlap=2)

x3 = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1])  # x3 == x2
block_length = 3
n_overlap = 2

# Blocks: 010, 101, 010, 101, 010, 100, 001
# P(010) = 3/7
# P(101) = 2/7
# P(100) = 1/7
# P(001) = 1/7
# All outcomes: 000, 001, 010, 011, 100, 101, 110, 111

p3_a = np.array([0, 1/7, 3/7, 0, 1/7, 2/7, 0, 0])

p3, bins3, patterns3 = block_prob_overlap(x3, block_length, n_overlap)

if np.all(np.isclose(p3, p3_a)):
    print("Test 3 - Overlapping blocks A (n_overlap=2): Ok")
else:
    raise AssertionError(f"Expected {p3_a}, got {p3}")
    
# %% Test 4: Overlapping blocks (n_overlap=1)

x4 = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1])  # x4 == x2
block_length = 3
n_overlap = 1

# Blocks: 010, 010, 010, 001
# P(010) = 3/4
# P(001) = 1/4
# All outcomes: 000, 001, 010, 011, 100, 101, 110, 111

p4_a = np.array([0, 1/4, 3/4, 0, 0, 0, 0, 0])

p4, bins4, patterns4 = block_prob_overlap(x4, block_length, n_overlap)

if np.all(np.isclose(p4, p4_a)):
    print("Test 4 - Overlapping blocks B (n_overlap=1): Ok")
else:
    raise AssertionError(f"Expected {p4_a}, got {p4}")
    

# %% Test 5: Compare block_prob_overlap(n_overlap=0) with block_prob_no_overlap

x5 = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1])   # x5 == x2
block_length = 3

# Blocks: 010, 101, 001
# P(010) = 1/3
# P(101) = 1/3
# P(001) = 1/3
# All outcomes: 000, 001, 010, 011, 100, 101, 110, 111

p5_a = np.array([0, 1/3, 1/3, 0, 0, 1/3, 0, 0])

p5, bins5, patterns5 = block_prob_no_overlap(x5, block_length)
p5b, bins5b, patterns5b = block_prob_overlap(x5, block_length, n_overlap=0)

if np.all(np.isclose(p5, p5_a)) and np.all(np.isclose(p5, p5b)):
    print("Test 5 - Non-overlapping blocks and overlapping blocks with n_overlap=0: Ok")
else:
    raise AssertionError("Non-overlapping and overlapping functions to give the same probability for n_overlap=0.")



# %% Test 6: Include NaN in blocks
print("-----------------")
print("Test 6 - Include NaN in blocks")

x6 = np.array([0, 1, 5, 8, 3, -2, np.nan, 3, 2, 2])
threshold = 1.5

expected6 = np.array([0, 0, 1, 1, 1, 0, np.nan, 1, 1, 1])
result6 = digitization_binary(x6, threshold)

if np.all(np.isclose(result6, expected6, equal_nan=True)):
    print("Test 6a - Digitization: Ok")
else:
    raise AssertionError(f"Expected {expected6}, got {result6}")

# non-overlapping blocks
block_length = 3

# Blocks: 001, 110, NaN11
# The blocks that contain np.nan or have length<block_length are discarded
# P(001) = 1/2
# P(110) = 1/2
# All outcomes: 000, 001, 010, 011, 100, 101, 110, 111
p6_a = np.array([0, 1/2, 0, 0, 0, 0, 1/2, 0])

p6, bins6, patterns6 = block_prob_no_overlap(result6, block_length)

if np.all(np.isclose(p6, p6_a)):
    print("Test 6b - Non-overlapping blocks: Ok")
else:
    raise AssertionError(f"Expected {p6_a}, got {p6}")


# overlapping blocks
block_length = 3
n_overlap = 2

# Blocks: 001, 011, 111, 110, 10NaN, 0NaN1, NaN11, 111
# The blocks that contain np.nan or have length<block_length are discarded
# P(001) = 1/5
# P(011) = 1/5
# P(111) = 2/5
# P(110) = 1/5
# All outcomes: 000, 001, 010, 011, 100, 101, 110, 111

p6b_a = np.array([0, 1/5, 0, 1/5, 0, 0, 1/5, 2/5])

p6b, bins6b, patterns6b = block_prob_overlap(result6, block_length, n_overlap)

if np.all(np.isclose(p6b, p6b_a)):
    print("Test 6c - Overlapping blocks: Ok")
else:
    raise AssertionError(f"Expected {p6b_a}, got {p6b}")


# %% Test 7: Check invalid inputs- Error handling

print("-----------------")
print("Test 7 - Invalid inputs")

# use x2 again

x7 = x2

# Invalid block_length
check_error("Test 7a - invalid block_length", lambda: block_prob_no_overlap(x7, block_length=0))
check_error("Test 7b - invalid block_length", lambda: block_prob_no_overlap(x7, block_length=2.5))

# Invalid overlap
check_error("Test 7c - invalid n_overlap", lambda: block_prob_overlap(x7, block_length=3, n_overlap=-1))
check_error("Test 7d - invalid n_overlap", lambda: block_prob_overlap(x7, block_length=3, n_overlap=3))


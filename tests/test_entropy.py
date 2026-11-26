
# Τest entropies
# Tests:
# 1. Uniform distribution (5 outcomes)
# 2. Uniform distribution (2 outcomes)
# 3. Deterministic distribution
# 4. Non-uniform distribution
# 4b. Renyi (a=1) and Tsallis (q=1) limits
# 5. Invalid probability distributions and error handling

# The analytical/expected results are calculated using a=2 (Renyi), q=2, k=1 (Tsallis). 
# Changing these parameters requires updating the expected results.

import numpy as np

from entropy import entropy_shannon, entropy_renyi, entropy_tsallis

#------------------------------
def check_message(test_name, entropy_name, H, H_norm, H_expected, N):

    if entropy_name == "Tsallis":
        H_max = k/(q-1) * (1 - N**(1-q))
        
    else:
        H_max = np.log(N)

    if not np.isclose(H, H_expected):
        raise AssertionError(f"{test_name}-{entropy_name}: Expected {H_expected}, got {H}")

    if not np.isclose(H_norm, H_expected/H_max):
        raise AssertionError(f"{test_name}-{entropy_name} normalized: Expected {H_expected/H_max}, got {H_norm}")

    print(f"{test_name}-{entropy_name}: Ok")
    

def check_error(test_name, entropy_name, func, p):

    try:
        func(p)
    except ValueError:
        print(f"{test_name}-{entropy_name}: Ok")
    else:
        raise AssertionError(f"{test_name}-{entropy_name}: Problem. Expected ValueError")
#------------------------------


base = np.e
a = 2
q = 2
k = 1

# %% Test 1: uniform distribution

p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
N1 = len(p1)

# Shannon analytically:
# H1 = - 5 * (1/5 * ln(1/5)) = ln5
H1_a = np.log(5) 

# Renyi analytically (a=2):
# H1 = - ln(5 * 0.2^2) = -ln5 -2 ln(0.2) = ln5
Hr1_a = np.log(5)

# Tsallis analytically (q=2):
# H1 = 1 * (1 - 5 * (0.2)^2 ) = 1 - 1/5 = 4/5
Ht1_a = 4/5

# calculated
H1 = entropy_shannon(p1, b=base)
Hr1 = entropy_renyi(p1, a=a)
Ht1 = entropy_tsallis(p1, q=q, k=k)

# Normalized
H1_norm = entropy_shannon(p1, b=base, norm=True)
Hr1_norm = entropy_renyi(p1, a=a, norm=True)
Ht1_norm = entropy_tsallis(p1, q=q, norm=True)


# Check
check_message("Test 1", "Shannon", H1, H1_norm, H1_a, N1)
check_message("Test 1", "Renyi", Hr1, Hr1_norm, Hr1_a, N1)
check_message("Test 1", "Tsallis", Ht1, Ht1_norm, Ht1_a, N1)

# %% Test 2: uniform distribution

p2 = np.array([0.5, 0.5])
N2 = len(p2)

# Shannon analytically:
# H2 = - 2 * (1/2 * ln(1/2)) = ln2
H2_a = np.log(2)

# Renyi analytically (a=2):
# H2 = - ln(2 * 0.5^2) = -ln2 -2 ln(1/2) = ln2
Hr2_a = np.log(2)

# Tsallis analytically (q=2):
# H2 = 1 * (1 - 2 * (0.5)^2 ) = 1 - 1/2 = 1/2
Ht2_a = 1/2

# calculated
H2 = entropy_shannon(p2, b=base)
Hr2 = entropy_renyi(p2, a=a)
Ht2 = entropy_tsallis(p2, q=q, k=k)

# Normalized
H2_norm = entropy_shannon(p2, b=base, norm=True)
Hr2_norm = entropy_renyi(p2, a=a, norm=True)
Ht2_norm = entropy_tsallis(p2, q=q, k=k, norm=True)


# Check
check_message("Test 2", "Shannon", H2, H2_norm, H2_a, N2)
check_message("Test 2", "Renyi", Hr2, Hr2_norm, Hr2_a, N2)
check_message("Test 2", "Tsallis", Ht2, Ht2_norm, Ht2_a, N2)

# %% Test 3: Deterministic distribution

p3 = np.array([0,0,0,1,0])
N3 = len(p3)

# Shannon analytically:
# H3 = 0 (lim_{x->0+} (x*lnx) =0 and 1*ln1=0)
H3_a = 0

# Renyi analytically (a=2):
# H3 = 0
Hr3_a = 0

# Tsallis analytically (q=2):
# H3 = 0
Ht3_a = 0

# calculated
H3 = entropy_shannon(p3, b=base)
Hr3 = entropy_renyi(p3, a=a)
Ht3 = entropy_tsallis(p3, q=q, k=k)

# Normalized
H3_norm = entropy_shannon(p3, b=base, norm=True)
Hr3_norm = entropy_renyi(p3, a=a, norm=True)
Ht3_norm = entropy_tsallis(p3, q=q, k=k, norm=True)


# Check
check_message("Test 3", "Shannon", H3, H3_norm, H3_a, N3)
check_message("Test 3", "Renyi", Hr3, Hr3_norm, Hr3_a, N3)
check_message("Test 3", "Tsallis", Ht3, Ht3_norm, Ht3_a, N3)


# %% Test 4: Non-uniform distribution

p4 = np.array([0.4, 0.2, 0.4])
N4 = len(p4)

# Shannon analytically:
# H4 = - 2 * (2/5 * ln(2/5)) - 1/5 * ln(1/5)
#    = ln5 - (4/5) * ln2
H4_a = np.log(5) - (4/5) * np.log(2)

# Renyi analytically (a=2):
# Hr4 = - ln(2 * (2/5)^2 + (1/5)^2)
#     = - ln(9/25) = ln(25/9)
Hr4_a = np.log(25/9)

# Tsallis analytically (q=2):
# Ht4 = 1 * (1 - 2 * (2/5)^2 - (1/5)^2)
#     = 1 - 9/25 = 16/25
Ht4_a = 16/25

# calculated
H4 = entropy_shannon(p4, b=base)
Hr4 = entropy_renyi(p4, a=a)
Ht4 = entropy_tsallis(p4, q=q, k=k)

# Normalized
H4_norm = entropy_shannon(p4, b=base, norm=True)
Hr4_norm = entropy_renyi(p4, a=a, norm=True)
Ht4_norm = entropy_tsallis(p4, q=q, k=k, norm=True)


# Check
check_message("Test 4", "Shannon", H4, H4_norm, H4_a, N4)
check_message("Test 4", "Renyi", Hr4, Hr4_norm, Hr4_a, N4)
check_message("Test 4", "Tsallis", Ht4, Ht4_norm, Ht4_a, N4)

# %% Test 4b: Renyi and Tsallis for a=1 and q=1 respectively

Hr4_ = entropy_renyi(p4, a=1)
Ht4_ = entropy_tsallis(p4, q=1, k=k)

if np.isclose(Hr4_, H4):
    print("Test 4b - Renyi with a=1: Equal to Shannon - Ok")
else:
    raise AssertionError(f"Test 4b - Renyi with a=1: Expected {H4}, got {Hr4_}")

if np.isclose(Ht4_, k*H4):
    print("Test 4b - Tsallis with q=1: Equal to k*Shannon - Ok")
else:
    raise AssertionError(f"Test 4b - Tsallis with q=1: Expected {k*H4}, got {Ht4_}")



# %% Test 5: Invalid inputs - Check Errors

p5_a = np.array([0.1, 0.3])
p5_b = np.array([0.9, 0.3, -0.2])
p5_c = np.array([0.5, 0.3, np.nan])

print("-"*25)
print("Error Check")
check_error("Test 5a", "Shannon", entropy_shannon, p5_a)
check_error("Test 5a", "Renyi", lambda p: entropy_renyi(p, a=a), p5_a)
check_error("Test 5a", "Tsallis", lambda p: entropy_tsallis(p, q=q, k=k), p5_a)
check_error("Test 5b", "Shannon", entropy_shannon, p5_b)
check_error("Test 5b", "Renyi", lambda p: entropy_renyi(p, a=a), p5_b)
check_error("Test 5b", "Tsallis", lambda p: entropy_tsallis(p, q=q, k=k), p5_b)
check_error("Test 5c", "Shannon", entropy_shannon, p5_c)
check_error("Test 5c", "Renyi", lambda p: entropy_renyi(p, a=a), p5_c)
check_error("Test 5c", "Tsallis", lambda p: entropy_tsallis(p, q=q, k=k), p5_c)




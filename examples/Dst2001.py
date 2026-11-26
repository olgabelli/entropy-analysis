
# Entropy analysis of the Dst 

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from entropy import entropy_shannon, entropy_renyi, entropy_tsallis

from symbolic_dynamics import digitization_binary, block_prob_no_overlap, block_prob_overlap



# %% 1. Load DST data from OMNI


project_folder = Path.cwd()
data_folder = project_folder / "data" / "dst"
filename = "omni_DST_2001.lst"

data = np.loadtxt(data_folder/filename)


with open(data_folder / "omni_DST_2001.fmt.txt", "r") as f:
    data_format = f.read()

print(data_format)
print("Data shape:", data.shape)

YEAR = data[:,0]
DAY  = data[:,1] 
HOUR = data[:,2]
DST =  data[:,3]

TIME = []

for y, d, h in zip(YEAR, DAY, HOUR):
    t = datetime.strptime(f"{int(y)} {int(d)} {int(h)}", "%Y %j %H")
    TIME.append(t)

TIME = np.array(TIME, dtype='datetime64[ms]')

# %% 2. Check there are no missing measurements

if np.any(np.isnan(DST)) or np.any(DST == 99999):
    raise ValueError("Missing Dst measurements found.")
else:
    print("No missing Dst measurements found.")

# %% 3. Check there are no missing time intervals (dt == 1h)

dt = np.diff(TIME)

if np.all(dt == np.timedelta64(1, 'h')):
    print("Time intervals: Ok (1 hour)")
else:
    raise ValueError("Irregular time intervals found.")
    
    
    
# %% 4. Moving window entropy

# Parameters
block_length = 2
n_overlap = 1

a_renyi = 0.5
q_tsallis = 1.8 

# Window
window_length = 256 #hrs  
step = 1   # moving by 1 hour each time

# Initialize results
time_entropy = []

# normalized Shannon entropy
H_norm_no = []
H_norm_o = []

# Renyi entropy
H_renyi_no = []
H_renyi_o = []

# Tsallis entropy
H_tsallis_no = []
H_tsallis_o = []


N = len(DST)


for start in range(0, N - window_length + 1, step):
    
    end = start + window_length
    
    # Associate each entropy value with the start of window
    time_entropy.append(TIME[start])    
    DST_window = DST[start:end]
    
    # A. digitization using the window mean
    threshold = np.mean(DST_window)
    DST_dig = digitization_binary(DST_window, threshold)
    
    # B. block probability
    p_no, bins_no, patterns_no = block_prob_no_overlap(DST_dig, block_length)
    p_o, bins_o, patterns_o = block_prob_overlap(DST_dig, block_length, n_overlap)

    # C. Normalized entropies
    # normalized Shannon
    H_norm_no.append(entropy_shannon(p_no, norm=True, n_cases=len(patterns_no)))
    H_norm_o.append(entropy_shannon(p_o, norm=True, n_cases=len(patterns_o)))
    
    # Renyi
    H_renyi_no.append(entropy_renyi(p_no, a_renyi, norm=True))
    H_renyi_o.append(entropy_renyi(p_o, a_renyi, norm=True))
    
    # Tsallis
    H_tsallis_no.append(entropy_tsallis(p_no, q_tsallis, norm=True))
    H_tsallis_o.append(entropy_tsallis(p_o, q_tsallis, norm=True))
    

# Convert to numpy arrays
time_entropy = np.array(time_entropy)

H_norm_no = np.array(H_norm_no)
H_norm_o = np.array(H_norm_o)

H_renyi_no = np.array(H_renyi_no)
H_renyi_o = np.array(H_renyi_o)

H_tsallis_no = np.array(H_tsallis_no)
H_tsallis_o = np.array(H_tsallis_o)



# %% 5. Plot Dst and entropy measures

fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex=True)

# Dst index
axs[0].plot(TIME[YEAR == 2001], DST[YEAR == 2001], linewidth=0.8)
axs[0].set_ylabel('Dst [nT]')
axs[0].grid(True)


# Normalized Shannon entropy
axs[1].plot(time_entropy, H_norm_no, linewidth=1.5, label='Non-overlapping')
axs[1].plot(time_entropy, H_norm_o, linewidth=1, label=f'Overlapping (n_overlap={n_overlap})')
axs[1].set_ylabel(r'$H_{\mathrm{Shannon}}$')
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(0.25, 1)


# Normalized Renyi entropy
axs[2].plot(time_entropy, H_renyi_no, linewidth=1.5, label='Non-overlapping')
axs[2].plot(time_entropy, H_renyi_o, linewidth=1, label=f'Overlapping (n_overlap={n_overlap})')
axs[2].set_ylabel(r'$H_{\mathrm{Renyi}}$')
axs[2].legend()
axs[2].grid(True)
axs[2].set_ylim(0.25, 1)


# Normalized Tsallis entropy
axs[3].plot(time_entropy, H_tsallis_no, linewidth=1.5, label='Non-overlapping')
axs[3].plot(time_entropy, H_tsallis_o, linewidth=1, label=f'Overlapping (n_overlap={n_overlap})')
axs[3].set_ylabel(r'$H_{\mathrm{Tsallis}}$')
axs[3].set_xlabel('Date')
axs[3].legend()
axs[3].grid(True)
axs[3].set_ylim(0.25, 1)


fig.suptitle(f'Dst (2001) and normalized block entropies\n' 
             f'Block length = {block_length}, Window length = {window_length} h')

plt.tight_layout()
plt.show()

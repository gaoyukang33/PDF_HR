import sys
sys.path.append(".")

import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce

import tools.util.plot_util as PlotUtil


import sys
import numpy as np
import os
import matplotlib.pyplot as plt


files = [
    [
        "experiment/output_amp_walk_1/log.txt",
        "experiment/output_amp_walk_2/log.txt",
        "experiment/output_amp_walk_3/log.txt"
    ],
    [
        "experiment/output_amp_run_1/log.txt",
        "experiment/output_amp_run_2/log.txt",
        "experiment/output_amp_run_3/log.txt"
    ],
    [
        "experiment/output_amp_spinkick_1/log.txt",
        "experiment/output_amp_spinkick_2/log.txt",
        "experiment/output_amp_spinkick_3/log.txt"
    ],
    [
        "experiment/output_amp_cartwheel_1/log.txt",
        "experiment/output_amp_cartwheel_2/log.txt",
        "experiment/output_amp_cartwheel_3/log.txt"
    ],
]

labels = [
    'walk',
    'run',
    'spinkick',
    'cartwheel'
]

assert len(files) == len(labels), "Files and labels length mismatch"

# style
DRAW_INDIVIDUAL_LINES = True  # draw every individual line
DRAW_MEAN_AND_BAND = False      # draw mean line with std band
BAND_ALPHA = 0.2               # band transparency
INDIVIDUAL_ALPHA = 1           # inidividual line transparency
LINE_WIDTH_MEAN = 2            # mean line width
LINE_WIDTH_INDIVIDUAL = 1      # individual line width

X_LIMIT = [0, 8e9]             # x axis range

# data keys
x_key = "Samples"
y_key = "Test_Episode_Length"
plot_title = "Performance Comparison"

plt.figure(figsize=(8, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

loaded_data_groups = []
global_min_max_x = float('inf')
# ===========================================
# load all data
for group_idx, file_group in enumerate(files):
    group_data = []

    for file_path in file_group:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            with open(file_path, "r") as f:
                clean_lines = [line.replace(",", "\t") for line in f]
                
            data = np.genfromtxt(clean_lines, names=True, dtype=None, encoding=None)

            if x_key in data.dtype.names and y_key in data.dtype.names:
                raw_xs = data[x_key]
                raw_ys = data[y_key]
                group_data.append({'x': raw_xs, 'y': raw_ys})

                if len(raw_xs) > 0:
                    current_max_x = raw_xs[-1]
                    if current_max_x < global_min_max_x:
                        global_min_max_x = current_max_x
            else:
                print(f"Key error in {file_path}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    loaded_data_groups.append(group_data)

valid_max_x = min(X_LIMIT[1], global_min_max_x)
valid_min_x = X_LIMIT[0]
print(f"Global plotting range: {valid_min_x:.1f} to {valid_max_x:.1f} (limited by shortest experiment)")

# filter and align data
for group_idx, group_data in enumerate(loaded_data_groups):
    if not group_data:
        continue

    color = colors[group_idx % len(colors)]
    current_label = labels[group_idx]
    
    filtered_xs_list = []
    filtered_ys_list = []
    
    # suppose every run in the group has data within valid range
    group_min_len = float('inf')

    for run in group_data:
        raw_x = run['x']
        raw_y = run['y']
        
        # filter by valid x range
        mask = (raw_x >= valid_min_x) & (raw_x <= valid_max_x)
        
        if not np.any(mask):
            continue
            
        xs = raw_x[mask]
        ys = raw_y[mask]
        
        filtered_xs_list.append(xs)
        filtered_ys_list.append(ys)
        
        if len(xs) < group_min_len:
            group_min_len = len(xs)

    if not filtered_xs_list:
        print(f"Group {current_label} has no data in valid range.")
        continue

    # truncate all runs to the shortest length in the group
    final_xs = [x[:group_min_len] for x in filtered_xs_list]
    final_ys = [y[:group_min_len] for y in filtered_ys_list]
    
    np_ys = np.array(final_ys)
    
    # suppose all x are the same within the group
    plot_x = final_xs[0]
    
    # statistics
    mean_y = np.mean(np_ys, axis=0)
    min_y = np.min(np_ys, axis=0)
    max_y = np.max(np_ys, axis=0)

    # plotting
    # single lines
    if DRAW_INDIVIDUAL_LINES:
        for i in range(len(final_ys)):
            plt.plot(plot_x, np_ys[i], 
                     color=color, 
                     alpha=INDIVIDUAL_ALPHA, 
                     linewidth=LINE_WIDTH_INDIVIDUAL,
                     linestyle='-')

    # mean and band
    if DRAW_MEAN_AND_BAND:
        plt.plot(plot_x, mean_y, 
                 color=color, 
                 label=current_label, 
                 linewidth=LINE_WIDTH_MEAN)
        
        plt.fill_between(plot_x, 
                         min_y, 
                         max_y, 
                         color=color, 
                         alpha=BAND_ALPHA, 
                         edgecolor=None)



# save and decorate
ax = plt.gca()
plt.xlabel(x_key)
plt.ylabel(y_key)
plt.title(plot_title)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid(linestyle='dotted', alpha=0.5)
plt.legend()
plt.tight_layout()

output_name = "plot_result.png"
plt.savefig(output_name, dpi=300)
print(f"Saved plot to {output_name}")

# plt.show()
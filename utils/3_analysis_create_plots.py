import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

TEMP_COLS = ['Temp_CH0', 'Temp_Laser_Core', 'Temp_Chamber', 'Temp_Ambient']
AREA_COLS = ['Area_Trigger', 'Area_PD_Ext', 'Area_Laser_A', 'Area_PD_Int']


# Y-axis limits for 2x2 grid plots (edit as needed)
Y_LIMS_4X = [
    (450, 550),  # Plot 1
    (330, 500),  # Plot 2
    (110, 115),  # Plot 3
    (110, 160),  # Plot 4
]

# Select file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Processed Area CSV File", filetypes=[("CSV Files", "*.csv")])
if not file_path or not os.path.basename(file_path).startswith('processed_area'):
    print("File must start with 'processed_area'.")
    exit(1)


# Read CSV
df = pd.read_csv(file_path)

# Convert SampleTime_s to minutes for plotting
df['SampleTime_min'] = df['SampleTime_s'] / 60.0


# Chart title from filename
chart_title = os.path.basename(file_path)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)


# Top plot: Temperatures vs SampleTime_min
for col in TEMP_COLS:
    if col in df.columns:
        ax1.plot(df['SampleTime_min'], df[col], label=col)
ax1.set_ylabel('Temperature (C)')
ax1.set_title(f'Temperature Channels vs Sample Time\n{chart_title}')
ax1.legend()
ax1.grid(True)

# Bottom plot: Area columns vs SampleTime_min
for col in AREA_COLS:
    if col in df.columns:
        ax2.plot(df['SampleTime_min'], df[col], label=col)
ax2.set_ylabel('Area')
ax2.set_xlabel('Sample Time (min)')
ax2.set_title(f'Area Channels vs Sample Time\n{chart_title}')
ax2.legend()
ax2.grid(True)


plt.tight_layout()

# Save 2-level plot
output_name = f"plot_{os.path.basename(file_path).replace('.csv', '.png')}"
output_path = os.path.join(os.path.dirname(file_path), output_name)
plt.savefig(output_path)
plt.close(fig)
print(f"Plot saved to {output_path}")

# --- 2x2 grid plot: Temp_Laser_Core, Temp_Chamber, and one Area column each ---
TEMP_CORE = 'Temp_Laser_Core'
TEMP_CHAMBER = 'Temp_Chamber'
fig2, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig2.suptitle(chart_title, fontsize=16)
axs = axs.flatten()
for i, area_col in enumerate(AREA_COLS):
    ax = axs[i]
    if TEMP_CORE in df.columns:
        ax.plot(df['SampleTime_min'], df[TEMP_CORE], label=TEMP_CORE, color='tab:blue')
    if TEMP_CHAMBER in df.columns:
        ax.plot(df['SampleTime_min'], df[TEMP_CHAMBER], label=TEMP_CHAMBER, color='tab:orange')
    if area_col in df.columns:
        ax2 = ax.twinx()
        area_line, = ax2.plot(df['SampleTime_min'], df[area_col], color='tab:green', label=area_col)
        ax2.set_ylabel(area_col)
        # Set y-limits for area axis if specified
        if i < len(Y_LIMS_4X) and Y_LIMS_4X[i] is not None:
            ax2.set_ylim(*Y_LIMS_4X[i])
        # Add legend for the area line only
        ax2.legend([area_line], [area_col], loc='upper right')
    ax.set_title(f"{TEMP_CORE}, {TEMP_CHAMBER} & {area_col}")
    ax.set_xlabel('Sample Time (min)')
    ax.set_ylabel('Temperature (C)')
    ax.legend(loc='upper left')
    ax.grid(True)
plt.tight_layout()
output_name_4x = f"plot_4x_{os.path.basename(file_path).replace('.csv', '.png')}"
output_path_4x = os.path.join(os.path.dirname(file_path), output_name_4x)
plt.savefig(output_path_4x)
plt.close(fig2)
print(f"4x plot saved to {output_path_4x}")

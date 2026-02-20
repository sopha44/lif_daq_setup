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
    (330, 550),  # Plot 2
    (110, 120),  # Plot 3
    (110, 180),  # Plot 4
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
    # Data summary overlay logic
    temp_chamber = df[TEMP_CHAMBER] if TEMP_CHAMBER in df.columns else None
    start_temp = temp_chamber.iloc[0] if temp_chamber is not None else None
    end_temp = temp_chamber.iloc[-1] if temp_chamber is not None else None
    overlay_stats = None
    if area_col in df.columns:
        area_vals = df[area_col].values
        start_area = area_vals[0]
        end_area = area_vals[-1]
        slope = (end_area - start_area) / (end_temp - start_temp) if (end_temp is not None and start_temp is not None and (end_temp - start_temp) != 0) else float('nan')
        perc_inc_per_T = (slope / start_area * 100) if start_area != 0 else float('nan')
        overlay_stats = {
            'min': area_vals.min(),
            'max': area_vals.max(),
            'mean': area_vals.mean(),
            'std': area_vals.std(),
            'cv': area_vals.std() / area_vals.mean() if area_vals.mean() != 0 else float('nan'),
            'start': start_area,
            'end': end_area,
            'start_T': start_temp,
            'end_T': end_temp,
            'slope': slope,
            '%inc/T': perc_inc_per_T
        }
    if area_col in df.columns:
        ax2 = ax.twinx()
        area_line, = ax2.plot(df['SampleTime_min'], df[area_col], color='tab:green', label=area_col)
        ax2.set_ylabel(area_col)
        # Set y-limits for area axis if specified
        if i < len(Y_LIMS_4X) and Y_LIMS_4X[i] is not None:
            ax2.set_ylim(*Y_LIMS_4X[i])
        # Add legend for the area line only
        ax2.legend([area_line], [area_col], loc='upper right')
        # Overlay summary box
        if overlay_stats is not None:
            if i in [0, 2]:
                # Full stats for Area_Trigger and Area_Laser_A
                stat_text = (
                    f"min={overlay_stats['min']:.2f}\nmax={overlay_stats['max']:.2f}\nmean={overlay_stats['mean']:.2f}\nstd={overlay_stats['std']:.2f}\nCV={overlay_stats['cv']:.2f}"
                    f"\nstart={overlay_stats['start']:.2f}\nend={overlay_stats['end']:.2f}\nstart_T={overlay_stats['start_T']:.2f}\nend_T={overlay_stats['end_T']:.2f}\nslope(area/deltaT)={overlay_stats['slope']:.2f}\n%inc/T={overlay_stats['%inc/T']:.4f}"
                )
            else:
                # Only show start/end values and slope for Area_PD_Ext and Area_PD_Int
                stat_text = (
                    f"start={overlay_stats['start']:.2f}\nend={overlay_stats['end']:.2f}\nstart_T={overlay_stats['start_T']:.2f}\nend_T={overlay_stats['end_T']:.2f}\nslope(area/deltaT)={overlay_stats['slope']:.2f}\n%inc/T={overlay_stats['%inc/T']:.4f}"
                )
            ax2.text(0.98, 0.02, stat_text, transform=ax2.transAxes, fontsize=8, color='green', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
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

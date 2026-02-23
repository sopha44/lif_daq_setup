import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

# --- User editable axis limits ---
# Set axis limits for each plot column (None for auto)
# Example: (xmin, xmax, ymin, ymax, ymin2, ymax2)
AXES_LIMITS = [
    (0, 120, 21, 37, 450, 550),  # Plot 1
    (0, 120, 21, 37, 60, 275),  # Plot 2
    (0, 120, 21, 37, 90, 110),  # Plot 3
    (0, 120, 21, 37, 45, 75),  # Plot 4
]

# --- File selection dialog ---
def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = []
    for i in range(3):
        file_path = filedialog.askopenfilename(
            title=f"Select processed_area_merged CSV file #{i+1}",
            filetypes=[("CSV Files", "processed_area_merged*.csv")]
        )
        if file_path:
            file_paths.append(file_path)
    return file_paths

# --- Plotting function (adapted from 3_analysis_create_plots) ---
def plot_comparison(files):
    TEMP_COLS = ['Temp_Laser_Core', 'Temp_Chamber']
    TEMP_COLORS = ['tab:blue', 'tab:orange']
    AREA_COLS = ['Area_Trigger', 'Area_PD_Ext', 'Area_Laser_A', 'Area_PD_Int']
    n_files = len(files)
    n_plots = len(AREA_COLS)
    # Add an extra row for filename labels
    # Add two extra rows: one for each filename label above each row of plots
    fig, axes = plt.subplots(n_files * 2, n_plots, figsize=(5 * n_plots, 2.5 * n_files * 2), gridspec_kw={'height_ratios': [0.3, 1] * n_files})
    if n_files == 1:
        axes = axes.reshape(n_files * 2, n_plots)
    for row, file in enumerate(files):
        df = pd.read_csv(file)
        fname = os.path.basename(file)
        x = df['SampleTime_s'] / 60.0  # Convert seconds to minutes
        # Add filename label row (spanning all columns)
        for col in range(n_plots):
            ax_label = axes[row * 2, col]
            ax_label.axis('off')
        # Center filename label above the row (span all columns)
        axes[row * 2, 0].text(
            0.5, 0.5, fname, ha='center', va='center', fontsize=12, fontweight='bold', transform=axes[row * 2, 0].transAxes
        )
        # Precompute overlay data
        temp_chamber = df['Temp_Chamber'] if 'Temp_Chamber' in df.columns else None
        start_temp = temp_chamber.iloc[0] if temp_chamber is not None else None
        end_temp = temp_chamber.iloc[-1] if temp_chamber is not None else None
        # For Area_Laser_A and Area_Trigger, compute stats
        overlay_stats = {}
        for stat_col in ['Area_Laser_A', 'Area_Trigger']:
            if stat_col in df.columns:
                vals = df[stat_col].values
                overlay_stats[stat_col] = {
                    'min': np.min(vals),
                    'max': np.max(vals),
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'cv': np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else np.nan
                }
            else:
                overlay_stats[stat_col] = None
        # Plot row
        for col, area_col in enumerate(AREA_COLS):
            ax = axes[row * 2 + 1, col]
            lines = []
            labels = []
            # Plot Temperature (left y-axis) with different colors
            for t_idx, temp_col in enumerate(TEMP_COLS):
                if temp_col in df.columns:
                    l, = ax.plot(x, df[temp_col], label=temp_col, color=TEMP_COLORS[t_idx])
                    lines.append(l)
                    labels.append(temp_col)
            # Overlay start/end Temp_Chamber as orange dots (to match Temp_Chamber color)
            if temp_chamber is not None:
                ax.scatter(x.iloc[0], start_temp, color='tab:orange', marker='o', s=40, label='Start Temp_Chamber')
                ax.scatter(x.iloc[-1], end_temp, color='tab:orange', marker='x', s=40, label='End Temp_Chamber')
            # Restore left Y-axis label
            ax.set_ylabel('Temperature (C)')
            ax.set_xlabel('Time (min)')
            # Remove subplot title for less clutter
            ax2 = ax.twinx()
            # Plot Area (right y-axis) in green
            if area_col in df.columns:
                l2, = ax2.plot(x, df[area_col], label=area_col, color='tab:green')
                lines.append(l2)
                labels.append(area_col)
                # Overlay start/end points for Area as green dots (to match Area color)
                start_area = df[area_col].iloc[0]
                end_area = df[area_col].iloc[-1]
                ax2.scatter(x.iloc[0], start_area, color='tab:green', marker='o', s=40, label='Start '+area_col)
                ax2.scatter(x.iloc[-1], end_area, color='tab:green', marker='x', s=40, label='End '+area_col)
                slope = (end_area - start_area) / (end_temp - start_temp) if (end_temp - start_temp) != 0 else np.nan
                perc_inc_per_T = (slope / start_area * 100) if start_area != 0 else np.nan
                # Overlay summary for each plot
                if col in [0, 2]:
                    # Full stats for Area_Trigger and Area_Laser_A
                    if area_col in overlay_stats and overlay_stats[area_col] is not None:
                        stats = overlay_stats[area_col]
                        stat_text = (
                            f"min={stats['min']:.2f}\nmax={stats['max']:.2f}\nmean={stats['mean']:.2f}\nstd={stats['std']:.2f}\nCV={stats['cv']:.2f}"
                            f"\nstart={start_area:.2f}\nend={end_area:.2f}\nstart_T={start_temp:.2f}\nend_T={end_temp:.2f}\nslope(area/deltaT)={slope:.2f}\n%inc/T={perc_inc_per_T:.4f}"
                        )
                        ax2.text(0.98, 0.02, stat_text, transform=ax2.transAxes, fontsize=8, color='green', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                elif col in [1, 3]:
                    # Only show start/end values and slope for Area_PD_Ext and Area_PD_Int
                    stat_text = (
                        f"start={start_area:.2f}\nend={end_area:.2f}\nstart_T={start_temp:.2f}\nend_T={end_temp:.2f}\nslope(area/deltaT)={slope:.2f}\n%inc/T={perc_inc_per_T:.4f}"
                    )
                    ax2.text(0.98, 0.02, stat_text, transform=ax2.transAxes, fontsize=8, color='green', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            # Restore right Y-axis label
            ax2.set_ylabel(f'{area_col}')
            # Optionally set axis limits
            if AXES_LIMITS[col]:
                ax.set_xlim(AXES_LIMITS[col][0], AXES_LIMITS[col][1])
                ax.set_ylim(AXES_LIMITS[col][2], AXES_LIMITS[col][3])
                ax2.set_ylim(AXES_LIMITS[col][4], AXES_LIMITS[col][5])
            # Add legend to each subplot, bottom left
            ax.legend(lines, labels, loc='lower left', fontsize=8, markerscale=0.7, handlelength=1, borderaxespad=0.2, labelspacing=0.2)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Save to /data/1_plots with timestamp
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data', '1_plots')
    os.makedirs(outdir, exist_ok=True)
    dt_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outname = f"3_run_comparison_{dt_str}.png"
    outfile = os.path.join(outdir, outname)
    plt.savefig(outfile)
    print(f"Saved plot to {outfile}")
    plt.show()

if __name__ == "__main__":
    files = select_files()
    if len(files) < 1:
        print("No files selected.")
    else:
        plot_comparison(files)

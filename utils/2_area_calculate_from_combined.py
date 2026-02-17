import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

# User variables
TRIGGER_THRESHOLD = 0.00  # Remove rows where Trigger < this value
N_KEEP = 5  # Number of lagging rows to keep after first drop below threshold
AREA_COLS = ['Trigger', 'PD_Ext', 'Laser_A', 'PD_Int']  # Columns to calculate area for
TEMP_COLS = ['Temp_CH0', 'Temp_Laser_Core', 'Temp_Chamber', 'Temp_Ambient']  # Temp columns to keep

# Select CSV file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Combined CSV File", filetypes=[("CSV Files", "*.csv")])
if not file_path or not os.path.basename(file_path).startswith('merged'):
    print("File must start with 'merged'.")
    exit(1)

# Read CSV
df = pd.read_csv(file_path)

# Group by batch (contiguous blocks with same Trigger_or_N)
df['__batch_id'] = (df['Trigger_or_N'] != df['Trigger_or_N'].shift()).cumsum()

output_rows = []
for batch_idx, (batch_id, batch) in enumerate(df.groupby('__batch_id')):
    # Find first index where Trigger drops below threshold
    below_idx = batch[batch['Trigger'] < TRIGGER_THRESHOLD].index
    if len(below_idx) > 0:
        first_below = below_idx[0]
        keep_idxs = list(batch.index[batch.index <= first_below]) + list(batch.index[batch.index > first_below][:N_KEEP])
        batch = batch.loc[keep_idxs]
    else:
        batch = batch.copy()

    # Area calculation (Riemann sum)
    resolution = batch['Resolution_us'].iloc[0] if 'Resolution_us' in batch.columns else 1
    area_results = {}
    for col in AREA_COLS:
        if col in batch.columns:
            area = np.sum(batch[col].values * resolution)
            area_results[f'Area_{col}'] = area
        else:
            area_results[f'Area_{col}'] = np.nan

    # Temp values (take first row in this batch)
    temp_values = {col: batch[col].iloc[0] if col in batch.columns else np.nan for col in TEMP_COLS}

    # Compose output row
    out_row = {col: batch[col].iloc[0] for col in batch.columns if col not in AREA_COLS + TEMP_COLS + ['__batch_id', 'BatchTime_us']}
    trigger_time_s = batch['TriggerTime_s'].iloc[0] if 'TriggerTime_s' in batch.columns else 1
    out_row['SampleTime_s'] = trigger_time_s * (batch_idx + 1)
    out_row.update(area_results)
    # Move SampleTime_s to first column, Temp columns to end
    sample_time = out_row.pop('SampleTime_s')
    out_row = {'SampleTime_s': sample_time, **out_row, **temp_values}
    output_rows.append(out_row)

# Output CSV
output_name = f"processed_area_{os.path.basename(file_path)}"
output_path = os.path.join(os.path.dirname(file_path), output_name)
df_out = pd.DataFrame(output_rows)
df_out.to_csv(output_path, index=False)
print(f"Output written to {output_path}")

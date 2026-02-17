import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Constants
SAMPLE_RATE_HZ = 1_000_000  # 1 MHz
US_PER_SAMPLE = 1_000_000 / SAMPLE_RATE_HZ  # microseconds per sample (should be 1)
TRIGGER_TIME_S = 30  # seconds

# Assumes board0 is temperature DAQ and board1 is data DAQ

# Additional experiment metadata (edit as needed)
LASER_ON_TIME_US = 100  # microseconds
POWER_W = 0.005  # Watts
SLOPE_W_A = 0.6559  # W/A

# Channel mapping for relabeling
DATA_DAQ_CHANNELS = {
    'CH0': 'Trigger',  # e.g., 'Voltage', customize as needed
    'CH1': 'PD_Ext',
    'CH2': 'Laser_A',
    'CH3': 'PD_Int',
}
TEMP_DAQ_CHANNELS = {
    'CH0': 'Temp_CH0',  # e.g., 'Ambient_Temp', customize as needed
    'CH1': 'Temp_Laser_Core',
    'CH2': 'Temp_Chamber',
    'CH3': 'Temp_Ambient',
}

# Open dialog to select folder
def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Data Folder")
    return folder_selected

def main():
    folder = select_folder()
    if not folder:
        print("No folder selected.")
        return

    board0_path = os.path.join(folder, "board0_USB_TEMP_AI.csv")  # Temperature DAQ
    board1_path = os.path.join(folder, "board1_USB_1604HS_2AO.csv")  # Data DAQ
    if not (os.path.exists(board0_path) and os.path.exists(board1_path)):
        print("Required CSV files not found in selected folder.")
        return

    # Read CSVs
    df_temp = pd.read_csv(board0_path)
    df_data = pd.read_csv(board1_path)

    # Relabel columns for clarity
    data_col_map = {col: DATA_DAQ_CHANNELS.get(col, col) for col in df_data.columns}
    temp_col_map = {col: TEMP_DAQ_CHANNELS.get(col, col) for col in df_temp.columns}
    df_data = df_data.rename(columns=data_col_map)
    df_temp = df_temp.rename(columns=temp_col_map)

    # Ensure all expected TEMP_DAQ_CHANNELS columns exist in df_temp, even if missing from file
    for expected_col in TEMP_DAQ_CHANNELS.values():
        if expected_col not in df_temp.columns:
            df_temp[expected_col] = None

    # Assume 'Trigger_or_N' column exists in data DAQ
    if 'Trigger_or_N' not in df_data.columns:
        print("'Trigger_or_N' column not found in board1_USB_1604HS_2AO.csv")
        return

    # Exclude 'Timestamp' if present in temp DAQ, but keep all expected temp columns in order
    temp_cols = [col for col in TEMP_DAQ_CHANNELS.values() if col != 'Timestamp']
    # Identify the original timestamp column from board1 (data DAQ)
    orig_timestamp_col = None
    for col in df_data.columns:
        if 'timestamp' in col.lower():
            orig_timestamp_col = col
            break
    # Exclude 'Timestamp' and 'Trigger_or_N' from data DAQ columns
    data_cols = [col for col in df_data.columns if col not in ['Timestamp', 'Trigger_or_N', orig_timestamp_col]]

    # Assign batch index for contiguous trigger batches
    df_data = df_data.copy()
    df_data['__batch_id'] = (df_data['Trigger_or_N'] != df_data['Trigger_or_N'].shift()).cumsum()
    # Assign BatchTime_us within each batch
    df_data['BatchTime_us'] = df_data.groupby('__batch_id').cumcount() * US_PER_SAMPLE

    # Merge temp DAQ values by Trigger_or_N
    temp_batch = df_temp.drop_duplicates(subset=['Trigger_or_N'])
    temp_batch = temp_batch[['Trigger_or_N'] + temp_cols]
    df_data = pd.merge(df_data, temp_batch, on='Trigger_or_N', how='left', suffixes=('', '_temp'))

    # Compose output rows
    output_rows = []
    for idx, row in df_data.iterrows():
        output_row = {
            'BatchTime_us': row['BatchTime_us'],
            'Timestamp_us': row[orig_timestamp_col] if orig_timestamp_col else None,
            'Resolution_us': US_PER_SAMPLE,
            'SampleRate_Hz': SAMPLE_RATE_HZ,
            'TriggerTime_s': TRIGGER_TIME_S,
            'LaserOnTime_us': LASER_ON_TIME_US,
            'Power_W': POWER_W,
            'Slope_W_A': SLOPE_W_A,
            'Trigger_or_N': row['Trigger_or_N'],
            **{col: row[col] for col in data_cols},
            **{col: row[col] for col in temp_cols}
        }
        output_rows.append(output_row)

    # Write output CSV with name matching the selected folder
    folder_name = os.path.basename(os.path.normpath(folder))
    output_path = os.path.join(folder, f"merged_{folder_name}.csv")
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main()

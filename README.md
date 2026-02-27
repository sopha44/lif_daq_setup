# LIF DAQ Setup

Multi-device data acquisition system for Measurement Computing DAQ boards with (WIP) real-time signal processing capabilities.

Features to Add:
-Signal processing similar to DASYLab (Simulink like, but will be slower through Python than C++)
-Merge in LabJack usage

## Supported Hardware

- **USB-1608GX-2AO**: 8 analog input channels
- **USB-1604HS-2AO**: 4 analog input channels (high-speed, up to 1.33 MHz)
- **USB-TEMP**: Thermocouple/RTD temperature measurement device (2Hz). MCC library does not do background scan for USB-TEMP so the scan is done 'like background' using a seperate processing thread.
- Other MCC DAQ devices compatible with the Universal Library

## Installation

### Prerequisites

1. **Python 3.7+** (recommended: Python 3.9 or later)
2. **InstaCal** - Measurement Computing's configuration utility
   - Download from [Measurement Computing's website](https://www.mccdaq.com/Software-Downloads)
   - Install and run to configure your DAQ boards

### Python Dependencies

Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```

This includes:
- `mcculw` - MCC Universal Library Python wrapper
- `pandas` - Data manipulation and CSV export
- `numpy` - Numerical processing

### Hardware Configuration

#### Measurement Computing Devices
**Important**: Before running the acquisition software, you must configure your DAQ boards using InstaCal.

1. **Launch InstaCal** (installed with MCC drivers)
2. **Add/Configure each DAQ device**:
   - Connect your DAQ hardware via USB
   - Click "Add Device" or use auto-detection
   - Assign a **board number** to each device (e.g., Board 0, Board 1, Board 2)
   - Test the device to ensure proper communication
3. **Note the board numbers** - these are used in configuring the DAQs in Python

![InstaCAL example](assets/instacal.png)

#### LabJack Devices

WIP

## Usage

### Basic Acquisition

1. **Configure your DAQ setup** in `acquisition_controller.py` by editing the `main()` function:

```python
def main():
    # Create controller
    controller = AcquisitionController()
    
    # Set acquisition duration
    duration = 10.0  # seconds
    
    # Board 0: USB-TEMP-AI
    # Slave device - no trigger, will be measured when Board 1 triggers
    controller.setup_daq(
        board_num=0,
        name="USB-TEMP-AI",
        sample_rate=1,  # 1 Hz
        low_chan=1,
        high_chan=3,
        enable_processing=False #PLACEHOLDER - does not work
    )
    
    # Board 1: USB-1604HS-2AO
    # Master trigger device - monitors CH0 for trigger signal
    controller.setup_daq(
        board_num=1,
        name="USB-1604HS-2AO",
        sample_rate=1_000_000,  # 1 MHz
        low_chan=0,
        high_chan=3,  # Read CH0, CH1, CH2, CH3 (4 channels)
        trigger_channel=0,  # Monitor CH0 for trigger
        trigger_voltage_high=1.0,  # Rising edge at 1V
        # trigger_voltage_low=1.0,   # Falling edge (not used in time-gated mode)
        acquisition_window_us=300,  # 300 µs per channel
        enable_processing=False #PLACEHOLDER - does not work
    )

    # Start acquisition on all configured DAQs
    controller.start_acquisition(
            restart_scan_each_trigger=True,  # Stops/Starts scan to ensure buffer clears
            check_buffer_every=1000,  # 0 = disabled, >0 = check buffer every N cycles, logs buffer fill
            trigger_check_decimation=10,  # Check every 10th sample = 100,000 checks/sec
            time_between_points=30.0,  # Wait 30 seconds between measurement cycles
            total_duration_minutes=120  # Run for 120 minutes, 2 hours (None = run indefinitely)
        )
```

2. **Run the acquisition**:

```bash
python acquisition_controller.py
```

3. **Data output**: Results are saved in timestamped folders:
```
data/
  └── 2026-02-06_11-09-26/
      ├── Board1_USB-TEMP.csv
      └── Board2_USB-1604HS-2AO.csv
```

### Signal Processing

WIP

### Performance Benchmarking

Note: Data can be lost during DAQ scans/reads due to PC processing speed. Benchmarking will help understand % data lost at higher sampling rates. Also note that Python is much slower than using C/C++ based software. TBD how Python signal processing at high speed compares with DASYLab.

Test your system's maximum acquisition rate:

```bash
python utils/benchmark_daq_performance.py
```

This runs tests from 1 kHz to 1.33 MHz and reports data loss percentages.

## Project Structure


```
lif_daq_setup/
├── acquisition_controller.py      # Main acquisition orchestrator
├── app_gui.py                    # GUI application (PLACEHOLDER)
├── config.py                     # Configuration settings
├── device_manager.py             # Device lifecycle management
├── signal_processor.py           # Real-time signal processing (PLACEHOLDER)
├── 1_slow_capture.py             # Primary acquisition script (slow capture)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── hardware/                     # Device-specific implementations
│   ├── mcc_device.py             # Analog input device class (USB-1604)
│   ├── mcc_temperature_device.py # Temperature device class (USB-TEMP)
│   └── __init__.py               # Hardware package init
├── utils/                        # Utility scripts for analysis, setup, and diagnostics
│   ├── 1_analysis_combine_csv.py           # Combine DAQ CSVs into merged file
│   ├── 2_area_calculate_from_combined.py   # Calculate area from merged CSVs
│   ├── 3_analysis_create_plots.py          # Generate summary plots from processed data
│   ├── 4_compare_multiple_runs_plots.py    # Compare multiple processed runs visually
│   ├── benchmark_daq_performance.py        # DAQ performance benchmarking
│   ├── logging_setup.py                   # Logging configuration utilities
│   ├── setup_mcc_path.py                  # MCC DLL path setup
│   ├── test_read_board1.py                # Board 1 analog input test script
│   └── __init__.py                        # Utils package init
├── data/                          # Output directory for CSV files (not tracked in git)
│   └── ...                        # Timestamped folders and output files
```

## Configuration

Edit `config.py` to customize:
- Data output folder location
- Logging level and file output
- Device-specific settings (no need to change if declared explicitly in measurement scripts)

## Troubleshooting

### "Board not found" errors
- Ensure InstaCal shows the device as connected and functional
- Verify the `board_num` matches the number assigned in InstaCal
- Check USB connections and drivers
- Voltage readings can also be checked in InstaCal to make sure signal are read

### Data loss at high sample rates

- Reduce sample rate or number of channels
- Run the benchmark tool to assess your system's maximum capability

### Utilities

The `utils` folder contains helpful scripts and modules for data analysis, diagnostics, and setup:

- **1_analysis_combine_csv.py**: Combines temperature and data DAQ CSVs from a run into a single merged file with aligned metadata.
- **2_area_calculate_from_combined.py**: Processes merged CSVs to calculate area under signal curves and filter outliers for each batch.
- **3_analysis_create_plots.py**: Generates summary plots (temperature and area) from processed CSV files for quick visualization.
- **4_compare_multiple_runs_plots.py**: Lets you select and compare multiple processed runs, overlaying plots for visual analysis. Currently made to select 3 processed data files into a large plot of 4 data plots with 3 rows.
- **benchmark_daq_performance.py**: Benchmarks DAQ sample rates and buffer performance, reporting data loss and throughput at different speeds. Used as high level assessment of CPU performance to SCAN (DAQ) and READ (clear DAQ buffer into PC memory) continuously.
- **test_read_board1.py**: Simple script to continuously read and print analog input values from Board 1 for hardware diagnostics.

### Temperature device issues
- USB-TEMP only supports software-based scanning (not hardware background scan)
- Keep sample rates low (typically 1-10 Hz) for temperature measurements
- Verify thermocouple type matches configuration in InstaCal


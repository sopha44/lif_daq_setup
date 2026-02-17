"""
Acquisition Controller for LIF DAQ system.
Orchestrates data acquisition workflows across multiple DAQ devices.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np

from utils.setup_mcc_path import setup_mcc_path
setup_mcc_path()  # Add MCC DLL path before importing mcculw

from device_manager import DeviceManager
from signal_processor import SignalProcessor
from utils.logging_setup import get_logger
from mcculw.enums import ULRange
import config

logger = get_logger(__name__)


class AcquisitionController:
    def _save_continuous_sample(self, board_num, daq_config, sample_num, scan, timestamp):
        """
        Save a single scan/sample to the board's CSV file. Creates the file and header if needed.
        Args:
            board_num: Board number
            daq_config: DAQ config dict for this board
            sample_num: Sample or cycle number (for Trigger_or_N column)
            scan: List of channel values
            timestamp: datetime object for this sample
        """
        # Get or create CSV file handle
        if 'csv_file' not in daq_config:
            num_channels = daq_config['high_chan'] - daq_config['low_chan'] + 1
            model_name = daq_config['name'].replace('-', '_')
            filename = self.acquisition_folder / f"board{board_num}_{model_name}.csv"
            daq_config['csv_file'] = open(filename, 'w', newline='', buffering=1)  # Line buffered
            # Write header
            headers = ['Trigger_or_N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
            daq_config['csv_file'].write(','.join(headers) + '\n')
        # Write single row directly
        row = [str(sample_num), timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]] + [str(v) for v in scan]
        daq_config['csv_file'].write(','.join(row) + '\n')
    """
    Orchestrates multi-device data acquisition workflows.
    Coordinates timing, data collection, and file saving.
    """
    
    def __init__(self):
        """Initialize the acquisition controller."""
        self.manager = DeviceManager()
        self.devices = {}
        self.daq_configs = {}
        self.acquisition_start_time = None
        self.acquisition_folder = None
        self.signal_processors = {}  # Signal processors for each DAQ
        logger.info("Acquisition Controller initialized")
    
    def setup_daq(self, board_num: int, name: str, sample_rate: int, 
                  low_chan: int, high_chan: int,
                  continuous_mode: bool = False,
                  sample_every_nth: int = 1,
                  trigger_channel: int = None, 
                  trigger_voltage_high: float = None, 
                  trigger_voltage_low: float = None,
                  acquisition_window_us: float = None,
                  enable_processing: bool = False):
        """
        Setup a single DAQ device.
        
        Args:
            board_num: Board number (0, 1, 2)
            name: Friendly name for logging
            sample_rate: Sample rate in Hz
            low_chan: First channel number
            high_chan: Last channel number
            continuous_mode: If True, save all data continuously (no triggers)
            sample_every_nth: In continuous mode, only save every Nth sample (default: 1 = save all)
            trigger_channel: Which channel to monitor for trigger (only if continuous_mode=False)
            trigger_voltage_high: Rising edge threshold voltage
            trigger_voltage_low: Falling edge threshold voltage
            acquisition_window_us: Time window per channel in microseconds (total acquisition time = this × num_channels)
            enable_processing: Signal processing (default: False)
        """
        # 10 second circular buffer for continuous operation
        duration_seconds = 10.0
        
        config = self.manager.setup_daq(
            board_num=board_num,
            name=name,
            sample_rate=sample_rate,
            low_chan=low_chan,
            high_chan=high_chan,
            duration_seconds=duration_seconds
        )
        
        # Configuration based on mode
        config['continuous_mode'] = continuous_mode
        config['sample_every_nth'] = sample_every_nth
        config['trigger_channel'] = trigger_channel
        config['trigger_voltage_high'] = trigger_voltage_high
        config['trigger_voltage_low'] = trigger_voltage_low
        config['acquisition_window_us'] = acquisition_window_us
        config['state'] = 'WAITING'  # WAITING or ACQUIRING
        config['last_voltage'] = 0.0

        # Add config to daq_configs for use in acquisition
        self.daq_configs[board_num] = config
    
    def start_acquisition(self, trigger_check_decimation: int = 20, time_between_points: float = 30.0, total_duration_minutes: float = None):
        """
        Start time-gated trigger-based acquisition on all configured DAQs.
        Waits time_between_points seconds, then monitors Board 1 CH0 for trigger.
        When triggered, both boards take measurements, then cycle repeats.
        
        Args:
            trigger_check_decimation: Check every Nth sample for trigger while WAITING (default: 20)
                                     Lower = more responsive but slower. Higher = faster but may miss short triggers.
            time_between_points: Time in seconds to wait between measurement cycles (default: 30.0)
            total_duration_minutes: Total duration in minutes to run acquisition (default: None = run indefinitely)
        """
        self.trigger_check_decimation = trigger_check_decimation
        self.time_between_points = time_between_points

        # FAST MODE: If time_between_points == 0.0, skip all logging and arming logic
        if time_between_points == 0.0:
            self.acquisition_start_time = datetime.now()
            timestamp_str = self.acquisition_start_time.strftime("%Y-%m-%d_%H-%M-%S")
            data_folder = Path(config.DATA_FOLDER)
            self.acquisition_folder = data_folder / timestamp_str
            self.acquisition_folder.mkdir(parents=True, exist_ok=True)
            try:
                # Start scans for all boards
                for board_num, daq_config in self.daq_configs.items():
                    is_temp_device = 'TEMP' in daq_config['name'].upper()
                    if is_temp_device:
                        daq_config['device'].start_scan(
                            low_chan=daq_config['low_chan'],
                            high_chan=daq_config['high_chan'],
                            rate=daq_config['sample_rate'],
                            points_per_channel=daq_config['points_per_channel']
                        )
                    else:
                        daq_config['device'].start_scan(
                            low_chan=daq_config['low_chan'],
                            high_chan=daq_config['high_chan'],
                            rate=daq_config['sample_rate'],
                            points_per_channel=daq_config['points_per_channel'],
                            ai_range=ULRange.BIP10VOLTS
                        )
                # Find master board
                master_board = None
                for board_num, daq_config in self.daq_configs.items():
                    if daq_config.get('trigger_channel') is not None:
                        master_board = board_num
                        break
                if master_board is None:
                    return
                master_config = self.daq_configs[master_board]
                start_time = time.time()
                measurement_cycle = 1
                end_time = None
                if total_duration_minutes is not None:
                    end_time = start_time + (total_duration_minutes * 60)
                while True:
                    current_time = time.time() - start_time
                    current_datetime = self.acquisition_start_time + timedelta(seconds=current_time)
                    if end_time is not None and time.time() >= end_time:
                        break
                    # Read new data from all boards
                    all_board_data = {}
                    for board_num, daq_config in self.daq_configs.items():
                        scans = daq_config['device'].read_new_data()
                        all_board_data[board_num] = scans
                    new_scans = all_board_data[master_board]
                    if len(new_scans) > 0:
                        trigger_ch_idx = master_config['trigger_channel'] - master_config['low_chan']
                        for idx, scan in enumerate(new_scans):
                            voltage = scan[trigger_ch_idx]
                            last_v = master_config.get('last_voltage', -10.0)
                            if last_v < master_config['trigger_voltage_high'] and voltage >= master_config['trigger_voltage_high']:
                                acquisition_window_s = master_config.get('acquisition_window_us', 1000000.0) / 1e6
                                samples_to_collect = int(master_config['sample_rate'] * acquisition_window_s)
                                scans_to_collect = samples_to_collect
                                acquisition_buffer = new_scans[idx:idx + scans_to_collect]
                                acquisition_start = time.time()
                                while len(acquisition_buffer) < scans_to_collect:
                                    elapsed = time.time() - acquisition_start
                                    if elapsed > 1.0:
                                        break
                                    new_data = master_config['device'].read_new_data()
                                    if len(new_data) > 0:
                                        acquisition_buffer.extend(new_data[:scans_to_collect - len(acquisition_buffer)])
                                    else:
                                        time.sleep(0.001)
                                for acq_scan in acquisition_buffer:
                                    self._save_continuous_sample(master_board, master_config, measurement_cycle, acq_scan, current_datetime)
                                for b_num, b_config in self.daq_configs.items():
                                    if b_num == master_board:
                                        continue
                                    board_scans = all_board_data[b_num]
                                    if len(board_scans) == 0 and 'last_scan_during_gate' in b_config:
                                        board_scans = [b_config['last_scan_during_gate']]
                                    if len(board_scans) > 0:
                                        measurement = board_scans[-1]
                                        self._save_continuous_sample(b_num, b_config, measurement_cycle, measurement, current_datetime)
                                    else:
                                        try:
                                            device = b_config['device']
                                            if hasattr(device, 'read_single_value'):
                                                direct_values = []
                                                for ch in range(b_config['low_chan'], b_config['high_chan'] + 1):
                                                    val = device.read_single_value(ch)
                                                    direct_values.append(val)
                                                self._save_continuous_sample(b_num, b_config, measurement_cycle, direct_values, current_datetime)
                                        except Exception:
                                            pass
                                # Stop and restart scan for all boards (buffer reset)
                                for board_num, daq_config in self.daq_configs.items():
                                    try:
                                        daq_config['device'].stop_scan()
                                        is_temp_device = 'TEMP' in daq_config['name'].upper()
                                        if is_temp_device:
                                            daq_config['device'].start_scan(
                                                low_chan=daq_config['low_chan'],
                                                high_chan=daq_config['high_chan'],
                                                rate=daq_config['sample_rate'],
                                                points_per_channel=daq_config['points_per_channel']
                                            )
                                        else:
                                            daq_config['device'].start_scan(
                                                low_chan=daq_config['low_chan'],
                                                high_chan=daq_config['high_chan'],
                                                rate=daq_config['sample_rate'],
                                                points_per_channel=daq_config['points_per_channel'],
                                                ai_range=ULRange.BIP10VOLTS
                                            )
                                    except Exception:
                                        pass
                                master_config['last_voltage'] = voltage
                                measurement_cycle += 1
                                break
                            master_config['last_voltage'] = voltage
                    else:
                        time.sleep(0.001)
            except KeyboardInterrupt:
                pass
            except Exception:
                raise
            finally:
                for board_num, daq_config in self.daq_configs.items():
                    if 'csv_file' in daq_config:
                        daq_config['csv_file'].close()
                self.manager.disconnect_all()
                self.daq_configs.clear()
            return

        # ...existing code for normal mode...
    
    def _create_single_daq_dataframe(self, timestamps, data, low_chan, high_chan):
        """
        Create a pandas dataframe for a single DAQ.
        
        Args:
            timestamps: List of timestamp values
            data: List of lists containing channel data
            low_chan: First channel number
            high_chan: Last channel number
        
        Returns:
            pandas DataFrame
        """
        # Create column names
        columns = ['Timestamp']
        for ch in range(low_chan, high_chan + 1):
            columns.append(f'CH{ch}')
        
        # Convert data to numpy array
        data_array = np.array(data)
        
        # Combine timestamp and data
        combined = np.column_stack([np.array(timestamps), data_array])
        
        # Create dataframe
        df = pd.DataFrame(combined, columns=columns)
        
        return df
    
    def _save_daq_data(self, df, board_num, product_name, start_time):
        """
        Save DAQ data to CSV file with formatted filename.
        
        Args:
            df: DataFrame to save
            board_num: Board number
            product_name: Product name of the DAQ
            start_time: datetime of acquisition start
        
        Returns:
            Path to saved file
        """
        # Format: Board0_USB-1608GX-2AO.csv (timestamp is in folder name)
        filename = f"Board{board_num}_{product_name}.csv"
        
        filepath = self.acquisition_folder / filename
        
        df.to_csv(filepath, index=False)
        
        return filepath


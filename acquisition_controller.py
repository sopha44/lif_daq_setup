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
 
from mcculw.enums import ULRange
import config


class AcquisitionController:
    def _get_integrated_csv_path(self, board_num, daq_config):
        """
        Returns the path for the integrated CSV file for a board, including datetime in the filename.
        """
        # Use acquisition start time for consistent naming
        if not hasattr(self, 'acquisition_start_time') or self.acquisition_start_time is None:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            timestamp_str = self.acquisition_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = daq_config['name'].replace('-', '_')
        filename = f"integrated_board{board_num}_{model_name}_{timestamp_str}.csv"
        return self.acquisition_folder / filename
        

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
            headers = ['N', 'Trigger_or_N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
            daq_config['csv_file'].write(','.join(headers) + '\n')
            daq_config['row_counter'] = 1
        # Write single row directly
        row = [str(daq_config['row_counter']), str(sample_num), timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]] + [str(v) for v in scan]
        daq_config['csv_file'].write(','.join(row) + '\n')
        daq_config['row_counter'] += 1
    
    def __init__(self):
        """Initialize the acquisition controller."""
        self.manager = DeviceManager()
        self.devices = {}
        self.daq_configs = {}
        self.acquisition_start_time = None
        self.acquisition_folder = None
    
    def setup_daq(self, board_num: int, name: str, sample_rate: int, 
                  low_chan: int, high_chan: int,
                  continuous_mode: bool = False,
                  sample_every_nth: int = 1,
                  trigger_channel: int = None, 
                  trigger_voltage_high: float = None, 
                  trigger_voltage_low: float = None,
                  acquisition_window_us: float = None,):
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
    
    def start_acquisition(
        self,
        trigger_check_decimation: int = 20,  # How often to check trigger (1 = every scan)
        time_between_points: float = 30.0,   # Time between measurement cycles (seconds)
        total_duration_minutes: float = None, # Total acquisition duration (minutes)
        restart_scan_each_trigger: bool = True, # Restart scan after each trigger (buffer reset)
        check_buffer_every: int = 0,           # How often to check buffer fullness (0 = never)
        max_rows_in_memory: int = 100000,           # Max rows to keep in memory for processing (per board)
        
    ):
        self.trigger_check_decimation = trigger_check_decimation
        print(f"[INFO] trigger_check_decimation set to {self.trigger_check_decimation}")
        self.time_between_points = time_between_points
        self.acquisition_start_time = datetime.now()
        timestamp_str = self.acquisition_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        data_folder = Path(config.DATA_FOLDER)
        self.acquisition_folder = data_folder / timestamp_str
        self.acquisition_folder.mkdir(parents=True, exist_ok=True)

        # Create CSV files for each board at the start, with headers
        for board_num, daq_config in self.daq_configs.items():
            print(f"[CONFIG] Board {board_num}: {daq_config}")
            integrate = daq_config.get('integrate_scan')
            model_name = daq_config['name'].replace('-', '_')
            num_channels = daq_config['high_chan'] - daq_config['low_chan'] + 1
            if integrate and daq_config['device'].__class__.__name__ == 'MCCDevice':
                # Integrated file only
                csv_path = self._get_integrated_csv_path(board_num, daq_config)
                if not csv_path.exists():
                    with open(csv_path, 'w', newline='') as f:
                        headers = ['N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
                        f.write(','.join(headers) + '\n')
                daq_config['row_counter'] = 1
            elif not (integrate and board_num == 1):
                # Only create standard file if not integrated board 1
                filename = self.acquisition_folder / f"board{board_num}_{model_name}.csv"
                if not filename.exists():
                    with open(filename, 'w', newline='') as f:
                        headers = ['Trigger_or_N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
                        f.write(','.join(headers) + '\n')

        try:
            # Step 1: Start scans on all configured devices
            for board_num, daq_config in self.daq_configs.items():
                # Check device type by name (temperature devices vs analog input devices)
                is_temp_device = 'TEMP' in daq_config['name'].upper()
                if is_temp_device:
                    # Temperature device - no ai_range parameter
                    try:
                        result = daq_config['device'].start_scan(
                            low_chan=daq_config['low_chan'],
                            high_chan=daq_config['high_chan'],
                            rate=daq_config['sample_rate'],
                            points_per_channel=daq_config['points_per_channel']
                        )
                        print(f"[SCAN] Started scan on board {board_num} (TEMP), result: {result}")
                    except Exception as e:
                        print(f"[SCAN][ERROR] Failed to start scan on board {board_num} (TEMP): {e}")
                    print(f"[SCAN][STATUS] Board {board_num} is_scanning: {getattr(daq_config['device'], 'is_scanning', 'N/A')}")
                    daq_config['temperature_buffer'] = []
                    daq_config['row_counter'] = 1
                else:
                    # Analog input device - use BIP10VOLTS range
                    try:
                        result = daq_config['device'].start_scan(
                            low_chan=daq_config['low_chan'],
                            high_chan=daq_config['high_chan'],
                            rate=daq_config['sample_rate'],
                            points_per_channel=daq_config['points_per_channel'],
                            ai_range=ULRange.BIP10VOLTS
                        )
                        print(f"[SCAN] Started scan on board {board_num} (AI), result: {result}")
                    except Exception as e:
                        print(f"[SCAN][ERROR] Failed to start scan on board {board_num} (AI): {e}")
                    print(f"[SCAN][STATUS] Board {board_num} scanning: {getattr(daq_config['device'], 'scanning', 'N/A')}")
            # Allow slow devices time to collect initial data
            time.sleep(1.5)

            # Step 2: Find the master trigger board
            master_board = None
            for board_num, daq_config in self.daq_configs.items():
                if daq_config.get('trigger_channel') is not None:
                    master_board = board_num
                    break
            if master_board is None:
                return
            # Step 3: Collect data - time-gated trigger monitoring
            start_time = time.time()
            last_status_time = start_time
            last_measurement_time = None  # Will be set after first trigger
            last_trigger_time = None  # For measuring time between triggers
            measurement_cycle = 1  # Start with cycle 1
            buffer_check_counter = 0
            trigger_armed = True  # Arm immediately for first trigger
            master_config = self.daq_configs[master_board]
            voltage_min = float('inf')
            voltage_max = float('-inf')
            # Calculate end time if total_duration_minutes is specified
            end_time = None
            if total_duration_minutes is not None:
                end_time = start_time + (total_duration_minutes * 60)
            # Run until Ctrl+C or total_duration_minutes elapsed
            print("[INFO] Starting acquisition loop...")
            while True:
                print(f"[LOOP] Acquisition loop running. Measurement cycle: {measurement_cycle}")
                current_time = time.time() - start_time
                current_datetime = self.acquisition_start_time + timedelta(seconds=current_time)
                # Check if total duration has elapsed
                if end_time is not None and time.time() >= end_time:
                    break
                # Check if we should arm the trigger (after time_between_points has elapsed)
                # Skip this check for the first cycle (last_measurement_time is None)
                if last_measurement_time is not None:
                    time_since_last = time.time() - last_measurement_time
                    if not trigger_armed and time_since_last >= time_between_points:
                        trigger_armed = True
                        measurement_cycle += 1
                        voltage_min = float('inf')  # Reset voltage range for new cycle
                        voltage_max = float('-inf')
                # Read new data from all boards
                if not trigger_armed:
                    # During timing gate, keep most recent sample from slow boards, drain fast boards
                    for board_num, daq_config in self.daq_configs.items():
                        new_data = daq_config['device'].read_new_data()
                        if len(new_data) > 0:
                            # Keep the most recent scan for this board
                            daq_config['last_scan_during_gate'] = new_data[-1]
                    time.sleep(0.001)  # Small 1ms sleep during timing gate

                    # Status update during timing gate
                    if last_measurement_time is not None and time.time() - last_status_time >= 5.0:
                        time_since_last = time.time() - last_measurement_time
                        time_remaining = time_between_points - time_since_last
                        # print(f"Status: [TIMING GATE] Next cycle in {time_remaining:.1f}s (Cycle #{measurement_cycle})")
                        last_status_time = time.time()
                    continue

                # ARMED - read data from ALL boards first (don't drain yet)
                all_board_data = {}
                for board_num, daq_config in self.daq_configs.items():
                    integrate = daq_config.get('integrate_scan')
                    device = daq_config['device']
                    # Only MCCDevice supports integrate_scan; MCCTemperatureDevice does not
                    if hasattr(device, 'read_new_data'):
                        if integrate and device.__class__.__name__ == 'MCCDevice':
                            scans = device.read_new_data(integrate_scan=True)
                            print(f"[DATA] Board {board_num} ({daq_config['name']}): {len(scans)} scans read (integrate_scan=True)")
                            # Integration buffer logic for MCCDevice
                            if 'integration_buffer' not in daq_config:
                                daq_config['integration_buffer'] = []
                            if 'integration_counter' not in daq_config:
                                daq_config['integration_counter'] = 0
                            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            for scan in scans:
                                daq_config['integration_buffer'].append([daq_config['row_counter'], timestamp_now] + list(scan))
                                daq_config['row_counter'] += 1
                            daq_config['integration_counter'] += len(scans)
                            rows_before_write = daq_config.get('rows_before_write')
                            if daq_config['integration_counter'] >= rows_before_write:
                                csv_path = self._get_integrated_csv_path(board_num, daq_config)
                                with open(csv_path, 'a', newline='') as f:
                                    if f.tell() == 0:
                                        num_channels = daq_config['high_chan'] - daq_config['low_chan'] + 1
                                        headers = ['N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
                                        f.write(','.join(headers) + '\n')
                                    for row in daq_config['integration_buffer']:
                                        f.write(','.join(str(v) for v in row) + '\n')
                                daq_config['integration_buffer'] = []
                                daq_config['integration_counter'] = 0
                            all_board_data[board_num] = scans
                        else:
                            # For MCCTemperatureDevice or non-integrated MCCDevice
                            scans = device.read_new_data()
                            print(f"[DATA] Board {board_num} ({daq_config['name']}): {len(scans)} scans read")
                    else:
                        print(f"[DATA] Board {board_num} ({daq_config['name']}): device has no read_new_data() method")
                        # Buffer temperature scans in memory
                        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        for scan in scans:
                            daq_config['temperature_buffer'].append([
                                daq_config['row_counter'], measurement_cycle, timestamp_now, *scan
                            ])
                            daq_config['row_counter'] += 1
                        all_board_data[board_num] = scans

                # Check master board for trigger
                new_scans = all_board_data[master_board]

                # --- Optimized trigger check logic ---
                if self.trigger_check_decimation == 1:
                    # Fastest path: check every scan, no decimation logic
                    if len(new_scans) > 0:
                        trigger_ch_idx = master_config['trigger_channel'] - master_config['low_chan']
                        for idx, scan in enumerate(new_scans):
                            voltage = scan[trigger_ch_idx]
                            last_v = master_config.get('last_voltage', -10.0)
                            master_config['last_scan'] = scan
                            print(f"[DEBUG] Scan {idx}: CH{master_config['trigger_channel']}={voltage:.4f}V (last_v={last_v:.4f}, threshold={master_config['trigger_voltage_high']:.4f})")
                            if last_v < master_config['trigger_voltage_high'] and voltage >= master_config['trigger_voltage_high']:
                                now = time.time()
                                if last_trigger_time is not None:
                                    delta = now - last_trigger_time
                                    # print(f"[CYCLE {measurement_cycle}] Time since last trigger: {delta:.6f} s")
                                last_trigger_time = now
                                ch_values = ", ".join([f"CH{master_config['low_chan']+i}={v:.3f}V" for i, v in enumerate(scan)])
                                print(f"[CYCLE {measurement_cycle}] Trigger detected! Board {master_board}: {ch_values}")
                                acquisition_window_s = master_config.get('acquisition_window_us', 1000000.0) / 1e6
                                samples_to_collect = int(master_config['sample_rate'] * acquisition_window_s)
                                scans_to_collect = samples_to_collect
                                # print(f"  Collecting {scans_to_collect} scans ({samples_to_collect} samples) over {acquisition_window_s*1000:.1f}ms...")
                                acquisition_buffer = new_scans[idx:idx + scans_to_collect]
                                acquisition_start = time.time()
                                while len(acquisition_buffer) < scans_to_collect:
                                    elapsed = time.time() - acquisition_start
                                    if elapsed > 1.0:
                                        print(f"  Acquisition timeout after {elapsed:.1f}s! Got {len(acquisition_buffer)}/{scans_to_collect} scans")
                                        break
                                    new_data = master_config['device'].read_new_data()
                                    if len(new_data) > 0:
                                        acquisition_buffer.extend(new_data[:scans_to_collect - len(acquisition_buffer)])
                                    else:
                                        time.sleep(0.00001)
                                # print(f"  Board {master_board}: Collected {len(acquisition_buffer)} samples in {(time.time()-acquisition_start)*1000:.1f}ms")
                                for acq_scan in acquisition_buffer:
                                    self._save_continuous_sample(master_board, master_config, measurement_cycle, acq_scan, current_datetime)
                                for b_num, b_config in self.daq_configs.items():
                                    if b_num == master_board:
                                        continue
                                    if b_num not in all_board_data:
                                        print(f"[WARN] No data for board {b_num} in all_board_data, skipping.")
                                        continue
                                    board_scans = all_board_data[b_num]
                                    if len(board_scans) == 0 and 'last_scan_during_gate' in b_config:
                                        board_scans = [b_config['last_scan_during_gate']]
                                        # print(f"  Board {b_num}: Using saved data from timing gate")
                                    if len(board_scans) > 0:
                                        measurement = board_scans[-1]
                                        self._save_continuous_sample(b_num, b_config, measurement_cycle, measurement, current_datetime)
                                        # print(f"  Board {b_num}: {measurement}")
                                    else:
                                        try:
                                            device = b_config['device']
                                            if hasattr(device, 'read_single_value'):
                                                direct_values = []
                                                for ch in range(b_config['low_chan'], b_config['high_chan'] + 1):
                                                    val = device.read_single_value(ch)
                                                    direct_values.append(val)
                                                self._save_continuous_sample(b_num, b_config, measurement_cycle, direct_values, current_datetime)
                                                print(f"  Board {b_num}: {direct_values} (read directly)")
                                            else:
                                                pass
                                        except Exception as e:
                                            print(f"  Board {b_num}: No data available! ({e})")
                                if restart_scan_each_trigger:
                                    for board_num, daq_config in self.daq_configs.items():
                                        try:
                                            daq_config['device'].stop_scan()
                                            # print(f"  Board {board_num}: Scan stopped for buffer reset.")
                                            is_temp_device = 'TEMP' in daq_config['name'].upper()
                                            if is_temp_device:
                                                daq_config['device'].start_scan(
                                                    low_chan=daq_config['low_chan'],
                                                    high_chan=daq_config['high_chan'],
                                                    rate=daq_config['sample_rate'],
                                                    points_per_channel=daq_config['points_per_channel']
                                                )
                                                # print(f"  Board {board_num}: Scan restarted (Temperature device)")
                                            else:
                                                daq_config['device'].start_scan(
                                                    low_chan=daq_config['low_chan'],
                                                    high_chan=daq_config['high_chan'],
                                                    rate=daq_config['sample_rate'],
                                                    points_per_channel=daq_config['points_per_channel'],
                                                    ai_range=ULRange.BIP10VOLTS
                                                )
                                                # print(f"  Board {board_num}: Scan restarted (Analog input device)")
                                        except Exception as e:
                                            print(f"  Board {board_num}: Error stopping/restarting scan: {e}")
                                trigger_armed = False
                                last_measurement_time = time.time()
                                master_config['last_voltage'] = voltage
                                print(f"[CYCLE {measurement_cycle}] Complete - waiting {time_between_points}s for next cycle")
                                # print(f"[DEBUG] Trigger disarmed at {current_time:.1f}s (Cycle {measurement_cycle})")
                                break
                            master_config['last_voltage'] = voltage
                            # Buffer check logic removed for speed
                    else:
                        # No scans available while armed - small sleep to avoid busy waiting
                        # print(f"[DEBUG] No new scans available while ARMED at {current_time:.1f}s.")
                        time.sleep(0.00001)
                    # Track voltage range
                    if trigger_armed and len(new_scans) > 0:
                        for scan in new_scans:
                            v = scan[trigger_ch_idx]
                            voltage_min = min(voltage_min, v)
                            voltage_max = max(voltage_max, v)
                    # Status update while armed removed for speed
                    # No sleep needed - read_new_data() is non-blocking
                else:
                    # Decimation logic for trigger_check_decimation > 1
                    trigger_ch_idx = master_config['trigger_channel'] - master_config['low_chan']
                    decimation_counter = 0
                    if len(new_scans) > 0:
                        for idx, scan in enumerate(new_scans):
                            decimation_counter += 1
                            if decimation_counter % self.trigger_check_decimation != 0:
                                continue
                            voltage = scan[trigger_ch_idx]
                            last_v = master_config.get('last_voltage', -10.0)
                            master_config['last_scan'] = scan
                            if last_v < master_config['trigger_voltage_high'] and voltage >= master_config['trigger_voltage_high']:
                                now = time.time()
                                if last_trigger_time is not None:
                                    delta = now - last_trigger_time
                                    # logger.info(f"[CYCLE {measurement_cycle}] Time since last trigger: {delta:.6f} s")
                                last_trigger_time = now
                                ch_values = ", ".join([f"CH{master_config['low_chan']+i}={v:.3f}V" for i, v in enumerate(scan)])
                                # logger.info(f"[CYCLE {measurement_cycle}] Trigger detected! Board {master_board}: {ch_values}")
                                acquisition_window_s = master_config.get('acquisition_window_us', 1000000.0) / 1e6
                                samples_to_collect = int(master_config['sample_rate'] * acquisition_window_s)
                                scans_to_collect = samples_to_collect
                                # logger.info(f"  Collecting {scans_to_collect} scans ({samples_to_collect} samples) over {acquisition_window_s*1000:.1f}ms...")
                                acquisition_buffer = new_scans[idx:idx + scans_to_collect]
                                acquisition_start = time.time()
                                while len(acquisition_buffer) < scans_to_collect:
                                    elapsed = time.time() - acquisition_start
                                    if elapsed > 1.0:
                                        # logger.warning(f"  Acquisition timeout after {elapsed:.1f}s! Got {len(acquisition_buffer)}/{scans_to_collect} scans")
                                        break
                                    new_data = master_config['device'].read_new_data()
                                    if len(new_data) > 0:
                                        acquisition_buffer.extend(new_data[:scans_to_collect - len(acquisition_buffer)])
                                    else:
                                        time.sleep(0.00001)
                                # print(f"  Board {master_board}: Collected {len(acquisition_buffer)} samples in {(time.time()-acquisition_start)*1000:.1f}ms")
                                for acq_scan in acquisition_buffer:
                                    self._save_continuous_sample(master_board, master_config, measurement_cycle, acq_scan, current_datetime)
                                for b_num, b_config in self.daq_configs.items():
                                    if b_num == master_board:
                                        continue
                                    board_scans = all_board_data[b_num]
                                    if len(board_scans) == 0 and 'last_scan_during_gate' in b_config:
                                        board_scans = [b_config['last_scan_during_gate']]
                                        # print(f"  Board {b_num}: Using saved data from timing gate")
                                    if len(board_scans) > 0:
                                        measurement = board_scans[-1]
                                        self._save_continuous_sample(b_num, b_config, measurement_cycle, measurement, current_datetime)
                                        # print(f"  Board {b_num}: {measurement}")
                                    else:
                                        try:
                                            device = b_config['device']
                                            if hasattr(device, 'read_single_value'):
                                                direct_values = []
                                                for ch in range(b_config['low_chan'], b_config['high_chan'] + 1):
                                                    val = device.read_single_value(ch)
                                                    direct_values.append(val)
                                                self._save_continuous_sample(b_num, b_config, measurement_cycle, direct_values, current_datetime)
                                                # print(f"  Board {b_num}: {direct_values} (read directly)")
                                            else:
                                                # print(f"  Board {b_num}: No data available!")
                                                pass    
                                        except Exception as e:
                                            print(f"  Board {b_num}: No data available! ({e})")
                                if restart_scan_each_trigger:
                                    for board_num, daq_config in self.daq_configs.items():
                                        try:
                                            daq_config['device'].stop_scan()
                                            # print(f"  Board {board_num}: Scan stopped for buffer reset.")
                                            is_temp_device = 'TEMP' in daq_config['name'].upper()
                                            if is_temp_device:
                                                daq_config['device'].start_scan(
                                                    low_chan=daq_config['low_chan'],
                                                    high_chan=daq_config['high_chan'],
                                                    rate=daq_config['sample_rate'],
                                                    points_per_channel=daq_config['points_per_channel']
                                                )
                                                # print(f"  Board {board_num}: Scan restarted (Temperature device)")
                                            else:
                                                daq_config['device'].start_scan(
                                                    low_chan=daq_config['low_chan'],
                                                    high_chan=daq_config['high_chan'],
                                                    rate=daq_config['sample_rate'],
                                                    points_per_channel=daq_config['points_per_channel'],
                                                    ai_range=ULRange.BIP10VOLTS
                                                )
                                                # print(f"  Board {board_num}: Scan restarted (Analog input device)")
                                        except Exception as e:
                                            print(f"  Board {board_num}: Error stopping/restarting scan: {e}")
                                trigger_armed = False
                                last_measurement_time = time.time()
                                master_config['last_voltage'] = voltage
                                print(f"[CYCLE {measurement_cycle}] Complete - waiting {time_between_points}s for next cycle")
                                # print(f"[DEBUG] Trigger disarmed at {current_time:.1f}s (Cycle {measurement_cycle})")
                                break
                            master_config['last_voltage'] = voltage
                            # Buffer check logic
                    else:
                        # No scans available while armed - small sleep to avoid busy waiting
                        # print(f"[DEBUG] No new scans available while ARMED at {current_time:.1f}s.")
                        time.sleep(0.00001)
                    # Track voltage range
                    if trigger_armed and len(new_scans) > 0:
                        for scan in new_scans:
                            v = scan[trigger_ch_idx]
                            voltage_min = min(voltage_min, v)
                            voltage_max = max(voltage_max, v)
            
        except KeyboardInterrupt:
            print("Acquisition interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"Error during acquisition: {e}")
            raise
        
        finally:
            # Final flush for any remaining integrated data in buffer
            for board_num, daq_config in self.daq_configs.items():
                integrate = daq_config.get('integrate_scan')
                device = daq_config['device']
                if integrate and device.__class__.__name__ == 'MCCDevice':
                    if 'integration_buffer' in daq_config and daq_config['integration_buffer']:
                        csv_path = self._get_integrated_csv_path(board_num, daq_config)
                        with open(csv_path, 'a', newline='') as f:
                            # Write header if file is empty
                            if f.tell() == 0:
                                num_channels = daq_config['high_chan'] - daq_config['low_chan'] + 1
                                headers = ['N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
                                f.write(','.join(headers) + '\n')
                            for row in daq_config['integration_buffer']:
                                f.write(','.join(str(v) for v in row) + '\n')
                        daq_config['integration_buffer'] = []
                        daq_config['integration_counter'] = 0
                # Close CSV files for non-integrated boards
                if 'temperature_buffer' in daq_config and daq_config['temperature_buffer']:
                    # Write all buffered temperature data at once
                    num_channels = daq_config['high_chan'] - daq_config['low_chan'] + 1
                    model_name = daq_config['name'].replace('-', '_')
                    filename = self.acquisition_folder / f"board{board_num}_{model_name}.csv"
                    with open(filename, 'w', newline='') as f:
                        headers = ['N', 'Trigger_or_N', 'Timestamp'] + [f'CH{daq_config["low_chan"] + i}' for i in range(num_channels)]
                        f.write(','.join(headers) + '\n')
                        for row in daq_config['temperature_buffer']:
                            f.write(','.join(str(v) for v in row) + '\n')
                if 'csv_file' in daq_config:
                    daq_config['csv_file'].close()
                    # print(f"Closed CSV file for Board {board_num}")
            # Cleanup - disconnect devices
            # print("Cleaning up and disconnecting devices...")
            self.manager.disconnect_all()
            self.daq_configs.clear()
            # print("Acquisition complete and devices disconnected")
    
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
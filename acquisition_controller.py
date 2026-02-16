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
        logger.info("=" * 70)
        logger.info("Starting Time-Gated Trigger-Based Data Acquisition")
        logger.info(f"Time between measurements: {time_between_points}s")
        if total_duration_minutes is not None:
            logger.info(f"Total duration: {total_duration_minutes} minutes")
        logger.info(f"Active DAQs: {len(self.daq_configs)}")
        
        # Log trigger configuration
        for board_num, daq_config in self.daq_configs.items():
            if daq_config['trigger_channel'] is not None:
                logger.info(f"  Board {board_num}: CH{daq_config['trigger_channel']} " +
                           f"rising@{daq_config['trigger_voltage_high']}V -> " +
                           f"falling@{daq_config['trigger_voltage_low']}V")
            else:
                logger.info(f"  Board {board_num}: Slave mode (triggered by master board)")
        
        logger.info("=" * 70)
        
        self.acquisition_start_time = datetime.now()
        
        # Create timestamped folder for this acquisition run
        timestamp_str = self.acquisition_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        data_folder = Path(config.DATA_FOLDER)
        self.acquisition_folder = data_folder / timestamp_str
        self.acquisition_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving data to: {self.acquisition_folder}")
        
        try:
            # Step 1: Start scans on all configured devices
            logger.info("Starting background scans...")
            for board_num, daq_config in self.daq_configs.items():
                # Check device type by name (temperature devices vs analog input devices)
                is_temp_device = 'TEMP' in daq_config['name'].upper()
                
                if is_temp_device:
                    # Temperature device - no ai_range parameter
                    daq_config['device'].start_scan(
                        low_chan=daq_config['low_chan'],
                        high_chan=daq_config['high_chan'],
                        rate=daq_config['sample_rate'],
                        points_per_channel=daq_config['points_per_channel']
                    )
                    logger.info(f"  Board {board_num} scan started (Temperature device)")
                else:
                    # Analog input device - use BIP10VOLTS range
                    daq_config['device'].start_scan(
                        low_chan=daq_config['low_chan'],
                        high_chan=daq_config['high_chan'],
                        rate=daq_config['sample_rate'],
                        points_per_channel=daq_config['points_per_channel'],
                        ai_range=ULRange.BIP10VOLTS
                    )
                    logger.info(f"  Board {board_num} scan started (Range: BIP10VOLTS)")
            logger.info("All scans started")
            
            # Allow slow devices time to collect initial data
            logger.info("Waiting 1.5 seconds for all devices to collect initial data...")
            time.sleep(1.5)
            
            # Step 2: Find the master trigger board
            master_board = None
            for board_num, daq_config in self.daq_configs.items():
                if daq_config.get('trigger_channel') is not None:
                    master_board = board_num
                    logger.info(f"Master trigger board: Board {master_board}")
                    break
            
            if master_board is None:
                logger.error("No master trigger board configured! Need at least one board with trigger_channel set.")
                return
            
            # Step 3: Collect data - time-gated trigger monitoring
            logger.info(f"Monitoring for triggers (with {time_between_points}s timing gate)...")
            
            start_time = time.time()
            last_status_time = start_time
            last_measurement_time = None  # Will be set after first trigger
            measurement_cycle = 1  # Start with cycle 1
            trigger_armed = True  # Arm immediately for first trigger
            master_config = self.daq_configs[master_board]
            voltage_min = float('inf')
            voltage_max = float('-inf')
            
            logger.info(f"[CYCLE {measurement_cycle}] Trigger armed immediately - waiting for first trigger...")
            
            # Calculate end time if total_duration_minutes is specified
            end_time = None
            if total_duration_minutes is not None:
                end_time = start_time + (total_duration_minutes * 60)
                logger.info(f"Acquisition will end at {(self.acquisition_start_time + timedelta(minutes=total_duration_minutes)).strftime('%H:%M:%S')}")
            
            # Run until Ctrl+C or total_duration_minutes elapsed
            while True:
                current_time = time.time() - start_time
                current_datetime = self.acquisition_start_time + timedelta(seconds=current_time)
                
                # Check if total duration has elapsed
                if end_time is not None and time.time() >= end_time:
                    logger.info(f"Total duration of {total_duration_minutes} minutes elapsed - ending acquisition")
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
                        logger.info(f"[CYCLE {measurement_cycle}] Trigger armed after {time_since_last:.1f}s - waiting for trigger...")
                
                # Read new data from all boards
                if not trigger_armed:
                    # During timing gate, keep most recent sample from slow boards, drain fast boards
                    for board_num, daq_config in self.daq_configs.items():
                        new_data = daq_config['device'].read_new_data()
                        if len(new_data) > 0:
                            # Keep the most recent scan for this board
                            daq_config['last_scan_during_gate'] = new_data[-1]
                    time.sleep(0.01)  # Small sleep during timing gate
                    
                    # Status update during timing gate
                    if last_measurement_time is not None and time.time() - last_status_time >= 5.0:
                        time_since_last = time.time() - last_measurement_time
                        time_remaining = time_between_points - time_since_last
                        logger.info(f"Status: [TIMING GATE] Next cycle in {time_remaining:.1f}s (Cycle #{measurement_cycle})")
                        last_status_time = time.time()
                    continue
                
                # ARMED - read data from ALL boards first (don't drain yet)
                all_board_data = {}
                for board_num, daq_config in self.daq_configs.items():
                    scans = daq_config['device'].read_new_data()
                    logger.debug(f"[DEBUG] Board {board_num}: {len(scans)} scans read, device={daq_config['device']}.")
                    all_board_data[board_num] = scans
                
                # Check master board for trigger
                new_scans = all_board_data[master_board]
                
                if len(new_scans) > 0:
                    trigger_ch_idx = master_config['trigger_channel'] - master_config['low_chan']
                    
                    for idx, scan in enumerate(new_scans):
                        voltage = scan[trigger_ch_idx]
                        last_v = master_config.get('last_voltage', -10.0)  # Default to low value
                        
                        # Store last scan for status display
                        master_config['last_scan'] = scan
                        
                        # Check for rising edge trigger on EVERY scan (no decimation)
                        # We need to check every scan to catch fast pulses
                        if last_v < master_config['trigger_voltage_high'] and \
                           voltage >= master_config['trigger_voltage_high']:
                            # TRIGGER DETECTED - Start acquisition window
                            # Format all channel values for Board 1
                            ch_values = ", ".join([f"CH{master_config['low_chan']+i}={v:.3f}V" 
                                                   for i, v in enumerate(scan)])
                            logger.info(f"[CYCLE {measurement_cycle}] Trigger detected! Board {master_board}: {ch_values}")
                            
                            # Calculate how many samples to collect
                            # acquisition_window_us is the simultaneous window for all channels
                            acquisition_window_s = master_config.get('acquisition_window_us', 1000000.0) / 1e6
                            samples_to_collect = int(master_config['sample_rate'] * acquisition_window_s)
                            scans_to_collect = samples_to_collect
                            logger.info(f"  Collecting {scans_to_collect} scans ({samples_to_collect} samples) over {acquisition_window_s*1000:.1f}ms...")
                            
                            # Use only the needed scans from this batch starting at trigger point
                            acquisition_buffer = new_scans[idx:idx + scans_to_collect]
                            acquisition_start = time.time()
                            
                            while len(acquisition_buffer) < scans_to_collect:
                                elapsed = time.time() - acquisition_start
                                if elapsed > 1.0:  # 1 second timeout (should only take milliseconds)
                                    logger.warning(f"  Acquisition timeout after {elapsed:.1f}s! Got {len(acquisition_buffer)}/{scans_to_collect} scans")
                                    break
                                
                                # Read more data
                                new_data = master_config['device'].read_new_data()
                                if len(new_data) > 0:
                                    acquisition_buffer.extend(new_data[:scans_to_collect - len(acquisition_buffer)])
                                else:
                                    time.sleep(0.001)  # Small sleep if no data
                            
                            logger.info(f"  Board {master_board}: Collected {len(acquisition_buffer)} samples in {(time.time()-acquisition_start)*1000:.1f}ms")
                            
                            # Save all collected samples for Board 1
                            for acq_scan in acquisition_buffer:
                                self._save_continuous_sample(master_board, master_config, 
                                                            measurement_cycle, 
                                                            acq_scan, 
                                                            current_datetime)

                            # Save single measurement from other boards
                            for b_num, b_config in self.daq_configs.items():
                                if b_num == master_board:
                                    continue  # Already saved
                                board_scans = all_board_data[b_num]
                                if len(board_scans) == 0 and 'last_scan_during_gate' in b_config:
                                    board_scans = [b_config['last_scan_during_gate']]
                                    logger.info(f"  Board {b_num}: Using saved data from timing gate")
                                if len(board_scans) > 0:
                                    measurement = board_scans[-1]
                                    self._save_continuous_sample(b_num, b_config, 
                                                                measurement_cycle, 
                                                                measurement, 
                                                                current_datetime)
                                    logger.info(f"  Board {b_num}: {measurement}")
                                else:
                                    try:
                                        device = b_config['device']
                                        if hasattr(device, 'read_single_value'):
                                            direct_values = []
                                            for ch in range(b_config['low_chan'], b_config['high_chan'] + 1):
                                                val = device.read_single_value(ch)
                                                direct_values.append(val)
                                            self._save_continuous_sample(b_num, b_config, 
                                                                        measurement_cycle, 
                                                                        direct_values, 
                                                                        current_datetime)
                                            logger.info(f"  Board {b_num}: {direct_values} (read directly)")
                                        else:
                                            logger.warning(f"  Board {b_num}: No data available!")
                                    except Exception as e:
                                        logger.warning(f"  Board {b_num}: No data available! ({e})")

                            # Stop and restart scan for all boards (buffer reset)
                            for board_num, daq_config in self.daq_configs.items():
                                try:
                                    daq_config['device'].stop_scan()
                                    logger.info(f"  Board {board_num}: Scan stopped for buffer reset.")
                                    is_temp_device = 'TEMP' in daq_config['name'].upper()
                                    if is_temp_device:
                                        daq_config['device'].start_scan(
                                            low_chan=daq_config['low_chan'],
                                            high_chan=daq_config['high_chan'],
                                            rate=daq_config['sample_rate'],
                                            points_per_channel=daq_config['points_per_channel']
                                        )
                                        logger.info(f"  Board {board_num}: Scan restarted (Temperature device)")
                                    else:
                                        daq_config['device'].start_scan(
                                            low_chan=daq_config['low_chan'],
                                            high_chan=daq_config['high_chan'],
                                            rate=daq_config['sample_rate'],
                                            points_per_channel=daq_config['points_per_channel'],
                                            ai_range=ULRange.BIP10VOLTS
                                        )
                                        logger.info(f"  Board {board_num}: Scan restarted (Analog input device)")
                                except Exception as e:
                                    logger.error(f"  Board {board_num}: Error stopping/restarting scan: {e}")

                            # Reset for next cycle
                            trigger_armed = False
                            last_measurement_time = time.time()
                            master_config['last_voltage'] = voltage
                            logger.info(f"[CYCLE {measurement_cycle}] Complete - waiting {time_between_points}s for next cycle")
                            logger.debug(f"[DEBUG] Trigger disarmed at {current_time:.1f}s (Cycle {measurement_cycle})")
                            break  # Exit scan loop

                        master_config['last_voltage'] = voltage
                else:
                    # No scans available while armed - small sleep to avoid busy waiting
                    logger.debug(f"[DEBUG] No new scans available while ARMED at {current_time:.1f}s.")
                    time.sleep(0.001)

                # Track voltage range
                if trigger_armed and len(new_scans) > 0:
                    trigger_ch_idx = master_config['trigger_channel'] - master_config['low_chan']
                    for scan in new_scans:
                        v = scan[trigger_ch_idx]
                        voltage_min = min(voltage_min, v)
                        voltage_max = max(voltage_max, v)

                # Status update while armed
                if trigger_armed and time.time() - last_status_time >= 5.0:
                    # Show all channel values from Board 1
                    last_scan = master_config.get('last_scan', None)
                    if last_scan is not None:
                        ch_values = ", ".join([f"CH{master_config['low_chan']+i}={v:.3f}V" 
                                               for i, v in enumerate(last_scan)])
                        if voltage_min != float('inf') and voltage_max != float('-inf'):
                            logger.info(f"Status [{current_time:.1f}s]: [ARMED] Board {master_board}: {ch_values} | Range: [{voltage_min:.3f}V to {voltage_max:.3f}V]")
                        else:
                            logger.info(f"Status [{current_time:.1f}s]: [ARMED] Board {master_board}: {ch_values} | Range: [no new data]")
                        logger.debug(f"[DEBUG] Status update while ARMED. Voltage min: {voltage_min}, max: {voltage_max}")
                        # Reset range for next period
                        voltage_min = float('inf')
                        voltage_max = float('-inf')
                    else:
                        logger.info(f"Status [{current_time:.1f}s]: [ARMED] Waiting for data from Board {master_board}")
                    last_status_time = time.time()

                # No sleep needed - read_new_data() is non-blocking
            
        except KeyboardInterrupt:
            logger.info("Acquisition interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error during acquisition: {e}", exc_info=True)
            raise
        
        finally:
            # Close CSV files
            for board_num, daq_config in self.daq_configs.items():
                if 'csv_file' in daq_config:
                    daq_config['csv_file'].close()
                    logger.info(f"Closed CSV file for Board {board_num}")
            
            # Cleanup - disconnect devices
            logger.info("Cleaning up and disconnecting devices...")
            self.manager.disconnect_all()
            self.daq_configs.clear()
            logger.info("Acquisition complete and devices disconnected")
    
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


def main():
    """Main entry point for running acquisition."""
    from utils.logging_setup import setup_logging
    
    # Setup logging
    setup_logging(log_level=config.LOG_LEVEL, log_to_file=config.LOG_TO_FILE)
    
    # Create controller
    controller = AcquisitionController()
    
    try:
        # Setup each DAQ with its specific configuration
        duration = 10.0
        
        # Board 0: USB-1608GX-2AO
        # controller.setup_daq(
        #     board_num=0,
        #     name="USB-1608GX-2AO",
        #     sample_rate=1000,
        #     low_chan=0,
        #     high_chan=3,
        #     duration_seconds=duration
        # )
        
        # Board 0: USB-TEMP-AI
        # Slave device - no trigger, will be measured when Board 1 triggers
        controller.setup_daq(
            board_num=0,
            name="USB-TEMP-AI",
            sample_rate=1,  # 1 Hz
            low_chan=1,
            high_chan=3,
            enable_processing=False 
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
            enable_processing=False
        )
        
        # Start acquisition with time-gated triggering
        # trigger_check_decimation: check every Nth sample while ARMED (at 400kHz: 4000 = 100 checks/sec)
        # time_between_points: wait 3 seconds between measurement cycles
        controller.start_acquisition(
            trigger_check_decimation=5,  # Check every 5th sample = 200,000 checks/sec
            time_between_points=30.0,
            total_duration_minutes=120  # Run for 120 minutes, 2 hours (None = run indefinitely)
        )
        
    except KeyboardInterrupt:
        logger.info("Acquisition interrupted by user")
    except Exception as e:
        logger.error(f"Acquisition failed: {e}")
        raise


if __name__ == '__main__':
    main()

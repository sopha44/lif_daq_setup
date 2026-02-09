"""
Acquisition Controller for LIF DAQ system.
Orchestrates data acquisition workflows across multiple DAQ devices.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from utils.setup_mcc_path import setup_mcc_path
setup_mcc_path()  # Add MCC DLL path before importing mcculw

from mcculw.enums import ULRange
from device_manager import DeviceManager
from signal_processor import SignalProcessor
from utils.logging_setup import get_logger
import config

logger = get_logger(__name__)


class AcquisitionController:
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
                  low_chan: int, high_chan: int, duration_seconds: float,
                  ul_range: 'ULRange' = None,
                  enable_processing: bool = True):
        """
        Setup a single DAQ device with its configuration.
        Delegates to DeviceManager for device configuration.
        
        Args:
            board_num: Board number (0, 1, 2)
            name: Friendly name for logging (e.g., "USB-1608GX-2AO", "USB-TEMP")
            sample_rate: Sample rate in Hz
            low_chan: First channel number
            high_chan: Last channel number
            duration_seconds: Acquisition duration for buffer sizing
            ul_range: ULRange enum value for input range/resolution
                ULRange.BIP10VOLTS - ±10V
                ULRange.BIP5VOLTS - ±5V
                ULRange.BIP2PT5VOLTS - ±2.5V
                ULRange.BIP1VOLTS - ±1V
                ULRange.UNI10VOLTS - 0-10V
            enable_processing: Whether to create a signal processor for this DAQ (default: True)
        """
        # Delegate to DeviceManager for configuration
        daq_config = self.manager.setup_daq(
            board_num=board_num,
            name=name,
            sample_rate=sample_rate,
            low_chan=low_chan,
            high_chan=high_chan,
            duration_seconds=duration_seconds,
            ul_range=ul_range
        )
        
        # Store configuration for acquisition control
        self.daq_configs[board_num] = daq_config
        
        # Create signal processor for this DAQ if enabled
        if enable_processing:
            self.signal_processors[board_num] = SignalProcessor( 
                sample_rate=sample_rate,
                num_channels=high_chan - low_chan + 1
            )
            logger.info(f"  Signal processor created for Board {board_num}")
        else:
            logger.info(f"  Signal processing disabled for Board {board_num}")
    
    def get_processor(self, board_num: int) -> SignalProcessor:
        """
        Get the signal processor for a specific board.
        Use this to configure processing blocks after setup_daq().
        
        Args:
            board_num: Board number
            
        Returns:
            SignalProcessor instance for this board
            
        Raises:
            KeyError: If board has no processor (processing disabled or not setup)
        """
        if board_num not in self.signal_processors:
            raise KeyError(f"No signal processor for board {board_num}. Enable processing in setup_daq().")
        return self.signal_processors[board_num]
    
    def start_acquisition(self, duration_seconds: float):
        """
        Start acquisition on all configured DAQs simultaneously.
        
        Args:
            duration_seconds: Acquisition duration in seconds
        """
        logger.info("=" * 70)
        logger.info("Starting Data Acquisition")
        logger.info(f"Duration: {duration_seconds}s")
        logger.info(f"Active DAQs: {len(self.daq_configs)}")
        logger.info("=" * 70)
        
        self.acquisition_start_time = datetime.now()
        
        # Create timestamped folder for this acquisition run
        timestamp_str = self.acquisition_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        data_folder = Path(daq_config.DATA_FOLDER)
        self.acquisition_folder = data_folder / timestamp_str
        self.acquisition_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving data to: {self.acquisition_folder}")
        
        try:
            # Step 1: Start scans on all configured devices
            logger.info("Starting background scans...")
            for board_num, daq_config in self.daq_configs.items():
                daq_config['device'].start_scan(
                    low_chan=daq_config['low_chan'],
                    high_chan=daq_config['high_chan'],
                    rate=daq_config['sample_rate'],
                    points_per_channel=daq_config['points_per_channel'],
                    ai_range=daq_config.get('ul_range')
                )
                logger.info(f"  Board {board_num} scan started")
            logger.info("All scans started")
            
            # Step 2: Collect data during acquisition
            logger.info(f"Acquiring data for {duration_seconds} seconds...")
            
            start_time = time.time()
            last_log_time = start_time
            
            while time.time() - start_time < duration_seconds:
                current_time = time.time() - start_time
                current_datetime = self.acquisition_start_time + timedelta(seconds=current_time)
                
                # Read new data from each DAQ
                for board_num, daq_config in self.daq_configs.items():
                    new_scans = daq_config['device'].read_new_data()
                    
                    if len(new_scans) > 0:
                        # Real-time signal processing if processor exists for this board
                        if board_num in self.signal_processors:
                            processor = self.signal_processors[board_num]
                            # One timestamp for the batch - users can calculate exact time from index
                            batch_timestamps = [current_datetime] * len(new_scans)
                            
                            process_result = processor.process_batch(new_scans, batch_timestamps)
                            
                            # Only store data if processor says to keep it
                            if process_result['keep_data']:
                                # Store processed write_data, not raw scans
                                write_data = process_result['write_data']
                                # write_data is a dict like {'ch0': array, 'ch1': array, 'Counter00': value}
                                # For now, just store raw scans - TODO: restructure for processed data
                                for scan in new_scans:
                                    daq_config['data'].append(scan)
                                    daq_config['timestamps'].append(current_datetime)
                        else:
                            # No processor - store all raw data (e.g., for temperature device)
                            for scan in new_scans:
                                daq_config['data'].append(scan)
                                daq_config['timestamps'].append(current_datetime)
                
                # Log progress every 2 seconds
                if time.time() - last_log_time >= 2.0:
                    elapsed = time.time() - start_time
                    sample_counts = ', '.join([f"Board{bn}: {len(daq_cfg['data'])}" 
                                              for bn, daq_cfg in self.daq_configs.items()])
                    logger.info(f"Progress: {elapsed:.1f}s / {duration_seconds}s ({sample_counts} samples)")
                    last_log_time = time.time()
                
                # Small sleep to prevent busy-waiting
                time.sleep(0.001)
            
            logger.info(f"Data collection complete.")
            for board_num, daq_config in self.daq_configs.items():
                logger.info(f"  Board {board_num}: {len(daq_config['data'])} samples")
            
            # Step 3: Stop scans
            logger.info("Stopping scans...")
            for board_num, daq_config in self.daq_configs.items():
                daq_config['device'].stop_scan()
                logger.info(f"  Board {board_num} stopped")
            logger.info("All scans stopped")
            
            # Step 4: Save data to CSV files
            logger.info("Saving data to CSV files...")
            for board_num, daq_config in self.daq_configs.items():
                if len(daq_config['data']) > 0:
                    df = self._create_single_daq_dataframe(
                        daq_config['timestamps'], 
                        daq_config['data'], 
                        daq_config['low_chan'], 
                        daq_config['high_chan']
                    )
                    file_path = self._save_daq_data(
                        df, 
                        board_num, 
                        daq_config['device'].info.product_name, 
                        self.acquisition_start_time
                    )
                    logger.info(f"  Board {board_num} saved: {file_path.name} ({len(df)} rows)")
            
            logger.info("All data saved successfully!")
            
        except Exception as e:
            logger.error(f"Error during acquisition: {e}", exc_info=True)
            raise
        
        finally:
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
        # Create column names - Index first, then Timestamp, then channels
        columns = ['Index', 'Timestamp']
        for ch in range(low_chan, high_chan + 1):
            columns.append(f'CH{ch}')
        
        # Convert data to numpy array
        data_array = np.array(data)
        
        # Create index column (0, 1, 2, ...)
        indices = np.arange(len(timestamps))
        
        # Combine index, timestamp, and data
        combined = np.column_stack([indices, np.array(timestamps), data_array])
        
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
    from signal_processing_config import configure_all_processors
    
    # Setup logging
    setup_logging(log_level=config.LOG_LEVEL, log_to_file=config.LOG_TO_FILE)
    
    # Create controller
    controller = AcquisitionController()
    
    try:
        # Setup each DAQ with its specific configuration
        duration = 10.0
        
        #TODO: Refactor to place DAQ config in config file

        # Board 0: USB-1608GX-2AO
        # controller.setup_daq(
        #     board_num=0,
        #     name="USB-1608GX-2AO",
        #     sample_rate=1000,
        #     low_chan=0,
        #     high_chan=3,
        #     duration_seconds=duration
        # )
        
        # Board 1: USB-TEMP (only channel 0 has thermocouple)
        # 2Hz
        controller.setup_daq(
            board_num=1,
            name="USB-TEMP",
            sample_rate=2, # 2Hz, Sampling rate dependent on DAQ model
            low_chan=0, # Lowest channel num
            high_chan=0,  # Highest channel num
            duration_seconds=duration, # Acquisition time
            ul_range=None, # No voltage range for temperature device, this variable is ignored in USB-TEMP scan
            enable_processing=True # Enable processing
        )
        
        # Board 2: USB-1604HS-2AO (channels 0-3 available)
        # 1,330,000Hz max on DAQ
        controller.setup_daq(
            board_num=2,
            name="USB-1604HS-2AO",
            sample_rate=100000, #100kHz
            low_chan=0,
            high_chan=3,  # All 4 channels
            duration_seconds=duration,
            ul_range=None, # Use default range for 1604HS-2AO (±10V)
            enable_processing=True # Enable processing
        )
        
        # Configure signal processing for all boards
        configure_all_processors(controller)
        
        # Start acquisition on all configured DAQs
        controller.start_acquisition(duration_seconds=duration)
        
    except KeyboardInterrupt:
        logger.info("Acquisition interrupted by user")
    except Exception as e:
        logger.error(f"Acquisition failed: {e}")
        raise


if __name__ == '__main__':
    main()

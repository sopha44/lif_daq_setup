"""
Acquisition Controller for LIF DAQ system.
Orchestrates data acquisition workflows across multiple DAQ devices.
"""
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from utils.setup_mcc_path import setup_mcc_path
setup_mcc_path()  # Add MCC DLL path before importing mcculw

from device_manager import DeviceManager
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
        logger.info("Acquisition Controller initialized")
    
    def run_dual_daq_acquisition(self, 
                                 duration_seconds: float = 10.0,
                                 sample_rate_daq1: int = 1000,
                                 sample_rate_daq2: int = 1000,
                                 sample_rate_daq3: int = 1000,
                                 daq1_low_chan: int = 0,
                                 daq1_high_chan: int = 3,
                                 daq2_low_chan: int = 0,
                                 daq2_high_chan: int = 0,
                                 daq3_low_chan: int = 0,
                                 daq3_high_chan: int = 2):
        """
        Run acquisition on three DAQ devices simultaneously with independent sampling rates and channels.
        Board 0: USB-1608GX-2AO (8 channels available: 0-7)
        Board 1: USB-TEMP (temperature channels, requires thermocouples)
        Board 2: USB-1604HS-2AO (4 channels available: 0-3)
        
        Args:
            duration_seconds: Acquisition duration in seconds (default: 10.0)
            sample_rate_daq1: Sample rate for DAQ1 in Hz (default: 1000)
            sample_rate_daq2: Sample rate for DAQ2 in Hz (default: 1000)
            sample_rate_daq3: Sample rate for DAQ3 in Hz (default: 1000)
            daq1_low_chan: First channel for DAQ1 (default: 0)
            daq1_high_chan: Last channel for DAQ1 (default: 3)
            daq2_low_chan: First channel for DAQ2/USB-TEMP (default: 0)
            daq2_high_chan: Last channel for DAQ2/USB-TEMP (default: 0 - single channel)
            daq3_low_chan: First channel for DAQ3 (default: 0)
            daq3_high_chan: Last channel for DAQ3 (default: 2, max is 3)
        """
        logger.info("=" * 70)
        logger.info("Starting Triple DAQ Acquisition")
        logger.info(f"Board 0: USB-1608GX-2AO @ {sample_rate_daq1} Hz, Channels {daq1_low_chan}-{daq1_high_chan}")
        logger.info(f"Board 1: USB-TEMP @ {sample_rate_daq2} Hz, Channels {daq2_low_chan}-{daq2_high_chan}")
        logger.info(f"Board 2: USB-1604HS-2AO @ {sample_rate_daq3} Hz, Channels {daq3_low_chan}-{daq3_high_chan}")
        logger.info(f"Duration: {duration_seconds}s")
        logger.info("=" * 70)
        
        daq1 = None
        daq2 = None
        daq3 = None
        acquisition_start_time = datetime.now()
        
        try:
            # Step 1: Connect to all three devices
            logger.info("Connecting to DAQ devices...")
            daq1 = self.manager.connect_mcc_device(board_num=0)  # USB-1608GX-2AO
            daq2 = self.manager.connect_mcc_device(board_num=1)  # USB-TEMP
            daq3 = self.manager.connect_mcc_device(board_num=2)  # USB-1604HS-2AO
            logger.info(f"Board 0 connected: {daq1.info.product_name}")
            logger.info(f"Board 1 connected: {daq2.info.product_name}")
            logger.info(f"Board 2 connected: {daq3.info.product_name}")
            
            # Calculate points per channel for the duration
            points_per_channel_daq1 = int(sample_rate_daq1 * duration_seconds)
            points_per_channel_daq2 = int(sample_rate_daq2 * duration_seconds)
            points_per_channel_daq3 = int(sample_rate_daq3 * duration_seconds)
            
            num_channels_daq1 = daq1_high_chan - daq1_low_chan + 1
            num_channels_daq2 = daq2_high_chan - daq2_low_chan + 1
            num_channels_daq3 = daq3_high_chan - daq3_low_chan + 1
            
            logger.info(f"Expected samples per channel:")
            logger.info(f"  DAQ1: {points_per_channel_daq1} ({num_channels_daq1} channels)")
            logger.info(f"  DAQ2: {points_per_channel_daq2} ({num_channels_daq2} channels)")
            logger.info(f"  DAQ3: {points_per_channel_daq3} ({num_channels_daq3} channels)")
            
            # Step 2: Start scans on all devices with individual channel configs
            logger.info("Starting background scans...")
            daq1.start_scan(
                low_chan=daq1_low_chan,
                high_chan=daq1_high_chan,
                rate=sample_rate_daq1,
                points_per_channel=points_per_channel_daq1
            )
            
            daq2.start_scan(
                low_chan=daq2_low_chan,
                high_chan=daq2_high_chan,
                rate=sample_rate_daq2,
                points_per_channel=points_per_channel_daq2
            )
            
            daq3.start_scan(
                low_chan=daq3_low_chan,
                high_chan=daq3_high_chan,
                rate=sample_rate_daq3,
                points_per_channel=points_per_channel_daq3
            )
            logger.info("All scans started")
            
            # Step 3: Collect data during acquisition
            logger.info(f"Acquiring data for {duration_seconds} seconds...")
            
            data_daq1 = []
            data_daq2 = []
            data_daq3 = []
            timestamps_daq1 = []
            timestamps_daq2 = []
            timestamps_daq3 = []
            
            start_time = time.time()
            last_log_time = start_time
            
            while time.time() - start_time < duration_seconds:
                current_time = time.time() - start_time
                
                # Read new data from each DAQ using built-in tracking
                new_scans_daq1 = daq1.read_new_data()
                for scan in new_scans_daq1:
                    data_daq1.append(scan)
                    timestamps_daq1.append(current_time)
                
                new_scans_daq2 = daq2.read_new_data()
                for scan in new_scans_daq2:
                    data_daq2.append(scan)
                    timestamps_daq2.append(current_time)
                
                new_scans_daq3 = daq3.read_new_data()
                for scan in new_scans_daq3:
                    data_daq3.append(scan)
                    timestamps_daq3.append(current_time)
                
                # Log progress every 2 seconds
                if time.time() - last_log_time >= 2.0:
                    elapsed = time.time() - start_time
                    logger.info(f"Progress: {elapsed:.1f}s / {duration_seconds}s "
                              f"(DAQ1: {len(data_daq1)}, DAQ2: {len(data_daq2)}, DAQ3: {len(data_daq3)} samples)")
                    last_log_time = time.time()
                
                # Small sleep to prevent busy-waiting
                time.sleep(0.001)
            
            logger.info(f"Data collection complete.")
            logger.info(f"  DAQ1: {len(data_daq1)} samples")
            logger.info(f"  DAQ2: {len(data_daq2)} samples")
            logger.info(f"  DAQ3: {len(data_daq3)} samples")
            
            # Step 4: Stop scans
            logger.info("Stopping scans...")
            daq1.stop_scan()
            daq2.stop_scan()
            daq3.stop_scan()
            logger.info("Scans stopped")
            
            # Step 5: Create separate dataframes and save to individual CSV files
            logger.info("Saving data to CSV files...")
            
            # Save DAQ1 data
            if len(data_daq1) > 0:
                df1 = self._create_single_daq_dataframe(timestamps_daq1, data_daq1, daq1_low_chan, daq1_high_chan)
                file1 = self._save_daq_data(df1, 0, daq1.info.product_name, acquisition_start_time)
                logger.info(f"  DAQ1 saved: {file1.name} ({len(df1)} rows)")
            
            # Save DAQ2 data
            if len(data_daq2) > 0:
                df2 = self._create_single_daq_dataframe(timestamps_daq2, data_daq2, daq2_low_chan, daq2_high_chan)
                file2 = self._save_daq_data(df2, 1, daq2.info.product_name, acquisition_start_time)
                logger.info(f"  DAQ2 saved: {file2.name} ({len(df2)} rows)")
            
            # Save DAQ3 data
            if len(data_daq3) > 0:
                df3 = self._create_single_daq_dataframe(timestamps_daq3, data_daq3, daq3_low_chan, daq3_high_chan)
                file3 = self._save_daq_data(df3, 2, daq3.info.product_name, acquisition_start_time)
                logger.info(f"  DAQ3 saved: {file3.name} ({len(df3)} rows)")
            
            logger.info("All data saved successfully!")
            
        except Exception as e:
            logger.error(f"Error during acquisition: {e}", exc_info=True)
            raise
        
        finally:
            # Step 6: Cleanup - disconnect devices
            logger.info("Cleaning up and disconnecting devices...")
            self.manager.disconnect_all()
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
        # Format: Board0_USB-1608GX-2AO_2026-02-04_14-30-25.csv
        timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Board{board_num}_{product_name}_{timestamp_str}.csv"
        
        data_folder = Path(config.DATA_FOLDER)
        data_folder.mkdir(exist_ok=True)
        filepath = data_folder / filename
        
        df.to_csv(filepath, index=False)
        
        return filepath


def main():
    """Main entry point for running acquisition."""
    from utils.logging_setup import setup_logging
    
    # Setup logging
    setup_logging(log_level=config.LOG_LEVEL, log_to_file=config.LOG_TO_FILE)
    
    # Create controller and run acquisition
    controller = AcquisitionController()
    
    try:
        controller.run_dual_daq_acquisition(
            duration_seconds=10.0,
            sample_rate_daq1=1000,
            sample_rate_daq2=10,  # Slower for temperature (doesn't change much)
            sample_rate_daq3=1000,
            daq1_low_chan=0,
            daq1_high_chan=3,
            daq2_low_chan=0,
            daq2_high_chan=0,  # USB-TEMP: Only channel 0 (has thermocouple)
            daq3_low_chan=0,
            daq3_high_chan=2   # USB-1604HS: Channels 0-2 (avoid non-existent channel 4)
        )
    except KeyboardInterrupt:
        logger.info("Acquisition interrupted by user")
    except Exception as e:
        logger.error(f"Acquisition failed: {e}")
        raise


if __name__ == '__main__':
    main()

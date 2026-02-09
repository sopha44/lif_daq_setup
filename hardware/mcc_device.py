"""
MCC DAQ Device implementation.
Provides operations for MCC USB-series DAQ devices.
"""
from ctypes import cast, POINTER, c_double, c_ushort, c_ulong
from typing import Optional, List, Tuple
import numpy as np

from mcculw import ul
from mcculw.enums import ScanOptions, FunctionType, Status, ULRange
from mcculw.device_info import DaqDeviceInfo

from utils.logging_setup import get_logger
import config as config

logger = get_logger(__name__)


class MCCDevice:
    """
    MCC DAQ device class for analog input operations.
    Handles scanning, data acquisition, and device configuration.
    """
    
    def __init__(self, board_num: int, daq_dev_info: DaqDeviceInfo):
        """
        Initialize MCC device.
        
        Args:
            board_num: Board number assigned to this device
            daq_dev_info: DaqDeviceInfo object with device capabilities
        """
        self.board_num = board_num
        self.info = daq_dev_info
        self.ai_info = daq_dev_info.get_ai_info()
        
        # Verify device supports analog input
        if not daq_dev_info.supports_analog_input:
            raise Exception(f"Device {daq_dev_info.product_name} does not support analog input")
        
        # Scanning state
        self.is_scanning = False
        self.memhandle = None
        self.ctypes_array = None
        self.scan_options = ScanOptions.BACKGROUND
        
        # Data collection tracking
        self.last_read_index = -1
        
        # Scan parameters (set during start_scan)
        self.low_chan = None
        self.high_chan = None
        self.num_chans = None
        self.rate = None
        self.ai_range = None
        self.points_per_channel = None
        self.total_count = None
        
        logger.info(f"MCC Device initialized: {self.info.product_name} "
                   f"(Board {self.board_num}, {self.ai_info.num_chans} channels)")
    
    def configure_scan(self, 
                      low_chan: Optional[int] = None,
                      high_chan: Optional[int] = None,
                      rate: Optional[int] = None,
                      points_per_channel: Optional[int] = None,
                      ai_range: Optional[ULRange] = None,
                      scan_options: Optional[ScanOptions] = None):
        """
        Configure scan parameters.
        
        Args:
            low_chan: First channel to scan (default: from config)
            high_chan: Last channel to scan (default: from config)
            rate: Sample rate in Hz (default: from config)
            points_per_channel: Samples per channel (default: from config)
            ai_range: Voltage range (default: from config)
            scan_options: Scan options (default: BACKGROUND | SCALEDATA if supported)
        """
        # Use config defaults if not specified
        self.low_chan = low_chan if low_chan is not None else config.DEFAULT_LOW_CHANNEL
        self.high_chan = high_chan if high_chan is not None else min(
            config.DEFAULT_HIGH_CHANNEL, 
            self.ai_info.num_chans - 1
        )
        self.num_chans = self.high_chan - self.low_chan + 1
        
        self.rate = rate if rate is not None else config.DEFAULT_SAMPLE_RATE
        self.points_per_channel = points_per_channel if points_per_channel is not None else config.DEFAULT_POINTS_PER_CHANNEL
        self.total_count = self.points_per_channel * self.num_chans
        
        # Use first supported range if not specified
        self.ai_range = ai_range if ai_range is not None else self.ai_info.supported_ranges[0]
        
        # Configure scan options
        if scan_options is not None:
            self.scan_options = scan_options
        else:
            self.scan_options = ScanOptions.BACKGROUND
            # Add SCALEDATA if supported
            if ScanOptions.SCALEDATA in self.ai_info.supported_scan_options:
                self.scan_options |= ScanOptions.SCALEDATA
        
        logger.info(f"Scan configured: CH{self.low_chan}-{self.high_chan}, "
                   f"{self.rate} Hz, {self.points_per_channel} pts/ch, "
                   f"Range: {self.ai_range.name}")
    
    def start_scan(self,
                   low_chan: Optional[int] = None,
                   high_chan: Optional[int] = None,
                   rate: Optional[int] = None,
                   points_per_channel: Optional[int] = None,
                   ai_range: Optional[ULRange] = None,
                   scan_options: Optional[ScanOptions] = None):
        """
        Start background analog input scan.
        
        Args:
            low_chan: First channel to scan
            high_chan: Last channel to scan
            rate: Sample rate in Hz
            points_per_channel: Samples per channel
            ai_range: Voltage range
            scan_options: Scan options
        
        Raises:
            Exception: If scan is already running or configuration fails
        """
        if self.is_scanning:
            raise Exception("Scan already in progress. Stop current scan first.")
        
        # Configure scan parameters
        self.configure_scan(low_chan, high_chan, rate, points_per_channel, ai_range, scan_options)
        
        try:
            # Allocate buffer based on scan options
            if ScanOptions.SCALEDATA in self.scan_options:
                # Scaled buffer returns floating-point values in engineering units
                self.memhandle = ul.scaled_win_buf_alloc(self.total_count)
                self.ctypes_array = cast(self.memhandle, POINTER(c_double))
            elif self.ai_info.resolution <= 16:
                # 16-bit or less resolution
                self.memhandle = ul.win_buf_alloc(self.total_count)
                self.ctypes_array = cast(self.memhandle, POINTER(c_ushort))
            else:
                # Greater than 16-bit resolution
                self.memhandle = ul.win_buf_alloc_32(self.total_count)
                self.ctypes_array = cast(self.memhandle, POINTER(c_ulong))
            
            if not self.memhandle:
                raise Exception("Failed to allocate memory buffer")
            
            # Start the background scan
            ul.a_in_scan(
                self.board_num,
                self.low_chan,
                self.high_chan,
                self.total_count,
                self.rate,
                self.ai_range,
                self.memhandle,
                self.scan_options
            )
            
            self.is_scanning = True
            
            # Reset last read index to -1 so first read starts from index 0
            self.last_read_index = -1
            
            logger.info(f"Background scan started on board {self.board_num}")
            
        except Exception as e:
            # Clean up buffer if scan failed
            if self.memhandle:
                ul.win_buf_free(self.memhandle)
                self.memhandle = None
                self.ctypes_array = None
            logger.error(f"Failed to start scan: {e}")
            raise
    
    def get_status(self) -> Tuple[Status, int, int]:
        """
        Get current scan status.
        
        Returns:
            Tuple of (status, current_count, current_index)
            - status: Scan status (IDLE, RUNNING, etc.)
            - current_count: Number of samples acquired
            - current_index: Index of last completed scan
        """
        if not self.is_scanning:
            return Status.IDLE, 0, 0
        
        status, curr_count, curr_index = ul.get_status(
            self.board_num, 
            FunctionType.AIFUNCTION
        )
        return status, curr_count, curr_index
    
    def read_latest_data(self) -> Optional[np.ndarray]:
        """
        Read the latest complete channel scan from the buffer.
        
        Returns:
            numpy array of shape (num_chans,) with latest values, or None if no data
        """
        if not self.is_scanning:
            logger.warning("No scan in progress")
            return None
        
        status, curr_count, curr_index = self.get_status()
        
        # Check if data is available
        if curr_count == 0:
            return None
        
        # Read latest complete scan
        data = []
        for i in range(curr_index, curr_index + self.num_chans):
            if ScanOptions.SCALEDATA in self.scan_options:
                # Data already in engineering units
                value = self.ctypes_array[i]
            else:
                # Convert to engineering units
                value = ul.to_eng_units(
                    self.board_num,
                    self.ai_range,
                    self.ctypes_array[i]
                )
            data.append(value)
        
        return np.array(data)
    
    def read_new_data(self) -> List[List[float]]:
        """
        Read new data since last read (incremental reading for long acquisitions).
        Returns data organized by scans (each scan contains values from all channels).
        Automatically tracks last read position to avoid duplicates.
        Handles circular buffer wraparound correctly.
        
        Returns:
            List of scans, where each scan is a list of channel values
        """
        if not self.is_scanning or not self.ctypes_array:
            return []
        
        status, curr_count, curr_index = self.get_status()
        
        if curr_count == 0:
            return []  # No data yet
        
        # Check if index hasn't changed
        if curr_index == self.last_read_index:
            return []  # No new data
        
        scans = []
        
        # Handle circular buffer wraparound
        # We need to read from last_read_index+1 to curr_index (inclusive)
        # But account for wraparound at total_count
        
        # Calculate number of new samples (accounting for circular buffer)
        if curr_index >= self.last_read_index:
            # No wraparound  
            new_samples = curr_index - self.last_read_index
        else:
            # Wraparound occurred
            new_samples = (self.total_count - self.last_read_index - 1) + curr_index + 1
        
        # Read new scans
        num_new_scans = new_samples // self.num_chans
        
        for scan_num in range(num_new_scans):
            scan_data = []
            for ch_offset in range(self.num_chans):
                # Calculate index with wraparound
                idx = (self.last_read_index + 1 + scan_num * self.num_chans + ch_offset) % self.total_count
                
                if self.scan_options & ScanOptions.SCALEDATA:
                    # Data already in engineering units
                    value = self.ctypes_array[idx]
                else:
                    # Convert to engineering units
                    value = ul.to_eng_units(
                        self.board_num,
                        self.ai_range,
                        self.ctypes_array[idx]
                    )
                scan_data.append(value)
            
            if len(scan_data) == self.num_chans:
                scans.append(scan_data)
        
        # Update last read index
        self.last_read_index = curr_index
        
        return scans
    
    def read_buffer_data(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Read multiple samples from the buffer.
        
        Args:
            num_samples: Number of complete scans to read (default: all available)
        
        Returns:
            numpy array of shape (num_samples, num_chans) or None if no data
        """
        if not self.is_scanning:
            logger.warning("No scan in progress")
            return None
        
        status, curr_count, curr_index = self.get_status()
        
        if curr_count == 0:
            return None
        
        # Determine how many complete scans are available
        available_scans = curr_count // self.num_chans
        if available_scans == 0:
            return None
        
        # Limit to requested number
        if num_samples is not None:
            available_scans = min(available_scans, num_samples)
        
        # Read data
        data = []
        start_index = max(0, curr_index - (available_scans * self.num_chans) + self.num_chans)
        
        for scan_idx in range(available_scans):
            scan_data = []
            for ch in range(self.num_chans):
                idx = start_index + scan_idx * self.num_chans + ch
                
                if ScanOptions.SCALEDATA in self.scan_options:
                    value = self.ctypes_array[idx]
                else:
                    value = ul.to_eng_units(
                        self.board_num,
                        self.ai_range,
                        self.ctypes_array[idx]
                    )
                scan_data.append(value)
            data.append(scan_data)
        
        return np.array(data)
    
    def stop_scan(self):
        """Stop the background scan and free buffer."""
        if not self.is_scanning:
            logger.warning("No scan in progress")
            return
        
        try:
            # Stop background operation
            ul.stop_background(self.board_num, FunctionType.AIFUNCTION)
            logger.info(f"Scan stopped on board {self.board_num}")
        except Exception as e:
            logger.error(f"Error stopping scan: {e}")
        finally:
            # Free buffer
            if self.memhandle:
                ul.win_buf_free(self.memhandle)
                self.memhandle = None
                self.ctypes_array = None
            
            self.is_scanning = False
    
    def cleanup(self):
        """Clean up device resources."""
        if self.is_scanning:
            self.stop_scan()
        logger.info(f"Device cleanup complete for board {self.board_num}")
    
    def get_device_info(self) -> dict:
        """
        Get device information.
        
        Returns:
            Dictionary with device details
        """
        return {
            'board_num': self.board_num,
            'product_name': self.info.product_name,
            'product_id': self.info.product_id,
            'unique_id': self.info.unique_id,
            'num_channels': self.ai_info.num_chans,
            'resolution': self.ai_info.resolution,
            'supported_ranges': [r.name for r in self.ai_info.supported_ranges],
            'is_scanning': self.is_scanning
        }
    
    def __repr__(self):
        return (f"MCCDevice(board={self.board_num}, "
                f"device={self.info.product_name}, "
                f"scanning={self.is_scanning})")

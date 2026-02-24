"""
MCC Temperature Device class for temperature input devices.
Handles devices like USB-TEMP that use temperature-specific functions.
"""
import ctypes
import threading
import time
from typing import List
from mcculw import ul
from mcculw.enums import ScanOptions, Status, TempScale
from mcculw.device_info import DaqDeviceInfo



class MCCTemperatureDevice:
    """
    Class to handle MCC DAQ devices with temperature input capabilities.
    Uses temperature-specific scanning functions (t_in_scan).
    """
    
    def __init__(self, board_num: int, daq_dev_info: DaqDeviceInfo):
        """
        Initialize MCC Temperature Device.
        
        Args:
            board_num: Board number assigned to this device
            daq_dev_info: DaqDeviceInfo object with device capabilities
        """
        self.board_num = board_num
        self.info = daq_dev_info
        
        # Verify device supports temperature input
        if not daq_dev_info.supports_temp_input:
            raise Exception(f"Device {daq_dev_info.product_name} does not support temperature input")
        
        # Scanning state
        self.is_scanning = False
        self.memhandle = None
        self.ctypes_array = None
        self.scan_options = ScanOptions.BACKGROUND
        # Background thread for scanning
        self.scan_thread = None
        self.stop_thread_event = threading.Event()
        
        # Data collection tracking
        self.current_index = -1
        self.current_count = 0
        self.last_read_index = -1
        self.data_lock = threading.Lock()
        
        # Scanning parameters (saved for reference)
        self.low_chan = None
        self.high_chan = None
        self.num_chans = None
        self.low_chan = None
        self.high_chan = None
        self.rate = None
        self.points_per_channel = None
        self.total_count = None
        
        # print(f"Temperature device initialized: {self.info.product_name} (Board {board_num})")
    
    def start_scan(self, 
                   low_chan: int = 0,
                   high_chan: int = 0,
                   rate: int = 100,
                   points_per_channel: int = 1000,
                   temp_scale: TempScale = TempScale.CELSIUS,
                   options: ScanOptions = ScanOptions.BACKGROUND):
        """
        Start a background temperature scan using a separate thread.
        Continuously polls temperature values at the specified rate.
        
        Args:
            low_chan: First temperature channel to scan
            high_chan: Last temperature channel to scan
            rate: Sample rate in Hz
            points_per_channel: Number of samples per channel to collect
            temp_scale: Temperature scale (CELSIUS, FAHRENHEIT, KELVIN)
            options: Scan options (stored for reference)
        """
        if self.is_scanning:
            print("Device is already scanning")
            raise Exception("Device is already scanning")
        
        # Save parameters
        self.low_chan = low_chan
        self.high_chan = high_chan
        self.num_chans = high_chan - low_chan + 1
        self.rate = rate
        self.points_per_channel = points_per_channel
        self.temp_scale = temp_scale
        self.scan_options = options
        
        # Calculate total points
        self.total_count = points_per_channel * self.num_chans
        
        # Allocate buffer for data (use scaled_win_buf_alloc for doubles)
        self.memhandle = ul.scaled_win_buf_alloc(self.total_count)
        if not self.memhandle:
            print("Failed to allocate memory buffer")
            raise Exception("Failed to allocate memory buffer")
        
        # Convert memhandle to ctypes array of doubles
        self.ctypes_array = ctypes.cast(self.memhandle, ctypes.POINTER(ctypes.c_double))
        
        # Reset data collection state
        with self.data_lock:
            self.current_index = -1
            self.current_count = 0
            self.last_read_index = -self.num_chans
        
        # print(f"Starting temperature scan on Board {self.board_num}")
        # print(f"  Channels: {low_chan}-{high_chan}")
        # print(f"  Rate: {rate} Hz")
        # print(f"  Points per channel: {points_per_channel}")
        # print(f"  Total points: {self.total_count}")
        # print(f"  Scale: {temp_scale.name}")
        
        # Start background scanning thread
        self.stop_thread_event.clear()
        self.scan_thread = threading.Thread(target=self._scan_thread_function, daemon=True)
        self.is_scanning = True
        self.scan_thread.start()
        
        # print("Temperature scan started in background thread")
    
    def _scan_thread_function(self):
        """Background thread function that continuously polls temperature values."""
        scan_interval = 1.0 / self.rate if self.rate > 0 else 0.1
        
        # print(f"Background scan thread started, interval: {scan_interval:.3f}s")
        
        # Keep reading continuously
        while not self.stop_thread_event.is_set():
            scan_start_time = time.time()
            
            # Read one scan (all channels)
            for chan_offset in range(self.num_chans):
                if self.stop_thread_event.is_set():
                    break
                
                chan = self.low_chan + chan_offset
                try:
                    temp_value = ul.t_in(self.board_num, chan, self.temp_scale)
                    # Store in buffer (wrap around when full)
                    with self.data_lock:
                        buffer_index = self.current_count % self.total_count
                        self.ctypes_array[buffer_index] = temp_value
                        self.current_count += 1
                        self.current_index = buffer_index
                except Exception as e:
                    print(f"Error reading temperature from channel {chan} on board {self.board_num}: {e}")
            
            # Sleep to maintain desired sample rate
            elapsed = time.time() - scan_start_time
            sleep_time = max(0, scan_interval - elapsed)
            if sleep_time > 0 and not self.stop_thread_event.is_set():
                self.stop_thread_event.wait(sleep_time)
        
        # print("Background scan thread finished")
    
    def stop_scan(self):
        """Stop the background scan thread and free resources."""
        if not self.is_scanning:
            print("No scan to stop")
            return
        
        # print(f"Stopping temperature scan on Board {self.board_num}")
        
        # Signal thread to stop
        self.stop_thread_event.set()
        
        # Wait for thread to finish (with timeout)
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=2.0)
            if self.scan_thread.is_alive():
                print("Scan thread did not stop within timeout")
        
        # Free the buffer
        if self.memhandle:
            ul.win_buf_free(self.memhandle)
            self.memhandle = None
        
        self.is_scanning = False
        self.ctypes_array = None
        # print("Temperature scan stopped")
    
    def get_status(self):
        """
        Get the current status of the background scan.
        
        Returns:
            tuple: (status, curr_count, curr_index)
                status: Status enum value
                curr_count: Number of samples collected so far
                curr_index: Index of the last sample in the buffer
        """
        if not self.is_scanning:
            return Status.IDLE, 0, -1
        
        with self.data_lock:
            # Return RUNNING while thread is active and buffer not full
            if self.current_count < self.total_count and self.scan_thread.is_alive():
                status = Status.RUNNING
            else:
                status = Status.IDLE
            
            return status, self.current_count, self.current_index
    
    def read_new_data(self) -> List[List[float]]:
        """
        Read new data since last read (incremental reading for long acquisitions).
        Returns data organized by scans (each scan contains values from all channels).
        
        Returns:
            List of scans, where each scan is a list of channel values
        """
        if not self.is_scanning or not self.ctypes_array:
            return []
        
        scans = []
        
        with self.data_lock:
            status, curr_count, curr_index = self.current_count, self.current_count, self.current_index
            
            if curr_count == 0 or curr_index == self.last_read_index:
                return []  # No new data
            
            # Read from last_read_index to curr_index in channel-sized chunks
            for scan_start in range(self.last_read_index + self.num_chans, curr_index + 1, self.num_chans):
                if scan_start >= 0:  # Skip negative indices
                    scan_data = []
                    for ch_offset in range(self.num_chans):
                        idx = scan_start + ch_offset
                        if idx <= curr_index and idx < self.total_count:
                            # Temperature values are already in engineering units
                            value = self.ctypes_array[idx]
                            scan_data.append(value)
                    
                    if len(scan_data) == self.num_chans:
                        scans.append(scan_data)
            
            # Update last read index
            self.last_read_index = curr_index
        
        return scans
    
    def read_single_value(self, channel: int) -> float:
        """
        Read a single temperature value from a channel (non-scanning mode).
        
        Args:
            channel: Channel number to read
            
        Returns:
            Temperature value in the configured scale
        """
        try:
            temp_value = ul.t_in(
                self.board_num,
                channel,
                self.temp_scale
            )
            return temp_value
        except Exception as e:
            print(f"Error reading temperature from channel {channel}: {e}")
            raise
    
    def read_buffer_data(self, start_index: int = 0, count: int = None) -> list:
        """
        Read temperature data from the buffer.
        
        Args:
            start_index: Starting index in the buffer
            count: Number of values to read (default: all available)
            
        Returns:
            List of temperature values
        """
        if not self.is_scanning or not self.ctypes_array:
            print("No active scan or buffer available")
            raise Exception("No active scan or buffer available")
        
        status, curr_count, curr_index = self.get_status()
        
        if count is None:
            count = curr_count
        
        # Ensure we don't read beyond available data
        if start_index + count > curr_count:
            count = max(0, curr_count - start_index)
        
        if count <= 0:
            return []
        
        # Read data from buffer
        data = []
        for i in range(start_index, start_index + count):
            data.append(self.ctypes_array[i])
        
        return data
    
    def __str__(self):
        """String representation of the device."""
        return f"MCCTemperatureDevice(Board {self.board_num}, {self.info.product_name})"
    
    def __repr__(self):
        """Detailed representation of the device."""
        return (f"MCCTemperatureDevice(board_num={self.board_num}, "
                f"product_name='{self.info.product_name}', "
                f"is_scanning={self.is_scanning})")

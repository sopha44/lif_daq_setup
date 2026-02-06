"""
Device Manager for DAQ devices.
Handles discovery, connection, and lifecycle management of DAQ hardware.
"""
from typing import Dict, List, Optional, Tuple
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.ul import ULError

from utils.logging_setup import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """
    Manages discovery, connection, and lifecycle of DAQ devices.
    Maintains a registry of connected devices and provides device objects.
    """
    
    def __init__(self):
        """Initialize the device manager."""
        self.connected_devices: Dict[int, object] = {}  # board_num -> device object
        self.device_info: Dict[int, DaqDeviceInfo] = {}  # board_num -> DaqDeviceInfo
        logger.info("Device Manager initialized")
    
    def discover_devices(self) -> List[Tuple[int, str, str]]:
        """
        Discover available DAQ devices on the system.
        
        Returns:
            List of tuples: (board_num, product_name, unique_id)
        """
        discovered = []
        
        try:
            # Scan through board numbers 0-9 to find connected devices
            for board_num in range(10):
                try:
                    # Try to get board name - if it exists, device is configured
                    board_name = ul.get_board_name(board_num)
                    if board_name:
                        daq_dev_info = DaqDeviceInfo(board_num)
                        discovered.append((
                            board_num,
                            daq_dev_info.product_name,
                            daq_dev_info.unique_id
                        ))
                        logger.info(f"Discovered: {daq_dev_info.product_name} "
                                  f"(ID: {daq_dev_info.unique_id}) on board {board_num}")
                except ULError:
                    # No device at this board number, continue
                    continue
        except Exception as e:
            logger.error(f"Error during device discovery: {e}")
        
        logger.info(f"Discovery complete. Found {len(discovered)} device(s)")
        return discovered
    
    def connect_mcc_device(self, board_num: int = 0) -> object:
        """
        Connect to an MCC DAQ device by board number.
        Automatically detects if device supports analog input or temperature input.
        
        Args:
            board_num: Board number to connect to (must be configured in InstaCal)
        
        Returns:
            Device object (MCCDevice or MCCTemperatureDevice instance)
        
        Raises:
            Exception: If device not found or connection fails
        """
        from hardware.mcc_device import MCCDevice
        from hardware.mcc_temperature_device import MCCTemperatureDevice
        
        try:
            # Check if already connected
            if board_num in self.connected_devices:
                logger.warning(f"Board {board_num} already connected")
                return self.connected_devices[board_num]
            
            # Verify device exists by trying to get board name
            board_name = ul.get_board_name(board_num)
            if not board_name:
                raise Exception(f"No device found on board {board_num}")
            
            # Get device info
            daq_dev_info = DaqDeviceInfo(board_num)
            
            # Determine device type and create appropriate object
            if daq_dev_info.supports_analog_input:
                device = MCCDevice(board_num, daq_dev_info)
                logger.info(f"Connected to analog input device: {daq_dev_info.product_name}")
            elif daq_dev_info.supports_temp_input:
                device = MCCTemperatureDevice(board_num, daq_dev_info)
                logger.info(f"Connected to temperature device: {daq_dev_info.product_name}")
            else:
                raise Exception(f"Device {daq_dev_info.product_name} does not support analog or temperature input")
            
            # Store in registry
            self.connected_devices[board_num] = device
            self.device_info[board_num] = daq_dev_info
            
            logger.info(f"Connected to {daq_dev_info.product_name} "
                       f"(ID: {daq_dev_info.unique_id}) on board {board_num}")
            
            return device
            
        except Exception as e:
            logger.error(f"Failed to connect to device on board {board_num}: {e}")
            raise
    
    def setup_daq(self, board_num: int, name: str, sample_rate: int, 
                  low_chan: int, high_chan: int, duration_seconds: float) -> dict:
        """
        Setup a single DAQ device with its configuration.
        
        Args:
            board_num: Board number (0, 1, 2)
            name: Friendly name for logging (e.g., "USB-1608GX-2AO", "USB-TEMP")
            sample_rate: Sample rate in Hz
            low_chan: First channel number
            high_chan: Last channel number
            duration_seconds: Acquisition duration for buffer sizing
        
        Returns:
            Dictionary containing device configuration:
                - device: Device object (MCCDevice or MCCTemperatureDevice)
                - name: Friendly name
                - sample_rate: Sample rate in Hz
                - low_chan: First channel
                - high_chan: Last channel
                - points_per_channel: Expected samples per channel
                - num_channels: Number of channels
                - data: Empty list for data collection
                - timestamps: Empty list for timestamps
        """
        logger.info(f"Setting up Board {board_num} ({name})...")
        
        # Connect to device
        daq = self.connect_mcc_device(board_num=board_num)
        logger.info(f"  Connected: {daq.info.product_name}")
        
        # Calculate points per channel
        points_per_channel = int(sample_rate * duration_seconds)
        num_channels = high_chan - low_chan + 1
        
        logger.info(f"  Config: {sample_rate} Hz, Channels {low_chan}-{high_chan} ({num_channels} channels)")
        logger.info(f"  Expected samples per channel: {points_per_channel}")
        
        # Create configuration dictionary
        config = {
            'device': daq,
            'name': name,
            'sample_rate': sample_rate,
            'low_chan': low_chan,
            'high_chan': high_chan,
            'points_per_channel': points_per_channel,
            'num_channels': num_channels,
            'data': [],
            'timestamps': []
        }
        
        logger.info(f"  Board {board_num} setup complete")
        
        return config

    def get_device(self, board_num: int) -> Optional[object]:
        """
        Get a connected device object by board number.
        
        Args:
            board_num: Board number
        
        Returns:
            Device object or None if not connected
        """
        return self.connected_devices.get(board_num)
    
    def disconnect_device(self, board_num: int):
        """
        Disconnect a specific device and release resources.
        
        Args:
            board_num: Board number to disconnect
        """
        if board_num in self.connected_devices:
            device = self.connected_devices[board_num]
            
            try:
                # Call device cleanup if available #TODO: standardize cleanup method
                if hasattr(device, 'cleanup'):
                    device.cleanup()
                
                # Release the DAQ device
                ul.release_daq_device(board_num)
                
                # Remove from registry
                del self.connected_devices[board_num]
                if board_num in self.device_info:
                    del self.device_info[board_num]
                
                logger.info(f"Disconnected device on board {board_num}")
            except Exception as e:
                logger.error(f"Error disconnecting device on board {board_num}: {e}")
        else:
            logger.warning(f"No device connected on board {board_num}")
    
    def disconnect_all(self):
        """Disconnect all connected devices and release resources."""
        logger.info("Disconnecting all devices...")
        board_nums = list(self.connected_devices.keys())
        
        for board_num in board_nums:
            self.disconnect_device(board_num)
        
        logger.info("All devices disconnected")
    
    def get_connected_devices(self) -> Dict[int, str]:
        """
        Get information about all connected devices.
        
        Returns:
            Dictionary mapping board_num -> product_name
        """
        return {
            board_num: info.product_name 
            for board_num, info in self.device_info.items()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all devices."""
        self.disconnect_all()


# ============================================================================
# Boilerplate for additional DAQ types (future expansion)
# ============================================================================

# Example: For LabJack devices
# def connect_labjack_device(self, device_type: str) -> object:
#     """
#     Connect to a LabJack DAQ device.
#     
#     Args:
#         device_type: LabJack device type (e.g., 'U3', 'U6', 'T7')
#     
#     Returns:
#         LabJackDevice object
#     """
#     from hardware.labjack_device import LabJackDevice
#     
#     try:
#         device = LabJackDevice(device_type)
#         # Store in registry with unique identifier
#         self.connected_devices[f'labjack_{device_type}'] = device
#         logger.info(f"Connected to LabJack {device_type}")
#         return device
#     except Exception as e:
#         logger.error(f"Failed to connect to LabJack {device_type}: {e}")
#         raise

# Example: For NI DAQmx devices
# def connect_ni_device(self, device_name: str) -> object:
#     """
#     Connect to a National Instruments DAQmx device.
#     
#     Args:
#         device_name: NI device name (e.g., 'Dev1')
#     
#     Returns:
#         NIDevice object
#     """
#     from hardware.ni_device import NIDevice
#     
#     try:
#         device = NIDevice(device_name)
#         self.connected_devices[f'ni_{device_name}'] = device
#         logger.info(f"Connected to NI device {device_name}")
#         return device
#     except Exception as e:
#         logger.error(f"Failed to connect to NI device {device_name}: {e}")
#         raise

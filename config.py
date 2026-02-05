"""
Configuration settings for LIF DAQ acquisition system.
Modify these values to customize system behavior.
"""
import logging
from mcculw.enums import ScanOptions, ULRange

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = logging.INFO
LOG_FOLDER = "logs"
DATA_FOLDER = "data"
LOG_TO_FILE = True

# ============================================================================
# MCC DAQ Configuration (USB-1608 Series)
# ============================================================================
# Supported device IDs for MCC USB-1608 series
# USB-1608FS = 130, USB-1608G = 110, USB-1608GX-2AO = 111, USB-1604HS = 169
MCC_DEVICE_IDS = [111, 130, 169]  # USB-1608GX-2AO, USB-1608FS, USB-1604HS
MCC_DEFAULT_BOARD_NUM = 0

# Default scan parameters
DEFAULT_SAMPLE_RATE = 1000  # Hz
DEFAULT_POINTS_PER_CHANNEL = 1000
DEFAULT_SCAN_OPTIONS = ScanOptions.BACKGROUND | ScanOptions.SCALEDATA
DEFAULT_VOLTAGE_RANGE = ULRange.BIP10VOLTS  # ±10V

# Channel configuration
DEFAULT_LOW_CHANNEL = 0
DEFAULT_HIGH_CHANNEL = 3
DEFAULT_NUM_CHANNELS = 4

# ============================================================================
# Additional Device-Specific Settings
# ============================================================================
# Note: The devices share common settings above
# Device-specific configurations can be added here if needed 
MCC_DEFAULT_BOARD_NUM = 0

# Default scan parameters
DEFAULT_SAMPLE_RATE = 100000  # Hz, Up to 500 kS/s
DEFAULT_POINTS_PER_CHANNEL = 10000
DEFAULT_SCAN_OPTIONS = ScanOptions.BACKGROUND | ScanOptions.SCALEDATA
DEFAULT_VOLTAGE_RANGE = ULRange.BIP10VOLTS  # ±10V of ±10 V, ±5 V, ±2 V, ±1 V input ranges

# Channel configuration
# Assuming we're using Differential Mode for USB-1608 instead of single-ended mode
DEFAULT_CHANNEL_FIRST = 0
DEFAULT_CHANNEL_LAST = 7
DEFAULT_NUM_CHANNELS = 8

# 2AO analog outputs
NUM_ANALOG_OUTPUTS = 2
DEFAULT_AO_VOLTAGE_RANGE = ULRange.BIP10VOLTS  # ±5V for analog outputs
DEFAULT_AO_CHANNEL = 0  # AO0 or AO1

# For output scanning
AO_SCAN_RATE = 1000  # Hz
AO_POINTS_PER_CHANNEL = 1000

# # ============================================================================
# # MCC DAQ Configuration (USB-1604 Series)
# # ============================================================================
# MCC_DEVICE_IDS = [110, 111, 112]  # Update with your specific model IDs
# MCC_DEFAULT_BOARD_NUM = 0

# # Default scan parameters
# DEFAULT_SAMPLE_RATE = 1000  # Hz
# DEFAULT_POINTS_PER_CHANNEL = 1000
# DEFAULT_SCAN_OPTIONS = ScanOptions.BACKGROUND | ScanOptions.SCALEDATA
# DEFAULT_VOLTAGE_RANGE = ULRange.BIP10VOLTS  # ±10V

# # Channel configuration
# DEFAULT_LOW_CHANNEL = 0
# DEFAULT_HIGH_CHANNEL = 3
# DEFAULT_NUM_CHANNELS = 4

# ============================================================================
# GUI Configuration
# ============================================================================
GUI_REFRESH_RATE_MS = 100  # Milliseconds between GUI updates
PLOT_MAX_POINTS = 1000  # Maximum points to display in real-time plot
DATA_PRECISION = 3  # Decimal places for display

# ============================================================================
# Data Storage Configuration
# ============================================================================
DATA_FILE_PREFIX = "mcc_acquisition"
DATA_FILE_EXTENSION = "csv"
CSV_DELIMITER = ","
INCLUDE_TIMESTAMP_IN_DATA = True

# ============================================================================
# Acquisition Workflow Settings
# ============================================================================
AUTO_START_ON_CONNECT = False
CONTINUOUS_ACQUISITION = True
MAX_ACQUISITION_DURATION = 3600  # Seconds (1 hour)
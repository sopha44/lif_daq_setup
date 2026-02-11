"""
Test script to continuously read Board 1 analog inputs.
Reads all 4 channels every 0.5 seconds and prints the values.
"""
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.setup_mcc_path import setup_mcc_path
setup_mcc_path()

from mcculw import ul
from mcculw.enums import ULRange
from mcculw.device_info import DaqDeviceInfo


def main():
    """Read and display Board 1 analog input values continuously."""
    board_num = 1
    low_chan = 0
    high_chan = 3
    ai_range = ULRange.BIP10VOLTS
    
    print("=" * 70)
    print("Board 1 Analog Input Test - Reading CH0-CH3 every 0.5 seconds")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        # Get device info
        board_name = ul.get_board_name(board_num)
        if not board_name:
            print(f"ERROR: No device found on board {board_num}")
            print("Make sure the device is connected and configured in InstaCal")
            return
        
        daq_dev_info = DaqDeviceInfo(board_num)
        print(f"Connected to: {daq_dev_info.product_name}")
        print(f"Channels: {low_chan}-{high_chan}")
        print(f"Range: {ai_range.name}")
        print()
        
        # Continuous reading loop
        count = 0
        while True:
            count += 1
            values = []
            
            # Read each channel
            for chan in range(low_chan, high_chan + 1):
                value = ul.a_in(board_num, chan, ai_range)
                # Convert to voltage
                voltage = ul.to_eng_units(board_num, ai_range, value)
                values.append(voltage)
            
            # Print values
            timestamp = time.strftime("%H:%M:%S")
            ch_str = ", ".join([f"CH{i}={values[i]:.3f}V" for i in range(len(values))])
            print(f"[{count:4d}] {timestamp} - {ch_str}")
            
            # Wait 0.5 seconds
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Test stopped by user")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

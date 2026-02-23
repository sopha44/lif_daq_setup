"""
Test script for USB-1604HS buffer overrun detection.
This script starts a background scan with a specified buffer size (points_per_channel),
polls the first entry of the buffer immediately after the scan starts and again after the scan finishes,
prints both values, and compares them to detect if a buffer overrun occurred (i.e., if the oldest data was overwritten).
User can specify number of channels, sampling rate, delay, and a set of points_per_channel values to test.
"""


import sys
import time
# Setup MCC DLL path before importing mcculw
from setup_mcc_path import setup_mcc_path
setup_mcc_path()
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.mcc_device import MCCDevice
from mcculw.device_info import DaqDeviceInfo
from mcculw import ul
import config
import numpy as np


def test_buffer_size(board_num, num_chans, points_per_channel_list, delay=0.1, rate=1_000_000):
    info = DaqDeviceInfo(board_num)
    print(f"Testing buffer sizes on board {board_num} ({info.product_name}) with {num_chans} channels at {rate} Hz...")
    max_success = None
    for pts in points_per_channel_list:
        try:
            print(f"Testing points_per_channel={pts}...", end=" ")
            dev = MCCDevice(board_num, info)
            dev.start_scan(
                low_chan=0,
                high_chan=num_chans-1,
                points_per_channel=pts,
                rate=rate
            )
            # Poll first entry right after scan starts
            time.sleep(0.05)
            first_data = dev.read_buffer_data(num_samples=1)
            if first_data is None or len(first_data) == 0:
                print("FAILED (could not read first row)")
                dev.stop_scan()
                break
            print(f"First row at scan start: {first_data[0]}")

            # Wait for scan to finish
            while True:
                status, curr_count, curr_index = dev.get_status()
                if status.name == 'IDLE':
                    break
                time.sleep(0.05)

            # Poll first entry after scan finishes
            end_data = dev.read_buffer_data(num_samples=1)
            dev.stop_scan()
            if end_data is None or len(end_data) == 0:
                print("FAILED (could not read last row)")
                break
            print(f"First row at scan end:   {end_data[0]}")

            overrun = False
            if not np.allclose(end_data[0], first_data[0]):
                overrun = True
            if overrun:
                print("OVERRUN DETECTED")
                break
            print("SUCCESS")
            max_success = pts
        except Exception as e:
            print(f"FAILED ({e})")
            break
    if max_success:
        print(f"\nMaximum successful points_per_channel: {max_success}")
    else:
        print("\nNo successful buffer allocation.")

def main():
    # USB-1604HS typical settings
    board_num = 1
    num_chans = 4  # Set number of channels to test
    delay = 3    # Delay between scans in seconds
    rate = 1_000_000  # 1 MHz sampling rate
    # Define the list of points_per_channel values to test
    points_list = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 5000000, 10000000, 100000000]

    test_buffer_size(board_num, num_chans, points_list, delay, rate)

if __name__ == "__main__":
    main()

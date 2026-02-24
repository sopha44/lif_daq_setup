
from acquisition_controller import AcquisitionController
import config

#TODO: implement MCC scan with EXTTRIGGER and RETRIGMODE, takes re-arming trigger logic out of python and onto DAQ.
# See a_in_scan() https://github.com/mccdaq/mcculw/blob/master/mcculw/ul.py

def main():
    """Main entry point for running acquisition (slow capture)."""
    controller = AcquisitionController()
    # Setup DAQs

    # Board 0 as USB-TEMP-AI
    controller.setup_daq(
        board_num=0,
        name="USB-TEMP-AI",
        sample_rate=1,
        low_chan=1,
        high_chan=3,
    )

    # Board 1 as USB-1604HS-2AO
    controller.setup_daq(
        board_num=1,
        name="USB-1604HS-2AO",
        sample_rate=1_000_000,
        low_chan=0,
        high_chan=3,
        trigger_channel=0,
        trigger_voltage_high=1.0,
        acquisition_window_us=300,
    )

    # Configuration below only for high speed DAQ USB-1604HS-2AO (board 1)
    controller.daq_configs[1]['integrate_scan'] = True  # integrate samples over acquisition window to reduce data rate
    controller.daq_configs[1]['rows_before_write'] = 100  # write to disk every 10k rows to avoid memory issues during long acquisitions
    # Lower points_per_channel to 100000 for board 1
    controller.daq_configs[1]['points_per_channel'] = 100000
    # Explicitly set ai_range and update device config
    from mcculw.enums import ULRange
    controller.daq_configs[1]['ai_range'] = ULRange.BIP10VOLTS
    controller.daq_configs[1]['device'].configure_scan(
        ai_range=controller.daq_configs[1]['ai_range']
    )
    
    # Start acquisition
    controller.start_acquisition(
        restart_scan_each_trigger=False,
        check_buffer_every=0,
        trigger_check_decimation=1,
        time_between_points=0.0,
        total_duration_minutes=1,
        max_rows_in_memory=10000000 # number of rows to have available for scanning data into
    )

if __name__ == '__main__':
    main()

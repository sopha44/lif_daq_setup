from acquisition_controller import AcquisitionController
from utils.logging_setup import setup_logging
import config
import logging

logger = logging.getLogger(__name__)

def main():
    """Main entry point for running acquisition (slow capture)."""
    setup_logging(log_level=config.LOG_LEVEL, log_to_file=config.LOG_TO_FILE)
    controller = AcquisitionController()
    try:
        # Setup each DAQ with its specific configuration

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
            restart_scan_each_trigger=True,  # Keep scan running, just check for trigger condition
            check_buffer_every=1000,  # 0 = disabled, >0 = check buffer every N cycles
            trigger_check_decimation=10,  # Check every 10th sample = 100,000 checks/sec
            time_between_points=30.0,  # Wait 30 seconds between measurement cycles
            total_duration_minutes=120  # Run for 120 minutes, 2 hours (None = run indefinitely)
        )

    except KeyboardInterrupt:
        logger.info("Acquisition interrupted by user")
    except Exception as e:
        logger.error(f"Acquisition failed: {e}")
        raise

if __name__ == '__main__':
    main()

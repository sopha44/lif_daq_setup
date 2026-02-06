"""
DAQ Performance Benchmark
Tests maximum sustainable sample rates for the USB-1604HS-2AO.
"""
import sys
from pathlib import Path
import time
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.setup_mcc_path import setup_mcc_path
setup_mcc_path()

from device_manager import DeviceManager
from utils.logging_setup import setup_logging, get_logger
import config

from mcculw import ul
from mcculw.enums import FunctionType

logger = get_logger(__name__)


def benchmark_sample_rate(device, board_num: int, sample_rate: int, num_channels: int = 3, 
                          duration: float = 5.0):
    """
    Benchmark a specific sample rate.
    
    Args:
        device: Already connected device object
        board_num: Board number of the DAQ
        sample_rate: Sample rate in Hz to test
        num_channels: Number of channels (default 3)
        duration: Test duration in seconds
    
    Returns:
        Dictionary with benchmark results
    """
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {sample_rate:,} Hz with {num_channels} channels")
        logger.info(f"Expected throughput: {sample_rate * num_channels:,} samples/sec")
        logger.info(f"Duration: {duration}s")
        logger.info(f"{'='*70}")
        
        # Setup DAQ with larger buffer (10x expected to handle Python overhead)
        buffer_multiplier = 10
        points_per_channel = int(sample_rate * duration * buffer_multiplier)
        
        # Start acquisition
        device.start_scan(
            low_chan=0,
            high_chan=num_channels - 1,
            rate=sample_rate,
            points_per_channel=points_per_channel
        )
        
        # Start timing AFTER scan starts and do first read immediately
        start_time = time.time()
        
        # Collect data
        total_scans = 0
        total_reads = 0
        max_buffer_usage = 0
        errors = []
        last_status_check = start_time
        first_read_logged = False
        
        while time.time() - start_time < duration:
            try:
                new_data = device.read_new_data()
                num_new = len(new_data)
                total_scans += num_new
                total_reads += 1
                
                # Log first read for debugging
                if not first_read_logged and num_new > 0:
                    logger.info(f"First read got {num_new} scans at time {time.time() - start_time:.3f}s")
                    first_read_logged = True
                
                # Check buffer status only every 0.5 seconds to reduce overhead
                current_time = time.time()
                if current_time - last_status_check >= 0.5:
                    status, curr_count, curr_index = ul.get_status(
                        device.board_num, 
                        FunctionType.AIFUNCTION
                    )
                    
                    # Calculate buffer usage percentage (total_count is the buffer size)
                    buffer_usage = (curr_count / device.total_count) * 100
                    max_buffer_usage = max(max_buffer_usage, buffer_usage)
                    last_status_check = current_time
                
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Error during data read: {e}")
                break
            
            # Small sleep to prevent excessive overhead (matches acquisition_controller)
            time.sleep(0.001)
        
        # Read any remaining data in the buffer before stopping
        final_data = device.read_new_data()
        total_scans += len(final_data)
        if len(final_data) > 0:
            logger.debug(f"Final read collected {len(final_data)} scans")
        
        # Stop scan
        device.stop_scan()
        elapsed = time.time() - start_time
        
        # Calculate metrics
        expected_scans = int(sample_rate * duration)  # scans per channel = sample rate * duration
        actual_sample_rate = total_scans / elapsed
        throughput = (total_scans * num_channels) / elapsed  # total values per second
        sample_loss_pct = ((expected_scans - total_scans) / expected_scans) * 100 if expected_scans > 0 else 0
        
        # Results
        results = {
            'sample_rate': sample_rate,
            'num_channels': num_channels,
            'duration': elapsed,
            'expected_scans': expected_scans,
            'actual_scans': total_scans,
            'sample_loss_pct': sample_loss_pct,
            'actual_sample_rate': actual_sample_rate,
            'throughput': throughput,
            'total_reads': total_reads,
            'max_buffer_usage_pct': max_buffer_usage,
            'errors': errors,
            'success': len(errors) == 0 and sample_loss_pct < 5.0  # Allow up to 5% loss
        }
        
        # Log results
        logger.info(f"\n{'='*70}")
        logger.info(f"RESULTS for {sample_rate:,} Hz:")
        logger.info(f"  Expected scans:   {expected_scans:,}")
        logger.info(f"  Actual scans:     {total_scans:,}")
        logger.info(f"  Sample loss:      {sample_loss_pct:.2f}%")
        logger.info(f"  Actual rate:      {actual_sample_rate:,.0f} Hz")
        logger.info(f"  Throughput:       {throughput:,.0f} values/sec")
        logger.info(f"  Read calls:       {total_reads}")
        logger.info(f"  Max buffer usage: {max_buffer_usage:.1f}%")
        logger.info(f"  Status:           {'✓ PASS' if results['success'] else '✗ FAIL'}")
        
        if errors:
            logger.error(f"  Errors: {errors}")
        
        logger.info(f"{'='*70}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return None


def run_full_benchmark(board_num: int = 2):
    """
    Run comprehensive benchmark across multiple sample rates.
    
    Args:
        board_num: Board number to test (default 2 for USB-1604HS)
    """
    # Test sample rates (Hz)
    test_rates = [
        1_000,      # 1 kHz - your current rate
        10_000,     # 10 kHz
        50_000,     # 50 kHz
        100_000,    # 100 kHz
        250_000,    # 250 kHz
        500_000,    # 500 kHz
        750_000,    # 750 kHz
        1_000_000,  # 1 MHz
        1_330_000,  # 1.33 MHz - maximum
    ]
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# DAQ PERFORMANCE BENCHMARK")
    logger.info(f"# Board: {board_num}")
    logger.info(f"# Channels: 3")
    logger.info(f"# Test duration: 5 seconds per rate")
    logger.info(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*70}\n")
    
    # Connect to device once and reuse
    manager = DeviceManager()
    device = manager.connect_mcc_device(board_num=board_num)
    logger.info(f"Connected to {device.info.product_name}\n")
    
    results = []
    
    for rate in test_rates:
        result = benchmark_sample_rate(
            device=device,
            board_num=board_num,
            sample_rate=rate,
            num_channels=3,
            duration=5.0
        )
        
        if result:
            results.append(result)
            
            # Stop testing if we hit failures
            if not result['success']:
                logger.warning(f"Failed at {rate:,} Hz - stopping benchmark")
                break
        
        # Pause between tests
        time.sleep(1.0)
    
    # Disconnect device
    manager.disconnect_all()
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"BENCHMARK SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Rate (Hz)':<15} {'Status':<10} {'Loss %':<10} {'Throughput':<20}")
    logger.info(f"{'-'*70}")
    
    for r in results:
        status = '✓ PASS' if r['success'] else '✗ FAIL'
        logger.info(f"{r['sample_rate']:>14,} {status:<10} {r['sample_loss_pct']:>8.2f}% "
                   f"{r['throughput']:>18,.0f} val/s")
    
    logger.info(f"{'='*70}")
    
    return results


def main():
    """Main entry point."""
    setup_logging(log_level=config.LOG_LEVEL, log_to_file=False)
    
    logger.info("Starting DAQ performance benchmark...")
    
    try:
        results = run_full_benchmark(board_num=2) # USB-1604HS-2AO is board 2 in setup
        
        logger.info("\nBenchmark complete!")
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)


if __name__ == '__main__':
    main()

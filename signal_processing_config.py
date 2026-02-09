"""
Signal Processing Configuration for DAQ devices.
Define processing blocks for each board separately from acquisition logic.
"""
from signal_processor import SignalProcessor


def configure_board1_processor(processor: SignalProcessor):
    """
    Configure signal processing for Board 1 (USB-TEMP).
    Simple - no processing, just pass through all data.
    
    Args:
        processor: SignalProcessor instance for Board 1
    """
    # No blocks needed - just save raw temperature data
    processor.set_write_sources("ch0")
    processor.build_pipeline()


def configure_board2_processor(processor: SignalProcessor):
    """
    Configure signal processing for Board 2 (USB-1604HS-2AO).
    Implements: CH0 trigger at 1.0V → counter → decimate by 600 → write all 4 channels
    
    Args:
        processor: SignalProcessor instance for Board 2
    """
    # Step 1: Trigger when CH0 >= 1.0V
    processor.add_block(
        "CombiTrig01",
        SignalProcessor.trigger_block,
        condition="ch0 >= 1.0"
    )
    
    # Step 2: Count trigger events
    processor.add_block(
        "Counter00",
        SignalProcessor.counter_block,
        input_key='CombiTrig01'
    )
    
    # Step 3: Decimate - only True every 600th count
    processor.add_block(
        "Formula00",
        SignalProcessor.formula_block,
        equation="(Counter00 % 600) == 0"
    )
    
    # Step 4: Final decision - trigger AND decimation
    processor.add_block(
        "BitLogic00",
        SignalProcessor.bit_logic_block,
        operation='AND',
        input_keys=['CombiTrig01', 'Formula00']
    )
    
    # Relays: Control which channels get written (matching DASYLab layout)
    # -----------------------------------------------------------------
    
    # Relay01: Controls CH0
    processor.add_block(
        "Relay01",
        SignalProcessor.relay_block,
        relay_index=0,
        condition="np.any(BitLogic00)"
    )
    
    # Relay02: Controls CH1
    processor.add_block(
        "Relay02",
        SignalProcessor.relay_block,
        relay_index=1,
        condition="np.any(BitLogic00)"
    )
    
    # Relay03: Controls CH2
    processor.add_block(
        "Relay03",
        SignalProcessor.relay_block,
        relay_index=2,
        condition="np.any(BitLogic00)"
    )
    
    # Relay04: Controls CH3
    processor.add_block(
        "Relay04",
        SignalProcessor.relay_block,
        relay_index=3,
        condition="np.any(BitLogic00)"
    )
    
    # Configure what gets written (all 4 channels only - not intermediate blocks)
    processor.set_write_sources("ch0", "ch1", "ch2", "ch3")
    
    # Map each channel to its controlling relay
    processor.set_relay_channel_map(
        ch0="Relay01",
        ch1="Relay02",
        ch2="Relay03",
        ch3="Relay04"
    )
    
    # Build the pipeline
    processor.build_pipeline()


def configure_all_processors(controller):
    """
    Configure signal processors for all boards.
    Call this after setup_daq() and before start_acquisition().
    
    Args:
        controller: AcquisitionController instance
    """
    # Configure Board 1 (USB-TEMP) if it exists
    try:
        processor = controller.get_processor(1)
        configure_board1_processor(processor)
    except KeyError:
        pass  # Board 1 not configured or processing disabled
    
    # Configure Board 2 (USB-1604HS-2AO) if it exists
    try:
        processor = controller.get_processor(2)
        configure_board2_processor(processor)
    except KeyError:
        pass  # Board 2 not configured or processing disabled

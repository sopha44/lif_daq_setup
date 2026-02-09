"""
Signal Processor for real-time data processing during acquisition.
Implements DASYLab-style processing blocks: formulas, triggers, counters, bit logic, relays.

Architecture:
    1. Block functions: Individual processing units (formulas, triggers, counters, etc.)
    2. Block configuration: Stores parameters for each block
    3. Pipeline: Chains blocks together with input/output routing
    4. Write routing: Collects data destined for Write00 block
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class ProcessingBlock:
    """Base class for a DASYLab-style processing block."""
    
    def __init__(self, name: str, block_type: str):
        """
        Initialize processing block.
        
        Args:
            name: Block name (e.g., "Formula00", "CombiTrig01")
            block_type: Block type (e.g., "formula", "trigger", "counter")
        """
        self.name = name
        self.block_type = block_type
        self.inputs = {}  # Input connections: {input_name: source_block}
        self.outputs = {}  # Cached output data
        self.config = {}  # Block-specific configuration
    
    def configure(self, **kwargs):
        """Configure block parameters."""
        self.config.update(kwargs)
    
    def connect_input(self, input_name: str, source_block: 'ProcessingBlock', output_name: str = 'output'):
        """Connect an input to another block's output."""
        self.inputs[input_name] = (source_block, output_name)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through this block.
        
        Args:
            data: Input data dictionary
        
        Returns:
            Output data dictionary
        """
        raise NotImplementedError("Subclasses must implement process()")


class SignalProcessor:
    """
    Real-time signal processor for DAQ data.
    Manages a pipeline of processing blocks similar to DASYLab.
    """
    
    def __init__(self, sample_rate: int, num_channels: int):
        """
        Initialize signal processor.
        
        Args:
            sample_rate: Sample rate in Hz
            num_channels: Number of input channels
        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        
        # Processing blocks registry
        self.blocks: Dict[str, ProcessingBlock] = {}
        self.pipeline_order: List[str] = []  # Execution order
        
        # Global state (for blocks that need state persistence)
        self.state = {
            'counter_value': 0,
            'relay_states': [False, False, False, False],
            'last_trigger': False
        }
        
        # Output routing for Write00 block
        self.write_sources: List[str] = []  # Block names to write
        self.keep_data_block: Optional[str] = None  # Block that controls data retention
        self.relay_channel_map: Dict[str, str] = {}  # Maps channel/block to relay that controls it
        
        logger.info(f"Signal Processor initialized: {sample_rate} Hz, {num_channels} channels")
    
    # ========================================================================
    # Block Management
    # ========================================================================
    
    def add_block(self, block_name: str, block_function: Callable, **config):
        """
        Add a processing block to the pipeline.
        
        Args:
            block_name: Unique name for the block (e.g., "Formula00", "CombiTrig01")
            block_function: Function that processes data for this block
            **config: Configuration parameters for the block
        
        Example:
            processor.add_block("Formula00", formula_block, equation="ch0 * 6.28")
            processor.add_block("CombiTrig01", trigger_block, threshold=5.0, channel=0)
        """
        self.blocks[block_name] = {
            'function': block_function,
            'config': config,
            'inputs': [],  # List of (source_block, output_key) tuples
            'outputs': {}  # Cached results
        }
        logger.info(f"Added block: {block_name}")
    
    def connect_blocks(self, source_block: str, dest_block: str, output_key: str = 'output'):
        """
        Connect output of source block to input of destination block.
        
        Args:
            source_block: Name of source block
            dest_block: Name of destination block  
            output_key: Which output from source to use (default: 'output')
        """
        if dest_block not in self.blocks:
            raise ValueError(f"Destination block '{dest_block}' not found")
        if source_block not in self.blocks and source_block != 'INPUT':
            raise ValueError(f"Source block '{source_block}' not found")
        
        self.blocks[dest_block]['inputs'].append((source_block, output_key))
        logger.info(f"Connected: {source_block} -> {dest_block}")
    
    def set_write_sources(self, *block_names: str):
        """
        Configure which blocks should output to Write00.
        
        Args:
            *block_names: Names of blocks whose outputs should be written
        
        Example:
            processor.set_write_sources("Formula00", "Formula01", "CombiTrig01")
        """
        self.write_sources = list(block_names)
        logger.info(f"Write sources: {self.write_sources}")
    
    def set_keep_data_block(self, block_name: str):
        """
        Set which block determines whether to keep/discard data.
        
        Args:
            block_name: Block that outputs True/False for keep decision
        
        Example:
            processor.set_keep_data_block("BitLogic00")  # Only save when BitLogic00 is True
        """
        self.keep_data_block = block_name
        logger.info(f"Keep data controlled by: {block_name}")
    
    def set_relay_channel_map(self, **mappings):
        """
        Map channels/blocks to relays that control whether they get written.
        
        Args:
            **mappings: channel/block_name="RelayXX" pairs
        
        Example:
            processor.set_relay_channel_map(
                ch0="Relay01",
                ch1="Relay02",
                ch2="Relay03",
                Formula00="Relay01"
            )
        """
        self.relay_channel_map = mappings
        logger.info(f"Relay-channel mapping: {mappings}")
    
    def build_pipeline(self):
        """
        Determine execution order based on block dependencies.
        Uses topological sort to ensure blocks execute in correct order.
        """
        # Simple execution: process in order blocks were added
        # TODO: Implement proper dependency-based ordering if needed
        self.pipeline_order = list(self.blocks.keys())
        logger.info(f"Pipeline order: {self.pipeline_order}")
    
    # ========================================================================
    # Batch Processing
    # ========================================================================
    
    def process_batch(self, scans: List[List[float]], timestamps: List[float]) -> Dict[str, Any]:
        """
        Process a batch of scans through the configured pipeline.
        
        Args:
            scans: List of scans, each scan is [ch0, ch1, ch2, ...]
            timestamps: List of timestamps for each scan
        
        Returns:
            Dictionary with:
                - raw_data: Original scans
                - write_data: Data from blocks routed to Write00
                - block_outputs: All block outputs
                - keep_data: Whether to save this batch
        """
        if len(scans) == 0:
            return {'keep_data': False, 'write_data': {}}
        
        # Convert to numpy for vectorized processing
        data_array = np.array(scans)  # Shape: (num_scans, num_channels)
        
        # Initial data context (available to all blocks)
        # Dynamically create ch0, ch1, ch2, ... chN for all channels
        context = {
            'INPUT': {
                'raw': data_array,
                'timestamps': timestamps,
                'num_scans': len(scans)
            }
        }
        
        # Add all channels dynamically
        for ch_idx in range(self.num_channels):
            context['INPUT'][f'ch{ch_idx}'] = data_array[:, ch_idx]
        
        # Execute pipeline in order
        for block_name in self.pipeline_order:
            block = self.blocks[block_name]
            
            # Gather inputs for this block
            block_inputs = context['INPUT'].copy()
            
            # Add outputs from connected blocks
            for source_name, output_key in block['inputs']:
                if source_name in context:
                    block_inputs[source_name] = context[source_name]
            
            # Also add ALL previous block outputs to namespace (for formula/relay blocks)
            for prev_block in self.pipeline_order:
                if prev_block == block_name:
                    break  # Don't include future blocks
                if prev_block in context:
                    block_inputs[prev_block] = context[prev_block]
            
            # Execute block function
            try:
                # Inject block_name into config so blocks can use it for unique state keys
                config_with_name = block['config'].copy()
                config_with_name['block_name'] = block_name
                
                block_output = block['function'](
                    block_inputs, 
                    config_with_name,
                    self.state
                )
                context[block_name] = block_output
                block['outputs'] = block_output
            except Exception as e:
                logger.error(f"Error in block {block_name}: {e}")
                context[block_name] = {'error': str(e)}
        
        # Collect data for Write00 block with relay filtering
        write_data = {}
        for source_name in self.write_sources:
            # Check if source is in top-level context or INPUT context
            source_value = None
            if source_name in context:
                source_value = context[source_name]
            elif source_name in context.get('INPUT', {}):
                source_value = context['INPUT'][source_name]
            else:
                logger.warning(f"Write source '{source_name}' not found in context")
                continue
            
            # Check if this source has a relay controlling it
            if source_name in self.relay_channel_map:
                relay_name = self.relay_channel_map[source_name]
                if relay_name in context:
                    relay_output = context[relay_name].get('output', False)
                    # Only include if relay is True (or any element is True for arrays)
                    if isinstance(relay_output, np.ndarray):
                        relay_state = bool(np.any(relay_output))
                    else:
                        relay_state = bool(relay_output)
                    
                    if relay_state:
                        write_data[source_name] = source_value
                else:
                    logger.warning(f"Relay '{relay_name}' for '{source_name}' not found")
            else:
                # No relay control, always include
                write_data[source_name] = source_value
        
        # Decision: keep data?
        if self.keep_data_block:
            # Use specified block's output to determine keep/discard
            if self.keep_data_block in context:
                block_output = context[self.keep_data_block].get('output')
                # Check if ANY sample in batch should be kept
                if isinstance(block_output, np.ndarray):
                    keep_data = bool(np.any(block_output))
                else:
                    keep_data = bool(block_output)
            else:
                logger.warning(f"Keep data block '{self.keep_data_block}' not found, keeping all data")
                keep_data = True
        else:
            # Default: keep if any write source exists
            keep_data = len(write_data) > 0
        
        return {
            'raw_data': scans,
            'write_data': write_data,
            'block_outputs': context,
            'keep_data': keep_data,
            'num_scans': len(scans)
        }
    
    # ========================================================================
    # DASYLab Block Library (static functions)
    # ========================================================================
    
    @staticmethod
    def formula_block(inputs: Dict, config: Dict, state: Dict) -> Dict[str, Any]:
        """
        Formula block: Apply mathematical expression to channels.
        
        Config:
            - equation: String equation (e.g., "ch0 * 6.28", "ch5 + ch7")
            - input_channels: Optional list of specific channels to expose
        
        Returns:
            {'output': np.ndarray with computed values}
        """
        equation = config.get('equation', 'ch0')
        
        # Build namespace with ALL available channels and block outputs
        namespace = {'np': np}
        for key, value in inputs.items():
            if key.startswith('ch') and key[2:].isdigit():
                namespace[key] = value
            elif isinstance(value, dict) and 'output' in value:
                # Add block outputs (extract the 'output' value from dicts)
                namespace[key] = value['output']
        
        # Safe eval with limited namespace
        # TODO: Replace with proper expression parser for security
        try:
            result = eval(equation, {"__builtins__": {}}, namespace)
            return {'output': result}
        except Exception as e:
            logger.error(f"Formula error in '{equation}': {e}")
            # Return zeros of appropriate shape
            first_channel = next((v for k, v in inputs.items() if k.startswith('ch')), np.array([]))
            return {'output': np.zeros_like(first_channel), 'error': str(e)}
    
    @staticmethod
    def trigger_block(inputs: Dict, config: Dict, state: Dict) -> Dict[str, Any]:
        """
        Trigger block: Detect threshold crossings or conditions.
        
        Config:
            - condition: String condition (e.g., "ch0 > 5.0")
            - edge: 'rising', 'falling', or 'both'
        
        Returns:
            {'output': np.ndarray of bool, 'triggered': bool (if any trigger in batch)}
        """
        condition = config.get('condition', 'ch0 > 5.0')
        ch0 = inputs.get('ch0', np.array([]))
        ch1 = inputs.get('ch1', np.array([]))
        
        namespace = {'ch0': ch0, 'ch1': ch1, 'np': np}
        try:
            trigger_states = eval(condition, {"__builtins__": {}}, namespace)
            triggered = np.any(trigger_states)
            return {'output': trigger_states, 'triggered': triggered}
        except Exception as e:
            logger.error(f"Trigger error: {e}")
            return {'output': np.zeros(len(ch0), dtype=bool), 'triggered': False}
    
    @staticmethod
    def counter_block(inputs: Dict, config: Dict, state: Dict) -> Dict[str, Any]:
        """
        Counter block: Count events (e.g., rising edges).
        
        Config:
            - input_key: Which input to count (block name, default: uses 'output' from inputs)
            - edge: 'rising', 'falling', or 'both'
            - block_name: Automatically injected, used for unique state keys
        
        Returns:
            {'output': int (current count), 'increments': int (events this batch)}
        """
        input_key = config.get('input_key', 'output')
        block_name = config.get('block_name', 'Counter00')
        
        # Create unique state keys for this counter
        counter_key = f'{block_name}_value'
        last_input_key = f'{block_name}_last_input'
        
        # Initialize counter if doesn't exist
        if counter_key not in state:
            state[counter_key] = 0
        
        # Get trigger data from the specified block's output
        if input_key in inputs and isinstance(inputs[input_key], dict):
            trigger_data = inputs[input_key].get('output', np.array([False]))
        else:
            trigger_data = inputs.get('output', np.array([False]))
        
        # Ensure it's a numpy array
        if not isinstance(trigger_data, np.ndarray):
            trigger_data = np.array([trigger_data])
        
        # Count rising edges (False -> True transitions)
        if len(trigger_data) > 1:
            edges = np.diff(trigger_data.astype(int))
            rising_edges = np.sum(edges > 0)
        else:
            # Single value - check if it's a rising edge from last state
            current_state = bool(trigger_data[0] if len(trigger_data) > 0 else False)
            last_state = state.get(last_input_key, False)
            rising_edges = 1 if (current_state and not last_state) else 0
            state[last_input_key] = current_state
        
        state[counter_key] += rising_edges
        
        return {
            'output': state[counter_key],
            'increments': rising_edges
        }
    
    @staticmethod
    def relay_block(inputs: Dict, config: Dict, state: Dict) -> Dict[str, Any]:
        """
        Relay block: Conditional switching based on inputs.
        
        Config:
            - relay_index: Which relay (0-3 for Relay01-04)
            - condition: String condition for switching (can reference block names)
        
        Returns:
            {'output': bool (relay state)}
        """
        relay_index = config.get('relay_index', 0)
        condition = config.get('condition', 'False')
        
        # Build namespace with all available data
        namespace = {'np': np}
        
        # Add all channel data (latest values)
        for key, value in inputs.items():
            if key.startswith('ch') and key[2:].isdigit():
                # Get latest value from channel array
                latest_val = value[-1] if isinstance(value, np.ndarray) and len(value) > 0 else value
                namespace[key] = latest_val
        
        # Add block outputs (use their 'output' key if dict, or value directly)
        for key, value in inputs.items():
            if not key.startswith('ch'):
                if isinstance(value, dict):
                    # Block output - get the 'output' field
                    namespace[key] = value.get('output', False)
                else:
                    namespace[key] = value
        
        # Add state values
        namespace['counter'] = state.get('counter_value', 0)
        
        # Evaluate condition
        try:
            relay_state = bool(eval(condition, {"__builtins__": {}}, namespace))
            state['relay_states'][relay_index] = relay_state
            return {'output': relay_state}
        except Exception as e:
            logger.error(f"Relay error in '{condition}': {e}")
            return {'output': False}
    
    @staticmethod
    def bit_logic_block(inputs: Dict, config: Dict, state: Dict) -> Dict[str, Any]:
        """
        Bit logic block: AND, OR, XOR, NOT operations.
        
        Config:
            - operation: 'AND', 'OR', 'XOR', 'NOT'
            - input_keys: List of input keys to operate on
        
        Returns:
            {'output': np.ndarray of bool}
        """
        operation = config.get('operation', 'AND').upper()
        input_keys = config.get('input_keys', ['ch0', 'ch1'])
        
        # Get input arrays - extract 'output' field from dicts
        arrays = []
        for key in input_keys:
            value = inputs.get(key, np.array([False]))
            # If it's a dict (block output), extract the 'output' field
            if isinstance(value, dict):
                arrays.append(value.get('output', np.array([False])))
            else:
                arrays.append(value)
        
        if operation == 'AND':
            result = np.logical_and.reduce(arrays)
        elif operation == 'OR':
            result = np.logical_or.reduce(arrays)
        elif operation == 'XOR':
            result = arrays[0]
            for arr in arrays[1:]:
                result = np.logical_xor(result, arr)
        elif operation == 'NOT':
            result = np.logical_not(arrays[0])
        else:
            result = np.zeros_like(arrays[0], dtype=bool)
        
        return {'output': result}


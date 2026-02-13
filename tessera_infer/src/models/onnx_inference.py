#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
import onnxruntime as ort
from typing import Tuple, Optional
import threading

logger = logging.getLogger(__name__)

class ONNXInferenceModel:
    """
    ONNX inference wrapper for CPU optimization
    """
    
    def __init__(self, onnx_model_path: str, num_threads: Optional[int] = None):
        """
        Initialize ONNX inference model
        
        Args:
            onnx_model_path: Path to ONNX model file
            num_threads: Number of threads for inference (None for auto)
        """
        self.onnx_model_path = onnx_model_path
        self.num_threads = num_threads
        self.session = None
        self.input_names = None
        self.output_names = None
        self._session_lock = threading.Lock()
        
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize ONNX Runtime inference session with optimizations"""
        try:
            # Configure session options for CPU optimization
            sess_options = ort.SessionOptions()
            
            # Set number of threads
            if self.num_threads is not None:
                sess_options.intra_op_num_threads = self.num_threads
                sess_options.inter_op_num_threads = 1  # Use 1 for inter-op to avoid over-threading
            
            # Enable optimizations
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            
            # Use CPU execution provider with optimizations
            providers = [
                ('CPUExecutionProvider', {
                    'use_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'initial_chunk_size_bytes': 1024 * 1024,  # 1MB initial chunks
                    'max_chunk_size_bytes': 32 * 1024 * 1024,  # 32MB max chunks
                    'initial_growth_chunk_size_bytes': 1024 * 1024,  # 1MB growth
                    'max_dead_bytes_per_chunk': 32 * 1024 * 1024,  # 32MB dead bytes
                })
            ]
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input and output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Input names: {self.input_names}")
            logger.info(f"Output names: {self.output_names}")
            logger.info(f"Using threads: {self.num_threads if self.num_threads else 'auto'}")
            
            # Print model info
            for i, input_meta in enumerate(self.session.get_inputs()):
                logger.info(f"Input {i}: {input_meta.name} - {input_meta.shape} ({input_meta.type})")
            
            for i, output_meta in enumerate(self.session.get_outputs()):
                logger.info(f"Output {i}: {output_meta.name} - {output_meta.shape} ({output_meta.type})")
                
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {str(e)}")
            raise
    
    def forward(self, s2_input: np.ndarray, s1_input: np.ndarray) -> np.ndarray:
        """
        Run inference on the ONNX model
        
        Args:
            s2_input: S2 input array, shape (batch, seq_len, 11)
            s1_input: S1 input array, shape (batch, seq_len, 3)
            
        Returns:
            Output array from the model
        """
        try:
            # Ensure inputs are float32 and contiguous
            s2_input = np.ascontiguousarray(s2_input, dtype=np.float32)
            s1_input = np.ascontiguousarray(s1_input, dtype=np.float32)
            
            # Prepare input dictionary
            ort_inputs = {
                self.input_names[0]: s2_input,  # s2_input
                self.input_names[1]: s1_input   # s1_input
            }
            
            # Run inference
            with self._session_lock:
                ort_outputs = self.session.run(self.output_names, ort_inputs)
            
            return ort_outputs[0]  # Return the first (and only) output
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {str(e)}")
            logger.error(f"S2 input shape: {s2_input.shape}, S1 input shape: {s1_input.shape}")
            raise
    
    def __call__(self, s2_input: np.ndarray, s1_input: np.ndarray) -> np.ndarray:
        """Make the model callable"""
        return self.forward(s2_input, s1_input)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if self.session is None:
            return {}
        
        info = {
            'input_names': self.input_names,
            'output_names': self.output_names,
            'num_threads': self.num_threads,
            'providers': [provider for provider in self.session.get_providers()]
        }
        
        # Add input/output shapes
        info['inputs'] = []
        for input_meta in self.session.get_inputs():
            info['inputs'].append({
                'name': input_meta.name,
                'shape': input_meta.shape,
                'type': input_meta.type
            })
        
        info['outputs'] = []
        for output_meta in self.session.get_outputs():
            info['outputs'].append({
                'name': output_meta.name,
                'shape': output_meta.shape,
                'type': output_meta.type
            })
        
        return info
    
    def cleanup(self):
        """Clean up resources"""
        if self.session:
            # ONNX Runtime handles cleanup automatically
            self.session = None
            logger.info("ONNX session cleaned up")


class BatchedONNXInferenceModel(ONNXInferenceModel):
    """
    Batched ONNX inference model with memory optimization for CPU
    """
    
    def __init__(self, onnx_model_path: str, num_threads: Optional[int] = None, 
                 max_batch_size: int = 1024):
        """
        Initialize batched ONNX inference model
        
        Args:
            onnx_model_path: Path to ONNX model file
            num_threads: Number of threads for inference
            max_batch_size: Maximum batch size for inference
        """
        super().__init__(onnx_model_path, num_threads)
        self.max_batch_size = max_batch_size
        self._preallocated_buffers = {}
        self._buffer_lock = threading.Lock()
    
    def _get_or_create_buffer(self, key: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Get or create a preallocated buffer"""
        with self._buffer_lock:
            if key not in self._preallocated_buffers:
                self._preallocated_buffers[key] = np.empty(shape, dtype=dtype)
            
            buffer = self._preallocated_buffers[key]
            if buffer.shape != shape or buffer.dtype != dtype:
                # Reallocate if shape or dtype changed
                self._preallocated_buffers[key] = np.empty(shape, dtype=dtype)
                buffer = self._preallocated_buffers[key]
            
            return buffer
    
    def forward_batch(self, s2_batch: np.ndarray, s1_batch: np.ndarray) -> np.ndarray:
        """
        Run batched inference with memory optimization
        
        Args:
            s2_batch: S2 batch array, shape (batch, seq_len, 11)
            s1_batch: S1 batch array, shape (batch, seq_len, 3)
            
        Returns:
            Output batch array from the model
        """
        batch_size = s2_batch.shape[0]
        
        if batch_size <= self.max_batch_size:
            # Single batch inference
            return self.forward(s2_batch, s1_batch)
        else:
            # Split into smaller batches
            results = []
            for i in range(0, batch_size, self.max_batch_size):
                end_idx = min(i + self.max_batch_size, batch_size)
                s2_sub = s2_batch[i:end_idx]
                s1_sub = s1_batch[i:end_idx]
                
                result = self.forward(s2_sub, s1_sub)
                results.append(result)
            
            return np.vstack(results)
    
    def cleanup(self):
        """Clean up resources including preallocated buffers"""
        with self._buffer_lock:
            self._preallocated_buffers.clear()
        super().cleanup()
        logger.info("Batched ONNX session and buffers cleaned up")
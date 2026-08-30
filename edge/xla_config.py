"""
Configuration for JAX/XLA Optimization System
Centralized settings for performance tuning and deployment
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class XLAConfig:
    """XLA Compiler Configuration"""
    
    # Enable XLA optimizations
    enable_xla: bool = True
    
    # Platform selection
    platform: Literal['gpu', 'cpu', 'tpu'] = 'gpu'
    
    # Precision settings
    use_float64: bool = False  # False = float32 (faster)
    
    # Memory optimization
    enable_memory_donation: bool = True
    
    # JIT compilation settings
    jit_cache_size: int = 1000
    
    # XLA flags (advanced)
    xla_flags: dict = None
    
    def __post_init__(self):
        if self.xla_flags is None:
            self.xla_flags = {
                'xla_gpu_cuda_data_dir': None,  # Auto-detect
                'xla_gpu_autotune_level': 4,     # Maximum autotuning
                'xla_force_host_platform_device_count': 1,
            }


@dataclass
class InferenceConfig:
    """Inference Engine Configuration"""
    
    # Batch processing
    batch_size: int = 32
    max_batch_size: int = 128
    
    # Input image settings
    input_height: int = 224
    input_width: int = 224
    input_channels: int = 3
    
    # Performance settings
    enable_batching: bool = True
    enable_fusion: bool = True
    enable_memory_optimization: bool = True
    
    # Warmup iterations
    warmup_iterations: int = 10
    
    # Timeout settings
    inference_timeout_ms: int = 5000
    
    @property
    def input_shape(self):
        return (1, self.input_height, self.input_width, self.input_channels)
    
    @property
    def batch_input_shape(self):
        return (self.batch_size, self.input_height, self.input_width, self.input_channels)


@dataclass
class MemoryConfig:
    """Memory Transfer Optimization Configuration"""
    
    # Transfer batching
    enable_batch_transfers: bool = True
    
    # Pinned memory
    use_pinned_memory: bool = True
    
    # Persistent buffers
    enable_persistent_buffers: bool = True
    buffer_pool_size: int = 10
    
    # Memory layout
    preferred_layout: Literal['NCHW', 'NHWC'] = 'NCHW'  # NCHW for GPU, NHWC for CPU


@dataclass
class TFLiteConfig:
    """TFLite Export Configuration"""
    
    # Quantization
    enable_quantization: bool = True
    quantization_type: Literal['float16', 'int8', 'dynamic'] = 'float16'
    
    # Optimization
    enable_optimizations: bool = True
    
    # Target hardware
    target_ops: list = None
    
    # Model size optimization
    enable_select_tf_ops: bool = False
    
    def __post_init__(self):
        if self.target_ops is None:
            self.target_ops = ['TFLITE_BUILTINS']


@dataclass
class ONNXConfig:
    """ONNX Export Configuration"""
    
    # ONNX opset version
    opset_version: int = 13
    
    # Optimization
    enable_optimization: bool = True
    optimization_level: int = 2  # 0=none, 1=basic, 2=extended, 99=all
    
    # Target runtime
    target_runtime: Literal['onnxruntime', 'tensorrt', 'openvino'] = 'onnxruntime'


@dataclass
class BenchmarkConfig:
    """Benchmark Configuration"""
    
    # Test parameters
    num_iterations: int = 100
    num_warmup_iterations: int = 10
    
    # Test data
    num_test_samples: int = 100
    test_batch_sizes: list = None
    
    # Output
    save_results: bool = True
    output_dir: str = "benchmark_results"
    generate_plots: bool = True
    
    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 8, 16, 32, 64]


@dataclass
class OptimizationConfig:
    """Master Configuration for XLA Optimization System"""
    
    # Sub-configurations
    xla: XLAConfig = None
    inference: InferenceConfig = None
    memory: MemoryConfig = None
    tflite: TFLiteConfig = None
    onnx: ONNXConfig = None
    benchmark: BenchmarkConfig = None
    
    # Performance targets
    target_throughput_increase: float = 30.0  # Percentage
    target_latency_ms: Optional[float] = None
    
    # Logging
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    enable_profiling: bool = False
    
    def __post_init__(self):
        # Initialize sub-configs with defaults if not provided
        if self.xla is None:
            self.xla = XLAConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.tflite is None:
            self.tflite = TFLiteConfig()
        if self.onnx is None:
            self.onnx = ONNXConfig()
        if self.benchmark is None:
            self.benchmark = BenchmarkConfig()
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            'xla': self.xla.__dict__,
            'inference': self.inference.__dict__,
            'memory': self.memory.__dict__,
            'tflite': self.tflite.__dict__,
            'onnx': self.onnx.__dict__,
            'benchmark': self.benchmark.__dict__,
            'target_throughput_increase': self.target_throughput_increase,
            'target_latency_ms': self.target_latency_ms,
            'log_level': self.log_level,
            'enable_profiling': self.enable_profiling
        }


# Preset configurations for different use cases

def get_high_throughput_config() -> OptimizationConfig:
    """
    Configuration optimized for maximum throughput
    Best for batch processing and offline inference
    """
    return OptimizationConfig(
        inference=InferenceConfig(
            batch_size=64,
            max_batch_size=256,
        ),
        memory=MemoryConfig(
            enable_batch_transfers=True,
            buffer_pool_size=20,
        ),
        target_throughput_increase=40.0
    )


def get_low_latency_config() -> OptimizationConfig:
    """
    Configuration optimized for minimum latency
    Best for real-time inference and interactive applications
    """
    return OptimizationConfig(
        inference=InferenceConfig(
            batch_size=1,
            max_batch_size=8,
            warmup_iterations=20,
        ),
        memory=MemoryConfig(
            enable_persistent_buffers=True,
        ),
        target_latency_ms=10.0
    )


def get_mobile_deployment_config() -> OptimizationConfig:
    """
    Configuration optimized for mobile/edge deployment
    Focuses on model size and compatibility
    """
    return OptimizationConfig(
        inference=InferenceConfig(
            batch_size=1,
            input_height=224,
            input_width=224,
        ),
        tflite=TFLiteConfig(
            enable_quantization=True,
            quantization_type='int8',
        ),
        memory=MemoryConfig(
            preferred_layout='NHWC',  # Better for mobile CPUs
        )
    )


def get_gpu_optimized_config() -> OptimizationConfig:
    """
    Configuration optimized for GPU inference
    Maximizes GPU utilization and throughput
    """
    return OptimizationConfig(
        xla=XLAConfig(
            platform='gpu',
            use_float64=False,
        ),
        inference=InferenceConfig(
            batch_size=32,
            max_batch_size=128,
        ),
        memory=MemoryConfig(
            preferred_layout='NCHW',  # Better for GPU
            enable_batch_transfers=True,
        ),
        target_throughput_increase=35.0
    )


def get_cpu_optimized_config() -> OptimizationConfig:
    """
    Configuration optimized for CPU inference
    Balances performance and resource usage
    """
    return OptimizationConfig(
        xla=XLAConfig(
            platform='cpu',
        ),
        inference=InferenceConfig(
            batch_size=16,
            max_batch_size=64,
        ),
        memory=MemoryConfig(
            preferred_layout='NHWC',  # Better for CPU
        ),
        target_throughput_increase=25.0
    )


# Default configuration
DEFAULT_CONFIG = OptimizationConfig()


# Example usage
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("XLA OPTIMIZATION CONFIGURATIONS")
    print("="*80)
    
    configs = {
        'default': DEFAULT_CONFIG,
        'high_throughput': get_high_throughput_config(),
        'low_latency': get_low_latency_config(),
        'mobile_deployment': get_mobile_deployment_config(),
        'gpu_optimized': get_gpu_optimized_config(),
        'cpu_optimized': get_cpu_optimized_config(),
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper().replace('_', ' ')} Configuration:")
        print("-" * 80)
        config_dict = config.to_dict()
        print(json.dumps(config_dict, indent=2))
    
    print("\n" + "="*80)
    print("Configuration examples complete")
    print("="*80)

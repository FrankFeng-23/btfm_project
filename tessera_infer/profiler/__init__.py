# profiler/__init__.py

# Conditional imports to avoid dependency issues
try:
    from .tessera_profiler import TesseraProfiler, create_tessera_profiler
    TESSERA_PROFILER_AVAILABLE = True
except ImportError:
    TESSERA_PROFILER_AVAILABLE = False
    TesseraProfiler = None
    create_tessera_profiler = None

try:
    from .unified_profiler import UnifiedProfiler
    UNIFIED_PROFILER_AVAILABLE = True
except ImportError:
    UNIFIED_PROFILER_AVAILABLE = False
    UnifiedProfiler = None

try:
    from .pytorch_profiler import PyTorchProfiler
    PYTORCH_PROFILER_AVAILABLE = True
except ImportError:
    PYTORCH_PROFILER_AVAILABLE = False
    PyTorchProfiler = None

try:
    from .onnx_profiler import ONNXProfiler
    ONNX_PROFILER_AVAILABLE = True
except ImportError:
    ONNX_PROFILER_AVAILABLE = False
    ONNXProfiler = None

try:
    from .metrics import MetricsCollector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    MetricsCollector = None

try:
    from .visualizer import HTMLReportGenerator  
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    HTMLReportGenerator = None

__all__ = [
    'TesseraProfiler',
    'create_tessera_profiler', 
    'UnifiedProfiler',
    'PyTorchProfiler', 
    'ONNXProfiler',
    'MetricsCollector',
    'HTMLReportGenerator',
    'TESSERA_PROFILER_AVAILABLE',
    'UNIFIED_PROFILER_AVAILABLE',
    'PYTORCH_PROFILER_AVAILABLE',
    'ONNX_PROFILER_AVAILABLE',
    'METRICS_AVAILABLE',
    'VISUALIZER_AVAILABLE'
]
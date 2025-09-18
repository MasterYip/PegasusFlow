import cProfile
import pstats
import os
import time
import functools
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
import torch


class TimeProfiler:
    """Time performance profiler for trajectory optimization methods."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.enabled = False
        
    def enable(self):
        """Enable time profiling."""
        self.enabled = True
        
    def disable(self):
        """Disable time profiling."""
        self.enabled = False
        
    def clear(self):
        """Clear all profiling data."""
        self.timing_data.clear()
        self.call_counts.clear()
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for method_name, times in self.timing_data.items():
            if times:
                stats[method_name] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'call_count': len(times),
                    'times': times
                }
        return stats
        
    def print_stats(self, top_n: int = 10):
        """Print timing statistics."""
        stats = self.get_stats()
        if not stats:
            print("No timing data available")
            return
            
        print("\n" + "="*80)
        print("TIME PROFILING RESULTS")
        print("="*80)
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"{'Method':<40} {'Calls':<8} {'Total (s)':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 80)
        
        for method_name, method_stats in sorted_stats[:top_n]:
            print(f"{method_name:<40} "
                  f"{method_stats['call_count']:<8} "
                  f"{method_stats['total_time']:<12.4f} "
                  f"{method_stats['avg_time']*1000:<12.2f} "
                  f"{method_stats['min_time']*1000:<12.2f} "
                  f"{method_stats['max_time']*1000:<12.2f}")


# Global profiler instance
time_profiler = TimeProfiler()


def time_profile(method_name: Optional[str] = None):
    """Decorator for timing method execution.
    
    Args:
        method_name: Optional custom name for the method (defaults to function name)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not time_profiler.enabled:
                return func(*args, **kwargs)
                
            name = method_name or f"{func.__module__}.{func.__qualname__}"
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                time_profiler.timing_data[name].append(execution_time)
                time_profiler.call_counts[name] += 1
                
        return wrapper
    return decorator


def do_cprofile(filename: str):
    """
    Decorator for function profiling using cProfile.
    """
    def wrapper(func):
        @functools.wraps(func)
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = os.getenv("PROFILING")
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper


def print_profile(filename: str):
    """
    Print the profiling results.
    """
    if not os.path.exists(filename):
        print(f"Profile file {filename} not found")
        return
        
    profile = pstats.Stats(filename)
    profile.strip_dirs().sort_stats("cumulative").print_stats(20)
    profile.strip_dirs().sort_stats("time").print_stats(20)
    profile.strip_dirs().sort_stats("calls").print_stats(20)


class BenchmarkContext:
    """Context manager for benchmarking code blocks."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_time = None
        
    def __enter__(self):
        if self.enabled:
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.start_time is not None:
            end_time = time.perf_counter()
            execution_time = end_time - self.start_time
            time_profiler.timing_data[self.name].append(execution_time)
            time_profiler.call_counts[self.name] += 1


def benchmark(name: str):
    """Create a benchmark context manager.
    
    Usage:
        with benchmark("my_operation"):
            # code to benchmark
            pass
    """
    return BenchmarkContext(name, time_profiler.enabled)


class GPUProfiler:
    """GPU memory and timing profiler for PyTorch operations."""
    
    def __init__(self):
        self.gpu_stats = defaultdict(list)
        self.enabled = False
        
    def enable(self):
        """Enable GPU profiling."""
        self.enabled = True
        
    def disable(self):
        """Disable GPU profiling."""
        self.enabled = False
        
    def clear(self):
        """Clear GPU profiling data."""
        self.gpu_stats.clear()
        
    def profile_memory(self, name: str):
        """Profile GPU memory usage."""
        if not self.enabled or not torch.cuda.is_available():
            return
            
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        
        self.gpu_stats[name].append({
            'memory_allocated': memory_allocated,
            'memory_reserved': memory_reserved,
            'timestamp': time.time()
        })
        
    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """Get GPU memory statistics."""
        stats = {}
        for name, measurements in self.gpu_stats.items():
            if measurements:
                allocated = [m['memory_allocated'] for m in measurements]
                reserved = [m['memory_reserved'] for m in measurements]
                
                stats[name] = {
                    'avg_allocated_mb': sum(allocated) / len(allocated) / 1024 / 1024,
                    'max_allocated_mb': max(allocated) / 1024 / 1024,
                    'avg_reserved_mb': sum(reserved) / len(reserved) / 1024 / 1024,
                    'max_reserved_mb': max(reserved) / 1024 / 1024,
                    'measurements': len(measurements)
                }
        return stats
        
    def print_memory_stats(self):
        """Print GPU memory statistics."""
        stats = self.get_memory_stats()
        if not stats:
            print("No GPU memory data available")
            return
            
        print("\n" + "="*80)
        print("GPU MEMORY PROFILING RESULTS")
        print("="*80)
        
        print(f"{'Operation':<40} {'Calls':<8} {'Avg Alloc (MB)':<15} {'Max Alloc (MB)':<15} {'Avg Res (MB)':<15}")
        print("-" * 80)
        
        for name, mem_stats in stats.items():
            print(f"{name:<40} "
                  f"{mem_stats['measurements']:<8} "
                  f"{mem_stats['avg_allocated_mb']:<15.2f} "
                  f"{mem_stats['max_allocated_mb']:<15.2f} "
                  f"{mem_stats['avg_reserved_mb']:<15.2f}")


# Global GPU profiler instance
gpu_profiler = GPUProfiler()


def gpu_profile(method_name: Optional[str] = None):
    """Decorator for GPU memory profiling.
    
    Args:
        method_name: Optional custom name for the method
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = method_name or f"{func.__module__}.{func.__qualname__}"
            
            if gpu_profiler.enabled:
                gpu_profiler.profile_memory(f"{name}_start")
                
            result = func(*args, **kwargs)
            
            if gpu_profiler.enabled:
                gpu_profiler.profile_memory(f"{name}_end")
                
            return result
        return wrapper
    return decorator


def enable_profiling(time_prof: bool = True, gpu_prof: bool = True):
    """Enable profiling globally.
    
    Args:
        time_prof: Enable time profiling
        gpu_prof: Enable GPU profiling
    """
    if time_prof:
        time_profiler.enable()
    if gpu_prof:
        gpu_profiler.enable()


def disable_profiling():
    """Disable all profiling."""
    time_profiler.disable()
    gpu_profiler.disable()


def clear_profiling():
    """Clear all profiling data."""
    time_profiler.clear()
    gpu_profiler.clear()


def print_all_stats():
    """Print all profiling statistics."""
    time_profiler.print_stats()
    gpu_profiler.print_memory_stats()


def save_profiling_results(filepath: str):
    """Save profiling results to file.
    
    Args:
        filepath: Path to save results
    """
    import json
    
    results = {
        'time_stats': time_profiler.get_stats(),
        'gpu_stats': gpu_profiler.get_memory_stats()
    }
    
    # Convert numpy arrays and tensors to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        else:
            return obj
    
    results = convert_for_json(results)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Profiling results saved to {filepath}")


def load_profiling_results(filepath: str) -> Dict[str, Any]:
    """Load profiling results from file.
    
    Args:
        filepath: Path to load results from
        
    Returns:
        Loaded profiling results
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results
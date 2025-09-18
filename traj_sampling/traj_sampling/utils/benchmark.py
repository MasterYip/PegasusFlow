import cProfile
import pstats
import os
import time
import functools
import torch
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager


def do_cprofile(filename):
    """
    Decorator for function profiling using cProfile.
    
    Args:
        filename: Output filename for the profile data
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


def time_profile(name: Optional[str] = None):
    """
    Decorator for timing function execution.
    
    Args:
        name: Optional name for the profiling entry
    """
    def wrapper(func):
        @functools.wraps(func)
        def timed_func(*args, **kwargs):
            # Check if time profiling is enabled
            DO_TIME_PROF = os.getenv("TIME_PROFILING")
            if DO_TIME_PROF:
                profile_name = name or f"{func.__module__}.{func.__name__}"
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                
                # Store timing information
                if not hasattr(time_profile, '_timings'):
                    time_profile._timings = {}
                if profile_name not in time_profile._timings:
                    time_profile._timings[profile_name] = []
                time_profile._timings[profile_name].append(elapsed)
                
                print(f"[TIME_PROFILE] {profile_name}: {elapsed:.6f}s")
            else:
                result = func(*args, **kwargs)
            return result
        return timed_func
    return wrapper


def gpu_profile(name: Optional[str] = None):
    """
    Decorator for GPU memory and timing profiling.
    
    Args:
        name: Optional name for the profiling entry
    """
    def wrapper(func):
        @functools.wraps(func)
        def gpu_profiled_func(*args, **kwargs):
            # Check if GPU profiling is enabled
            DO_GPU_PROF = os.getenv("GPU_PROFILING")
            if DO_GPU_PROF and torch.cuda.is_available():
                profile_name = name or f"{func.__module__}.{func.__name__}"
                
                # Clear cache and synchronize
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Record initial memory
                memory_before = torch.cuda.memory_allocated()
                max_memory_before = torch.cuda.max_memory_allocated()
                
                # Start timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                result = func(*args, **kwargs)
                
                # End timing and synchronize
                end_event.record()
                torch.cuda.synchronize()
                
                # Calculate metrics
                gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                memory_after = torch.cuda.memory_allocated()
                max_memory_after = torch.cuda.max_memory_allocated()
                memory_used = memory_after - memory_before
                max_memory_used = max_memory_after - max_memory_before
                
                # Store GPU profiling information
                if not hasattr(gpu_profile, '_gpu_timings'):
                    gpu_profile._gpu_timings = {}
                if profile_name not in gpu_profile._gpu_timings:
                    gpu_profile._gpu_timings[profile_name] = []
                
                gpu_profile._gpu_timings[profile_name].append({
                    'gpu_time': gpu_time,
                    'memory_used': memory_used,
                    'max_memory_used': max_memory_used
                })
                
                print(f"[GPU_PROFILE] {profile_name}: {gpu_time:.6f}s, "
                      f"Memory: {memory_used / 1024**2:.2f}MB, "
                      f"Max Memory: {max_memory_used / 1024**2:.2f}MB")
            else:
                result = func(*args, **kwargs)
            return result
        return gpu_profiled_func
    return wrapper


@contextmanager
def benchmark(name: str):
    """
    Context manager for benchmarking code blocks.
    
    Args:
        name: Name for the benchmark
    """
    DO_BENCHMARK = os.getenv("BENCHMARKING")
    if DO_BENCHMARK:
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                gpu_time = start_event.elapsed_time(end_event) / 1000.0
                print(f"[BENCHMARK] {name}: GPU {gpu_time:.6f}s")
            
            end_time = time.perf_counter()
            cpu_time = end_time - start_time
            print(f"[BENCHMARK] {name}: CPU {cpu_time:.6f}s")
    else:
        yield


def print_profile(filename):
    """
    Print the profiling results from cProfile.
    
    Args:
        filename: Profile data filename
    """
    if not os.path.exists(filename):
        print(f"Profile file {filename} not found")
        return
        
    profile = pstats.Stats(filename)
    print(f"\n{'='*60}")
    print(f"PROFILE RESULTS: {filename}")
    print(f"{'='*60}")
    
    print(f"\nTop 20 functions by cumulative time:")
    profile.strip_dirs().sort_stats("cumulative").print_stats(20)
    
    print(f"\nTop 20 functions by internal time:")
    profile.strip_dirs().sort_stats("time").print_stats(20)
    
    print(f"\nTop 20 functions by call count:")
    profile.strip_dirs().sort_stats("calls").print_stats(20)


def print_time_profile_summary():
    """Print summary of time profiling results."""
    if not hasattr(time_profile, '_timings') or not time_profile._timings:
        print("No time profiling data available")
        return
    
    print(f"\n{'='*60}")
    print("TIME PROFILING SUMMARY")
    print(f"{'='*60}")
    
    for name, timings in time_profile._timings.items():
        count = len(timings)
        total_time = sum(timings)
        avg_time = total_time / count
        min_time = min(timings)
        max_time = max(timings)
        
        print(f"{name}:")
        print(f"  Calls: {count}")
        print(f"  Total: {total_time:.6f}s")
        print(f"  Average: {avg_time:.6f}s")
        print(f"  Min: {min_time:.6f}s")
        print(f"  Max: {max_time:.6f}s")
        print()


def print_gpu_profile_summary():
    """Print summary of GPU profiling results."""
    if not hasattr(gpu_profile, '_gpu_timings') or not gpu_profile._gpu_timings:
        print("No GPU profiling data available")
        return
    
    print(f"\n{'='*60}")
    print("GPU PROFILING SUMMARY")
    print(f"{'='*60}")
    
    for name, measurements in gpu_profile._gpu_timings.items():
        count = len(measurements)
        
        gpu_times = [m['gpu_time'] for m in measurements]
        memory_used = [m['memory_used'] for m in measurements]
        max_memory_used = [m['max_memory_used'] for m in measurements]
        
        total_gpu_time = sum(gpu_times)
        avg_gpu_time = total_gpu_time / count
        avg_memory = sum(memory_used) / count
        avg_max_memory = sum(max_memory_used) / count
        
        print(f"{name}:")
        print(f"  Calls: {count}")
        print(f"  Total GPU time: {total_gpu_time:.6f}s")
        print(f"  Average GPU time: {avg_gpu_time:.6f}s")
        print(f"  Average memory used: {avg_memory / 1024**2:.2f}MB")
        print(f"  Average max memory: {avg_max_memory / 1024**2:.2f}MB")
        print()


def clear_profile_data():
    """Clear all accumulated profiling data."""
    if hasattr(time_profile, '_timings'):
        time_profile._timings.clear()
    if hasattr(gpu_profile, '_gpu_timings'):
        gpu_profile._gpu_timings.clear()
    print("Profile data cleared")


def save_profile_summary(filepath: str):
    """Save profiling summary to file.
    
    Args:
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        f.write("PERFORMANCE PROFILING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Time profiling summary
        if hasattr(time_profile, '_timings') and time_profile._timings:
            f.write("TIME PROFILING SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for name, timings in time_profile._timings.items():
                count = len(timings)
                total_time = sum(timings)
                avg_time = total_time / count
                min_time = min(timings)
                max_time = max(timings)
                
                f.write(f"{name}:\n")
                f.write(f"  Calls: {count}\n")
                f.write(f"  Total: {total_time:.6f}s\n")
                f.write(f"  Average: {avg_time:.6f}s\n")
                f.write(f"  Min: {min_time:.6f}s\n")
                f.write(f"  Max: {max_time:.6f}s\n\n")
        
        # GPU profiling summary
        if hasattr(gpu_profile, '_gpu_timings') and gpu_profile._gpu_timings:
            f.write("\nGPU PROFILING SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for name, measurements in gpu_profile._gpu_timings.items():
                count = len(measurements)
                
                gpu_times = [m['gpu_time'] for m in measurements]
                memory_used = [m['memory_used'] for m in measurements]
                max_memory_used = [m['max_memory_used'] for m in measurements]
                
                total_gpu_time = sum(gpu_times)
                avg_gpu_time = total_gpu_time / count
                avg_memory = sum(memory_used) / count
                avg_max_memory = sum(max_memory_used) / count
                
                f.write(f"{name}:\n")
                f.write(f"  Calls: {count}\n")
                f.write(f"  Total GPU time: {total_gpu_time:.6f}s\n")
                f.write(f"  Average GPU time: {avg_gpu_time:.6f}s\n")
                f.write(f"  Average memory used: {avg_memory / 1024**2:.2f}MB\n")
                f.write(f"  Average max memory: {avg_max_memory / 1024**2:.2f}MB\n\n")
    
    print(f"Profile summary saved to {filepath}")


# Environment variable helpers
def enable_profiling():
    """Enable all profiling modes."""
    os.environ["PROFILING"] = "1"
    os.environ["TIME_PROFILING"] = "1"
    os.environ["GPU_PROFILING"] = "1"
    os.environ["BENCHMARKING"] = "1"
    print("All profiling modes enabled")


def disable_profiling():
    """Disable all profiling modes."""
    for var in ["PROFILING", "TIME_PROFILING", "GPU_PROFILING", "BENCHMARKING"]:
        if var in os.environ:
            del os.environ[var]
    print("All profiling modes disabled")


def set_profiling_mode(cprofile: bool = False, time_prof: bool = False, 
                      gpu_prof: bool = False, benchmark: bool = False):
    """Set specific profiling modes.
    
    Args:
        cprofile: Enable cProfile profiling
        time_prof: Enable time profiling
        gpu_prof: Enable GPU profiling
        benchmark: Enable benchmarking
    """
    # Clear all first
    disable_profiling()
    
    if cprofile:
        os.environ["PROFILING"] = "1"
    if time_prof:
        os.environ["TIME_PROFILING"] = "1"
    if gpu_prof:
        os.environ["GPU_PROFILING"] = "1"
    if benchmark:
        os.environ["BENCHMARKING"] = "1"
    
    enabled_modes = []
    if cprofile: enabled_modes.append("cProfile")
    if time_prof: enabled_modes.append("Time")
    if gpu_prof: enabled_modes.append("GPU")
    if benchmark: enabled_modes.append("Benchmark")
    
    if enabled_modes:
        print(f"Enabled profiling modes: {', '.join(enabled_modes)}")
    else:
        print("All profiling modes disabled")
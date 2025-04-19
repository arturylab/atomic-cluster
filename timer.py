import time
import functools

def timer(func):
    """
    A decorator that measures and prints the execution time of a function.
    
    Args:
        func: The function to be timed.
        
    Returns:
        A wrapped function that reports execution time and returns the original result.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper
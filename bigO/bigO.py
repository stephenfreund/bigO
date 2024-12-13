import ast
import atexit
import builtins
import hashlib
import inspect
import json
import time
import marshal
import random
import numpy as np

from collections import defaultdict
from functools import lru_cache, wraps
from scipy.stats import linregress

# Global dictionary to store performance data
performance_data = defaultdict(list)

system_name = "bigO"

performance_data_filename = f"{system_name}_data.json"
performance_analysis_filename = f"{system_name}_analysis.json"

wrapped_functions = set()

delay_factor = defaultdict(float)

def monkey_patch_function(obj, func_name, time_dilation=False, space_dilation=False):
    """
    Monkey patch the specified function in the given object.
    If factor > 0, the patched version will time the function and then sleep 
    factor times as long as it took to run.
    """
    import customalloc
    original = obj[func_name]
    def wrapper(*args, **kwargs):
        delay = delay_factor[func_name]
        if delay > 0.0:
            start = time.perf_counter()
            try:
                if space_dilation:
                    customalloc.set_dilation_factor(delay)
                ret = original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                customalloc.set_dilation_factor(1.0)
            if time_dilation:
                time.sleep(elapsed * delay)
        else:
            ret = original(*args, **kwargs)
        return ret

    # Set the wrapped function in place of the original
    obj[func_name] = wrapper
    wrapped_functions.add(func_name)



class FunctionCallFinder(ast.NodeVisitor):
    def __init__(self):
        # Maps function name -> set of functions it calls
        self.call_graph = {}
        self.current_function = None
        self.builtin_names = set(dir(builtins))

    def visit_FunctionDef(self, node):
        func_name = node.name
        self.call_graph[func_name] = set()
        prev_function = self.current_function
        self.current_function = func_name

        self.generic_visit(node)

        self.current_function = prev_function

    def visit_Call(self, node):
        if self.current_function is not None:
            func_node = node.func
            func_name = None
            if isinstance(func_node, ast.Name):
                func_name = func_node.id
            elif isinstance(func_node, ast.Attribute):
                func_name = func_node.attr

            if func_name and func_name not in self.builtin_names:
                self.call_graph[self.current_function].add(func_name)

        self.generic_visit(node)

    @lru_cache(maxsize=None)
    def get_root_functions(self, entry_point):
        """
        Given an entry point function (like linear_function), return the functions
        that appear as "roots" in the call chain starting from it. According to the user's rule:
        
        If a function f transitively calls another function g (for any f and g),
        only list the function f.
        
        This means we look at all functions reachable from 'entry_point' and:
          - Include functions that call at least one other function (directly or transitively)
          - Exclude functions that do not call any other function (leaves)
        """
        # First, find all functions reachable from entry_point
        reachable = self._get_reachable(entry_point)

        # "If f transitively calls g, only list f."
        # This means:
        # - If a function has outgoing calls (transitive calls), it's included.
        
        # non_leaf_functions = [f for f in reachable if self.call_graph[f] and f != entry_point]
        non_leaf_functions = reachable - {entry_point}
        #    [f for f in reachable if f != entry_point]
        # print(f"non leaf functions = {non_leaf_functions}")
        return non_leaf_functions

    def _get_reachable(self, start):
        """
        Return all functions reachable from the given start function (including start).
        """
        visited = set()
        stack = [start]
        while stack:
            func = stack.pop()
            if func not in visited and func in self.call_graph:
                visited.add(func)
                stack.extend(self.call_graph[func])
        return visited
        
def set_performance_data_filename(fname):
    global performance_data_filename
    global performance_data
    performance_data_filename = fname
    try:
        with open(performance_data_filename, 'r') as infile:
            performance_data = json.load(infile)
    except FileNotFoundError:
        performance_data = {}
        pass

def track(length_computation, time_dilation=False, space_dilation=False, max_dilation=2.0):
    """
    A decorator to measure and store performance metrics of a function.

    Args:
        length_computation (callable): A function that calculates the "length"
                                       of one or more arguments.

    Returns:
        callable: The decorated function.
    """
    def decorator(func):
        called_functions = set()
        if time_dilation or space_dilation:
            # Grab the source code and identify any functions invoked by this function.
            source = inspect.getsource(inspect.getmodule(func))
            tree = ast.parse(source)
            finder = FunctionCallFinder()
            finder.visit(tree)
            called_functions = finder.get_root_functions(func.__name__)
            # print(f"Functions called by {func.__name__}: {called_functions}")
            if not called_functions:
                print(f"Warning: cannot augment samples for {func.__name__}")
            # Patch all the functions so we can individually delay them.
            for fn in called_functions:
                monkey_patch_function(func.__globals__,
                                      fn,
                                      time_dilation=time_dilation,
                                      space_dilation=space_dilation)

        # Store a hash of the code for checking if the function has changed
        # Currently not implemented.
        code = marshal.dumps(func.__code__)
        hash_value = hashlib.sha256(code).hexdigest()

        # Get the full name of the function (file + name)
        func_name = func.__name__
        file_name = inspect.getmodule(func).__file__
        full_name = str((func_name, file_name))
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Calculate the length based on the provided computation
            length = length_computation(*args, **kwargs)

            # Delay all the called functions.
            delay = 1.0
            if time_dilation or space_dilation:
                global delay_factor
                delay = random.uniform(1.0, max_dilation)
                # print(f"NEW DELAY {delay=}")
                for fn in finder.get_root_functions(func.__name__):
                    delay_factor[fn] = delay
            # Start measuring time and memory
            start_time = time.perf_counter()
            import gc
            gc.collect()
            import customalloc
            customalloc.reset_statistics();
            try:
                # print(f"RUNNING {func.__name__}")
                result = func(*args, **kwargs)
            finally:
                # Stop measuring time
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                
                peak = customalloc.get_peak_allocated()
                # print(f"{peak=}") # FIXME

                # Turn off the delay for all functions
                if time_dilation or space_dilation:
                    import customalloc
                    customalloc.set_dilation_factor(1.0)
                    for fn in finder.get_root_functions(func.__name__):
                        delay_factor[fn] = 0.0
                
                # Store the performance data. Only allow non-zero
                # lengths to avoid issues downstream when computing
                # logs of lengths.
                if length * delay:
                    new_entry = {
                        "hash" : hash_value,
                        "length": length * delay,
                        "time": elapsed_time,
                        "memory": peak,  # Peak memory usage in bytes
                    }
                    performance_data[full_name].append(new_entry)

            return result
        return wrapper
    return decorator


def fit_complexity_curves(input_file, output_file):
    """
    Reads a JSON file, fits time and space complexity curves for each function,
    and writes the results to another JSON file.

    :param input_file: Path to the input JSON file
    :param output_file: Path to the output JSON file
    """
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    results = {}

    for func_name, measurements in data.items():
        if len(measurements) <= 1:
            continue
        lengths = [entry['length'] for entry in measurements]
        times = [entry['time'] for entry in measurements]
        memory = [entry['memory'] for entry in measurements]

        # Log-transform data for linear regression (assumes log-log relationship)
        log_lengths = np.log(lengths)
        log_times = np.log(times)
        log_memory = np.log(memory)

        # Fit time complexity
        slope_time, intercept_time, r_value_time, p_value_time, _ = linregress(log_lengths, log_times)
        
        # Fit memory complexity
        slope_memory, intercept_memory, r_value_memory, p_value_memory, _ = linregress(log_lengths, log_memory)

        # print(func_name, slope_time, slope_memory)
        
        # Interpret slopes as complexity classes
        def interpret_complexity(slope):
            if np.isclose(slope, 0):
                return "O(1)"
            elif np.isclose(slope, 1):
                return "O(n)"
            elif np.isclose(slope, 2):
                return "O(n^2)"
            elif np.isclose(slope, np.log2(np.e)):
                return "O(log n)"
            elif np.isclose(slope, 1 + np.log2(np.e)):
                return "O(n log n)"
            else:
                return f"O(n^{slope:.2f})"

        time_complexity = interpret_complexity(slope_time)
        memory_complexity = interpret_complexity(slope_memory)

        # Store results
        results[func_name] = {
            "time_complexity": time_complexity,
            "time_complexity_r": r_value_time,
            "time_complexity_p": p_value_time,
            "memory_complexity": memory_complexity,
            "memory_complexity_r": r_value_memory,
            "memory_complexity_p": p_value_memory
        }

    # Write results to output file
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

# Example usage:
# fit_complexity_curves("input.json", "output.json")


@atexit.register
def save_performance_data():
    """
    Saves the collected performance data to a JSON file at program exit.
    """

    # Load any saved data into a dictionary.
    global performance_data_filename
    try:
        with open(performance_data_filename, 'r') as infile:
            old_data = json.load(infile)
    except FileNotFoundError:
        old_data = {}
        pass

    # Merge the old with the new dictionary
    for key, value_list in old_data.items():
        if key in performance_data:
            # Key exists in both dicts; extend the list from performance_data with the new entries
            performance_data[key].extend(value_list)
        else:
            # Key only exists in old_data; add it to performance_data
            performance_data[key] = value_list
    
    with open(performance_data_filename, "w") as f:
        json.dump(performance_data, f, indent=4)
    fit_complexity_curves(performance_data_filename, performance_analysis_filename)

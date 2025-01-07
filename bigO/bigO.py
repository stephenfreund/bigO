import atexit
import hashlib
import inspect
import json
import textwrap
import time
import marshal
import sys

from collections import defaultdict
from functools import wraps
from typing import Any, Callable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from bigO import models

import seaborn as sns

system_name = "bigO"

# Global dictionary to store performance data
performance_data: dict[str, list[Any]] = defaultdict(list)

# Where the performance data is stored.
performance_data_filename = f"{system_name}_data.json"

# Hashes of function implementations, used to discard outdated perf info for modified functions
hash_function_values: dict[str, Any] = {}

python_version = (sys.version_info[0], sys.version_info[1])
# Disabled for now
use_sys_monitoring = False
# use_sys_monitoring = python_version >= (3,12)
TOOL_ID = 1
if use_sys_monitoring:
    sys.monitoring.use_tool_id(TOOL_ID, system_name)


def set_performance_data_filename(fname: str) -> str:
    """Changes the file name where performance data is stored
    and loads the performance data.

    Returns the previous file name.
    """
    global performance_data_filename
    global performance_data
    old_performance_data_filename = performance_data_filename
    performance_data_filename = fname
    try:
        with open(performance_data_filename, "r") as infile:
            performance_data = json.load(infile)
    except FileNotFoundError:
        performance_data = dict()
    return old_performance_data_filename


def function_hash_value(func: Callable) -> str:
    """
    Returns the hash value of the function implementation.
    """
    code = marshal.dumps(func.__code__)
    return hashlib.sha256(code).hexdigest()


def function_full_name(func: Callable) -> str:
    """
    Returns the full name of the function (file + name).
    """
    func_name = func.__name__
    module = inspect.getmodule(func)
    file_name = module.__file__ if module and module.__file__ else "<unknown>"
    return str((func_name, file_name))


def track(length_function: Callable[..., int]) -> Callable:
    """
    A decorator to measure and store performance metrics of a function.

    Args:
        length_function (callable): A function that calculates the "length"
                                       of one or more arguments.

    Returns:
        callable: The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        # Store a hash of the code to enable discarding old perf data if the
        # function has changed
        hash_value = function_hash_value(func)

        # Get the full name of the function (file + name), and save the hash value.
        full_name = function_full_name(func)
        hash_function_values[full_name] = hash_value

        @wraps(func)
        def wrapper(*args, **kwargs):
            import customalloc

            # Calculate the length based on the provided computation
            length = length_function(*args, **kwargs)

            # Start measuring time and memory
            start_time = time.perf_counter()
            customalloc.reset_statistics()
            customalloc.enable()

            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                peak = customalloc.get_peak_allocated()
                nobjects = customalloc.get_objects_allocated()
                customalloc.disable()

                # Store the performance data. Only allow non-zero
                # lengths to avoid issues downstream when computing
                # logs of lengths.
                if length > 0:
                    perf_data = {
                        "hash": hash_value,
                        "length": length,
                        "time": elapsed_time,
                        "memory": peak,  # Peak memory usage in bytes
                        "nobjects": nobjects,
                    }
                    performance_data[full_name].append(perf_data)
            return result

        return wrapper

    return decorator


def plot_complexities(n_time, y_time, spec_fit, time_fits):
    sns.set_style("whitegrid")

    # Time plot (axes[i,0])
    fig, ax_time = plt.subplots(figsize=(6, 4))
    ax_time.plot(n_time, y_time, "o", color="blue", label="Data (outliers removed)")
    fit_n = np.sort(n_time)
    ax_time.plot(
        fit_n,
        spec_fit.predict(fit_n),
        "-",
        color="orange",
        linewidth=1,
        label=f"Spec: {spec_fit}",
    )

    for fitted in time_fits:
        ax_time.plot(
            fit_n,
            fitted.predict(fit_n),
            "-",
            color="blue",
            linewidth=1,
            label=f"Fit: {fitted}",
        )

    ax_time.set_xlabel("Input Size (n)")
    ax_time.set_ylabel("Time")
    ax_time.legend()

    plt.tight_layout()
    filename = f"{system_name}.pdf"
    plt.savefig(filename)
    print(f"{filename} written.")
    # plt.show()


def check(
    length_function: Callable[..., int],
    time_bound: str | None = None,
    # mem_bound: str | None = None,
    frequency: int = 25,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        full_name = function_full_name(func)
        tracked = track(length_function)(func)
        time_model = models.get_model(time_bound) if time_bound else None
        # mem_model = models.get_model(mem_bound) if mem_bound else None

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = tracked(*args, **kwargs)
            finally:
                pass

            if len(performance_data[full_name]) % frequency == 0:
                lengths = [entry["length"] for entry in performance_data[full_name]]
                if time_model:
                    times = [entry["time"] for entry in performance_data[full_name]]
                    spec_model_fit, slower_but_better_fits = models.check_bound(
                        lengths, times, time_model
                    )
                    if slower_but_better_fits:
                        as_str = "\n".join(
                            [
                                f"{str(model)} (pvalue={pvalue:.3f})"
                                for model, pvalue in slower_but_better_fits
                            ]
                        )

                        plot_complexities(
                            lengths,
                            times,
                            spec_model_fit,
                            [fitted for fitted, _ in slower_but_better_fits],
                        )

                        raise ValueError(
                            textwrap.dedent(
                                """\
                                {full_name} is not {time_bound} as expected.  
                                The following are better fits:
                                ```
                                {as_str}
                                ```
                                See {system_name}.pdf for a plot of the data and fits."""
                            ).format(
                                full_name=full_name,
                                time_bound=time_bound,
                                as_str=as_str,
                                system_name=system_name,
                            )
                        )

                # if mem_model:
                #     memories = [
                #         entry["memory"] for entry in performance_data[full_name]
                #     ]
                #     models.check_bound(lengths, memories, mem_model)

            return result

        return wrapper

    return decorator


@atexit.register
def save_performance_data() -> None:
    """
    Saves the collected performance data to a JSON file at program exit.
    """

    # Load any saved data into a dictionary.
    global performance_data_filename
    try:
        with open(performance_data_filename, "r") as infile:
            old_data = json.load(infile)
        # Discard any outdated entries.
        for full_name in old_data:
            if full_name in hash_function_values:
                validated = []
                current_hash = hash_function_values[full_name]
                for entry in old_data[full_name]:
                    if entry["hash"] == current_hash:
                        validated.append(entry)
                old_data[full_name] = validated

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

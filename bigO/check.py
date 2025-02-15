import time
import tracemalloc
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np

from functools import wraps

from bigO.models import (
    Model,
    fit_model,
    fit_models,
    get_model,
    remove_outliers,
)


def error_plot(func_name, bound_type, n, y, bound_model_fit, best_fit, pvalue):
    _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n, y, "o", color="blue", label="Data (outliers removed)")
    fit_n = np.sort(n)

    ax.plot(
        fit_n,
        bound_model_fit.predict(fit_n),
        "-",
        color="red",
        linewidth=1,
        label=f"Specified: {bound_model_fit}",
    )

    ax.plot(
        fit_n,
        best_fit.predict(fit_n),
        "-",
        color="blue",
        linewidth=1,
        label=f"Actual: {best_fit}",
    )
    title = f"{func_name}: {best_fit}\nmse={best_fit.mse(n,y):.3f} pvalue={pvalue:.3f}"

    ax.set_xlabel("Input Size (n)")
    ax.set_ylabel(bound_type)
    ax.set_title(title, fontsize=12)
    ax.legend()
    plt.show()


def check(
    length_computation: Callable,
    time_bound: str | None = None,
    mem_bound: str | None = None,
    frequency: int = 25,
):
    def decorator(func):

        func_name = func.__name__
        time_model = get_model(time_bound) if time_bound else None
        mem_model = get_model(mem_bound) if mem_bound else None
        performance_data = []

        print(
            f"Asserting complexity of {func_name} with time bound {time_bound} and memory bound {mem_bound}"
        )

        def check_bound(bound_type: str, bound_model: Model):
            lengths, y = remove_outliers(
                [entry["length"] for entry in performance_data],
                [entry[bound_type] for entry in performance_data],
            )
            fits = fit_models(lengths, y)
            bound_model_fit, _ = fit_model(lengths, y, bound_model)
            if fits:
                best_fit, _ = fits[0]
                if not (best_fit <= bound_model_fit):
                    pvalue = better_fit_pvalue_vectorized(
                        lengths, y, bound_model_fit, best_fit
                    )
                    if pvalue < 0.05:
                        print(
                            f"Error: {bound_type} complexity of {func_name} is {best_fit} instead of {bound_model_fit}, with pvalue={pvalue}."
                        )
                        error_plot(
                            func_name,
                            bound_type,
                            lengths,
                            y,
                            bound_model_fit,
                            best_fit,
                            pvalue,
                        )
                    else:
                        print(
                            f"Warning: {bound_type} complexity of {func_name} may be {best_fit} instead of {bound_model_fit}, but pvalue={pvalue} is not significant."
                        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            length = length_computation(*args, **kwargs)
            start_time = time.perf_counter()
            tracemalloc.start()
            try:
                result = func(*args, **kwargs)
            finally:
                # Stop measuring time
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Stop measuring memory
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                new_entry = {
                    "length": length,
                    "time": elapsed_time,
                    "memory": peak,  # Peak memory usage in bytes
                }
                performance_data.append(new_entry)

                if len(performance_data) > 0 and len(performance_data) % frequency == 0:
                    if time_model:
                        check_bound("time", time_model)
                    if mem_model:
                        check_bound("memory", mem_model)

            return result

        return wrapper

    return decorator

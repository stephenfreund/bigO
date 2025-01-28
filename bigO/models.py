from dataclasses import dataclass
import json
from math import e
import sys
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
from scipy.optimize import curve_fit

from bigO.output import log_timer

system_name = "bigO"


class Model:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
        self.param_count = func.__code__.co_argcount - 1  # Subtract 1 for 'n'

    def __call__(self, n, *args):
        return self.func(n, *args)

    def __str__(self):
        return self.name


class FittedModel:
    def __init__(
        self,
        model: Model,
        params: np.ndarray,
        n: np.ndarray,
        y: np.ndarray,
    ):
        self.model = model
        self.params = params
        self.n = n
        self.y = y

    def predict(self, n: np.ndarray | None = None):
        if n is None:
            n = self.n
        return self.model.func(n, *self.params)

    def residuals(self, n: np.ndarray | None = None, y: np.ndarray | None = None):
        if n is None:
            n = self.n
        if y is None:
            y = self.y
        return y - self.predict(n)

    def mse(self):
        return np.mean(self.residuals() ** 2)

    def aic(self, n: np.ndarray | None = None, y: np.ndarray | None = None):
        k = len(self.params)  # Number of parameters
        residuals = self.residuals(n, y)
        rss = np.sum(residuals**2)  # Residual sum of squares
        if y is None:
            y = self.y
        n_points = len(y)  # Number of data points

        if n_points < 2 or rss <= 0:
            return np.inf

        aic = 2 * k + n_points * np.log(rss / n_points)
        # print(k, rss, n_points, aic)
        return aic

    def replace_k(self, name):
        return name.replace("k", f"{self.params[-1]:.2f}")

    def __str__(self):
        name = self.model.name
        return f"O({self.replace_k(name)})"

    def __repr__(self):
        return str(self)

    def actual(self):
        return f"{self.params[0]} * {str(self)} + {self.params[1]}"


def safe_log(x):
    x = np.asarray(x)
    result = np.full_like(x, -np.inf, dtype=np.float64)  # Initialize with -inf
    np.log(x, where=(x > 0), out=result)
    return result


# Model definitions
model_constant = Model("1", lambda n, a: np.ones(np.shape(n)) * a)
model_log_log_n = Model("log(log(n))", lambda n, a, b: a * safe_log(safe_log(n)) + b)
model_log_n = Model("log(n)", lambda n, a, b: a * safe_log(n) + b)
model_sqrt_n = Model("sqrt(n)", lambda n, a, b: a * np.sqrt(n) + b)
model_linear_n = Model("n", lambda n, a, b: a * n + b)
model_n_log_n = Model("n*log(n)", lambda n, a, b: a * n * safe_log(n) + b)
model_n_squared = Model("n**2", lambda n, a, b: a * n**2 + b)
model_n_cubed = Model("n**3", lambda n, a, b: a * n**3 + b)
model_n_power_k = Model("n**k", lambda n, a, b, k: a * n**k + b)
model_log_n_squared = Model("(log(n))**2", lambda n, a, b: a * (safe_log(n) ** 2) + b)
model_log_n_cubed = Model("(log(n))**3", lambda n, a, b: a * (safe_log(n) ** 3) + b)
# model_n_log_n_power_k = Model(
#     "n*(log(n))**k", lambda n, a, b, k: a * n * (np.log(n) ** k) + b
# )

# List of models
models = [
    model_constant,
    model_log_log_n,
    model_log_n,
    model_log_n_squared,
    model_log_n_cubed,
    model_sqrt_n,
    model_linear_n,
    model_n_log_n,
    model_n_squared,
    model_n_cubed,
    model_n_power_k,
]


def get_model(name: str) -> Model:
    if not name.startswith("O(") or not name.endswith(")"):
        print(f"Invalid model name: {name}.  Should be of the form O(...)")
        return None

    # Strip "O(" and ")"
    name = name[2:-1]

    for model in models:
        if model.name == name:
            return model

    # if name matches pattern n**k, return a new model with k harcoded in
    if name.startswith("n**"):
        k = float(name[3:])
        return Model(name, lambda n, a, b: a * n**k + b)

    raise ValueError(f"Unknown model: {name}")


# TODO: Check the ranks for n^k.
def rank(fitted_model):
    if fitted_model.model != model_n_power_k:
        return models.index(fitted_model.model)
    else:
        # Handle the special case for model_n_power_k
        k = fitted_model.params[-1]
        if k == 0:  # Constant growth
            return 0  # Comparate to constant
        elif 0 < k < 1:  # Sub-linear growth
            return models.index(model_sqrt_n)  # Comparable to sqrt(n)
        elif k == 1:  # Linear growth
            return models.index(model_linear_n)
        elif 1 < k < 2:  # Super-linear but sub-quadratic
            return models.index(model_n_log_n)
        elif k == 2:  # Quadratic growth
            return models.index(model_n_squared)
        elif 2 < k < 3:  # Super-quadratic but sub-cubic
            return models.index(model_n_cubed) - 1  # Between quadratic and cubic
        elif k == 3:  # Cubic growth
            return models.index(model_n_cubed)
        else:  # k > 3, ranks higher than cubic
            return len(models)  # Ranks at the end


def leq(fitted_model_1, fitted_model_2):
    return rank(fitted_model_1) <= rank(fitted_model_2)


def fit_model(n, y, model) -> FittedModel:
    params, _ = curve_fit(model.func, n, y, p0=[1.0] * model.param_count, maxfev=10000)
    fitted = FittedModel(model=model, params=params, n=n, y=y)
    return fitted


def fit_models(n, y) -> List[FittedModel]:
    """Fit models to data and order by increasing aic. Return list of (model, aic) tuples."""
    fits = [fit_model(n, y, model) for model in models]
    return sorted(fits, key=lambda x: x.aic())


@dataclass
class CheckBoundResult:
    declared_bound_fit: FittedModel
    better_models: pd.DataFrame


def check_bound(n: np.ndarray, y: np.ndarray, bound: Model) -> CheckBoundResult:
    lengths, y = remove_outliers(n, y)
    fits = fit_models(lengths, y)
    bound_model_fit = fit_model(lengths, y, bound)

    fitted_models = pd.DataFrame(
        [
            (
                fitted,
                fitted.aic(lengths, y),
                pvalue_for_better_fit(fitted, bound_model_fit, lengths, y),
            )
            for fitted in fits
        ],
        columns=["model", "aic", "pvalue"],
    )

    fitted_models = fitted_models.sort_values(by="aic", ascending=True)
    fitted_models["rank"] = fitted_models["model"].apply(rank)
    fitted_models = fitted_models[fitted_models["rank"] > rank(bound_model_fit)]
    better_models = fitted_models[fitted_models["pvalue"] < 0.05]

    return CheckBoundResult(
        bound_model_fit,
        better_models,
    )


def infer_bound(n: np.ndarray, y: np.ndarray) -> List[FittedModel]:
    lengths, y = remove_outliers(n, y)
    fits = fit_models(lengths, y)
    return fits


def pvalue_for_better_fit(
    a: FittedModel, b: FittedModel, n, y, trials=1000  # low trials to start with
) -> float:
    """
    Return pvalue that model 'a' is a better fit than model 'b'.
    Technique from: https://aclanthology.org/D12-1091.pdf
    """
    delta = b.aic() - a.aic()

    bootstrap_indices = np.random.choice(len(n), size=(trials, len(n)), replace=True)

    data = np.column_stack((n, y))
    # a 3D array with shape (trials, n_samples, 2)
    resamples = data[bootstrap_indices]

    # compute the delta for each resample
    s = 0
    for resample in resamples:
        n2 = resample[:, 0]
        y2 = resample[:, 1]
        aic_a2 = a.aic(n2, y2)
        aic_b2 = b.aic(n2, y2)
        bootstrap_delta = aic_b2 - aic_a2
        # See paper for why 2
        if bootstrap_delta > 2 * delta:
            s += 1
    return s / trials


def remove_outliers(n, y):
    """
    Remove outliers using IQR method.
    """
    y = np.array(y, dtype=float)
    n = np.array(n, dtype=float)
    if len(y) < 4:
        return n, y
    Q1, Q3 = np.percentile(y, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (y >= lower_bound) & (y <= upper_bound)
    return n[mask], y[mask]


def plot_complexities_from_file(filename=f"{system_name}_data.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    entries = []
    for key, records in data.items():
        key_str = key.strip("()")
        parts = key_str.split(",")
        function_name = parts[0].strip().strip("'\"")
        file_name = parts[1].strip().strip("'\"")

        lengths = [r["length"] for r in records]
        times = [r["time"] for r in records]
        mems = [r["memory"] for r in records]

        entries.append((function_name, file_name, lengths, times, mems))

    n_entries = len(entries)

    sns.set_style("whitegrid")
    # Use squeeze=False to always get a 2D array of axes

    plot_memory = True

    if plot_memory:
        fig, axes = plt.subplots(
            nrows=n_entries, ncols=2, figsize=(12, 4 * n_entries), squeeze=False
        )
    else:
        fig, axes = plt.subplots(
            nrows=n_entries, ncols=1, figsize=(6, 4 * n_entries), squeeze=False
        )

    for i, (func, filepath, lengths, times, mems) in enumerate(entries):
        # Remove outliers
        n_time, y_time = remove_outliers(lengths, times)
        time_fits = fit_models(n_time, y_time)
        time_fits.sort(key=lambda x: x.aic(n_time, y_time))

        n_mem, y_mem = remove_outliers(lengths, mems)
        mem_fits = fit_models(n_mem, y_mem)

        # Time plot (axes[i,0])
        ax_time = axes[i, 0]
        ax_time.plot(n_time, y_time, "o", color="blue", label="Data (outliers removed)")
        if len(time_fits) > 0:
            best_fit = time_fits[0]
            fit_n = np.sort(n_time)
            fit_y = best_fit.predict(fit_n)

            ax_time.plot(
                fit_n,
                best_fit.predict(fit_n),
                "-",
                color="blue",
                linewidth=1,
                label=f"Fit: {best_fit}",
            )

            for j in np.arange(1, 3):
                ax_time.plot(
                    fit_n,
                    time_fits[j].predict(fit_n),
                    "--",
                    color="blue",
                    linewidth=0.5,
                    label=f"{j+1}nd Fit: {time_fits[j]}",
                )

            time_pvalue = 0
            best_aic = best_fit.aic(n_time, y_time)
            title = f"{func} (Time): {best_fit}\naic={best_aic:.3f} mse={best_fit.mse():.3f} pvalue={time_pvalue:.3f}"
        else:
            title = f"{func} (Time): No fit"

        ax_time.set_xlabel("Input Size (n)")
        ax_time.set_ylabel("Time")
        ax_time.set_title(title, fontsize=12)
        ax_time.legend()

        if False and plot_memory:
            # Memory plot (axes[i,1])
            ax_mem = axes[i, 1]
            ax_mem.plot(n_mem, y_mem, "o", color="red", label="Data (outliers removed)")
            if len(mem_fits) > 0:
                best_fit, best_aic = mem_fits[0]
                fit_n = np.sort(n_mem)
                fit_y = best_fit.predict(fit_n)

                ax_mem.plot(
                    fit_n,
                    fit_y,
                    "-",
                    color="red",
                    linewidth=1,
                    label=f"Fit: {best_fit}",
                )

                for j in np.arange(1, 3):
                    ax_mem.plot(
                        fit_n,
                        mem_fits[j][0].predict(fit_n),
                        "--",
                        color="red",
                        linewidth=0.5,
                        label=f"{j+1}nd Fit: {mem_fits[j][0]}",
                    )

                title = f"{func} (Mem): {best_fit}\naic={best_aic:.3f} mse={best_fit.mse(n_mem,y_mem):.3f} pvalue={mem_pvalue:.3f}"
            else:
                title = f"{func} (Memory): No fit"
            ax_mem.set_xlabel("Input Size (n)")
            ax_mem.set_ylabel("Memory")
            ax_mem.set_title(title, fontsize=12)
            ax_mem.legend()

    plt.tight_layout()
    filename = f"{system_name}.pdf"
    plt.savefig(filename)
    print(f"{filename} written.")
    # plt.show()


if __name__ == "__main__":
    fname = f"{system_name}_data.json"
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    plot_complexities_from_file(fname)

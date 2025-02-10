from dataclasses import dataclass
import json
import sys
from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit


system_name = "bigO"


@dataclass
class CanonicalForm:
    """
    Represent any function f(n) = r^n * n^s * (log n)^t * (log log n)^u
    as the coefficients.  All coefficients are floats and >= 0.  Use 
    None to indicate that the coefficient is the *last* parameter of 
    the fitted model for this funcion.
    """
    
    r: float
    s: float
    t: float
    u: float

    def __str__(self) -> str:
        if self.r == 0:
            return "0"
        parts = []
        if self.r != 1:
            parts.append(f"{self.r:.2f}**n")
        if self.s != 0:
            parts.append(f"n**{self.s:.2f}")
        if self.t != 0:
            parts.append(f"log(n)**{self.t:.2f}")
        if self.u != 0:
            parts.append(f"log(log(n))**{self.u:.2f}")
        return " * ".join(parts)
    

class Model:
    def __init__(self, name: str, func: Callable, canonical_form: CanonicalForm):
        self.name = name
        self.func = func
        self.param_count = func.__code__.co_argcount - 1  # Subtract 1 for 'n'
        self.canonical_form = canonical_form

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
    
    def __le__(self, other):

        def val_or_last_param(x):
            return x if x is not None else self.params[-1]

        cf = self.model.canonical_form
        cfo = other.model.canonical_form
        
        # exponential growth

        r = val_or_last_param(cf.r)
        ro = val_or_last_param(cfo.r)
        if r < ro:
            return True
        elif r > ro:
            return False
                
        # polynomial growth
        s = val_or_last_param(cf.s)
        so = val_or_last_param(cfo.s)
        if s < so:
            return True
        elif s > so:
            return False
        
        # log growth
        t = val_or_last_param(cf.t)
        to = val_or_last_param(cfo.t)
        if t < to:
            return True
        elif t > to:
            return False
        
        # log log growth
        u = val_or_last_param(cf.u)
        uo = val_or_last_param(cfo.u)
        if u < uo:
            return True
        elif u > uo:
            return False
        
        return True



def safe_log(x):
    x = np.asarray(x)
    result = np.full_like(x, -np.inf, dtype=np.float64)  # Initialize with -inf
    np.log(x, where=(x > 0), out=result)
    return result


# Model definitions
model_constant = Model("1", lambda n, a: np.ones(np.shape(n)) * a, CanonicalForm(1, 0, 0, 0))
model_log_log_n = Model("log(log(n))", lambda n, a, b: a * safe_log(safe_log(n)) + b, CanonicalForm(1, 0, 0, 1))
model_log_n = Model("log(n)", lambda n, a, b: a * safe_log(n) + b, CanonicalForm(1, 0, 1, 0))
model_sqrt_n = Model("sqrt(n)", lambda n, a, b: a * np.sqrt(n) + b, CanonicalForm(1, 0.5, 0, 0))
model_linear_n = Model("n", lambda n, a, b: a * n + b, CanonicalForm(1, 1, 0, 0))
model_n_log_n = Model("n*log(n)", lambda n, a, b: a * n * safe_log(n) + b, CanonicalForm(1, 1, 1, 0))
model_n_squared = Model("n**2", lambda n, a, b: a * n**2 + b, CanonicalForm(1, 2, 0, 0))
model_n_cubed = Model("n**3", lambda n, a, b: a * n**3 + b, CanonicalForm(1, 3, 0, 0))
model_n_power_k = Model("n**k", lambda n, a, b, k: a * n**k + b, CanonicalForm(1, None, 0, 0))
model_log_n_squared = Model("(log(n))**2", lambda n, a, b: a * (safe_log(n) ** 2) + b, CanonicalForm(1, 0, 2, 0))
model_log_n_cubed = Model("(log(n))**3", lambda n, a, b: a * (safe_log(n) ** 3) + b, CanonicalForm(1, 0, 3, 0))
model_n_exp = Model("2**n", lambda n, a, b: a * (2**n) + b, CanonicalForm(2, 0, 0, 0))

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
    model_n_exp,
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
    # fitted_models["rank"] = fitted_models["model"].apply(rank)
    print(fitted_models["model"] <= bound_model_fit)
    fitted_models = fitted_models[~(fitted_models["model"] <= bound_model_fit)]
    print(fitted_models)
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



if __name__ == "__main__":

    import unittest



    class TestFittedModelLE(unittest.TestCase):

        def setUp(self):
            # Dummy data for n and y.
            self.n_data = np.array([2, 3, 4])
            self.y_data = np.array([1, 2, 3])

        def create_fitted_model(self, model, params):
            return FittedModel(model, np.array(params), self.n_data, self.y_data)

        def test_constant_vs_log_log(self):
            # constant: (1, 0, 0, 0) vs log(log(n)): (1, 0, 0, 1)
            fm_const = self.create_fitted_model(model_constant, [5.0])
            fm_log_log = self.create_fitted_model(model_log_log_n, [5.0, 0.0])
            self.assertTrue(fm_const <= fm_log_log)
            self.assertFalse(fm_log_log <= fm_const)

        def test_log_vs_n_log(self):
            # log(n): (1, 0, 1, 0) vs n*log(n): (1, 1, 1, 0)
            fm_log = self.create_fitted_model(model_log_n, [5.0, 0.0])
            fm_n_log = self.create_fitted_model(model_n_log_n, [5.0, 0.0])
            self.assertTrue(fm_log <= fm_n_log)
            self.assertFalse(fm_n_log <= fm_log)

        def test_sqrt_vs_linear(self):
            # sqrt(n): (1, 0.5, 0, 0) vs n: (1, 1, 0, 0)
            fm_sqrt = self.create_fitted_model(model_sqrt_n, [5.0, 0.0])
            fm_linear = self.create_fitted_model(model_linear_n, [5.0, 0.0])
            self.assertTrue(fm_sqrt <= fm_linear)
            self.assertFalse(fm_linear <= fm_sqrt)

        def test_n_squared_vs_n_cubed(self):
            # n**2: (1, 2, 0, 0) vs n**3: (1, 3, 0, 0)
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            fm_n_cubed = self.create_fitted_model(model_n_cubed, [5.0, 0.0])
            self.assertTrue(fm_n_squared <= fm_n_cubed)
            self.assertFalse(fm_n_cubed <= fm_n_squared)

        def test_n_power_k_vs_n_squared_with_k_less(self):
            # n**k: (1, None, 0, 0) where None is substituted with k.
            # Use k = 1.5 which is less than 2 (from n**2)
            fm_n_power_k = self.create_fitted_model(model_n_power_k, [5.0, 0.0, 1.5])
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            # For fm_n_power_k, s becomes 1.5 vs 2.
            self.assertTrue(fm_n_power_k <= fm_n_squared)
            # In reverse, fm_n_squared <= fm_n_power_k compares 2 (its s) with
            # the substituted value from fm_n_power_k using fm_n_squared’s params[-1] (0.0), so False.
            self.assertFalse(fm_n_squared <= fm_n_power_k)

        def test_n_power_k_vs_n_squared_with_k_more(self):
            # Use k = 2.5 which is greater than 2.
            fm_n_power_k = self.create_fitted_model(model_n_power_k, [5.0, 0.0, 2.5])
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            self.assertFalse(fm_n_power_k <= fm_n_squared)

        def test_n_exp_vs_linear_less(self):
            # n_exp: (None, 1, 0, 0) where r is substituted with last param.
            # Use k = 0.8 (< 1 from linear)
            fm_n_exp = self.create_fitted_model(model_n_exp, [5.0, 0.0, 0.8])
            fm_linear = self.create_fitted_model(model_linear_n, [5.0, 0.0])
            # For fm_n_exp, r becomes 0.8 which is less than 1.
            self.assertTrue(fm_n_exp <= fm_linear)
            self.assertFalse(fm_linear <= fm_n_exp)

        def test_n_exp_vs_linear_more(self):
            # Use k = 1.2 (> 1)
            fm_n_exp = self.create_fitted_model(model_n_exp, [5.0, 0.0, 1.2])
            fm_linear = self.create_fitted_model(model_linear_n, [5.0, 0.0])
            self.assertFalse(fm_n_exp <= fm_linear)

        def test_log_n_squared_vs_log_n_cubed(self):
            # (log(n))**2: (1,0,2,0) vs (log(n))**3: (1,0,3,0)
            fm_log_n_squared = self.create_fitted_model(model_log_n_squared, [5.0, 0.0])
            fm_log_n_cubed = self.create_fitted_model(model_log_n_cubed, [5.0, 0.0])
            self.assertTrue(fm_log_n_squared <= fm_log_n_cubed)
            self.assertFalse(fm_log_n_cubed <= fm_log_n_squared)

        def test_n_log_n_vs_n_squared(self):
            # n*log(n): (1,1,1,0) vs n**2: (1,2,0,0)
            fm_n_log_n = self.create_fitted_model(model_n_log_n, [5.0, 0.0])
            fm_n_squared = self.create_fitted_model(model_n_squared, [5.0, 0.0])
            self.assertTrue(fm_n_log_n <= fm_n_squared)
            self.assertFalse(fm_n_squared <= fm_n_log_n)

    unittest.main()

import abc
from dataclasses import dataclass
import json
import textwrap
from typing import List, Optional

from matplotlib import gridspec, pyplot as plt
from matplotlib.figure import SubFigure
import pandas as pd
import bigO.models as models
from bigO.abtest import non_parametric_fit, segmented_permutation_test
from bigO.output import log, log_timer, message
import click
import numpy as np
import seaborn as sns


system_name = "bigO"


@dataclass
class FunctionData:
    function_name: str
    file_name: str
    lengths: np.ndarray
    times: np.ndarray
    mems: np.ndarray


class Analysis(abc.ABC):

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def message(self) -> str:
        pass

    @abc.abstractmethod
    def plot(self, fig: SubFigure):
        pass


class InferPerformance(Analysis):

    def __init__(self, function_data: FunctionData):
        self.function_data = function_data

    def run(self):
        with log_timer(
            f"Infer time and memory models for {self.function_data.function_name}"
        ):
            self.fitted_times = models.infer_bound(
                self.function_data.lengths, self.function_data.times
            )
            self.fitted_mems = models.infer_bound(
                self.function_data.lengths, self.function_data.mems
            )

    def message(self) -> str:
        best_time_model = self.fitted_times[0]
        best_mem_model = self.fitted_mems[0]
        return (
            f"Inferred Bounds for {self.function_data.function_name}\n"
            f"Best time model: {best_time_model}\nBest memory model: {best_mem_model}"
        )

    def plot_fits(self, ax, models: List[models.FittedModel], color, ylabel, title):
        best_fit = models[0]
        sns.scatterplot(
            x=best_fit.n,
            y=best_fit.y,
            ax=ax,
            color=color,
            label="Data (outliers removed)",
        )
        styles = ["-", "--", "-.", ":"]
        for i, model in enumerate(models[0:4]):
            sns.lineplot(
                x=model.n,
                y=model.predict(model.n),
                ax=ax,
                label=f"{model}",
                color=color,
                linewidth=2 if i == 0 else 1,
                linestyle=styles[i],
                alpha=(1 - i * 0.2),
            )

        ax.set_xlabel("Input Size (n)")
        ax.set_ylabel(ylabel)

        ax.set_title(title, fontsize=12)
        ax.legend()

    def plot(self, fig: SubFigure):
        time_ax, mem_ax, extra = fig.subplots(1, 3)
        extra.axis("off")
        self.plot_fits(
            time_ax,
            self.fitted_times,
            color="C0",
            ylabel="Time (s)",
            title=f"{self.function_data.function_name}: {self.fitted_times[0]}",
        )
        self.plot_fits(
            mem_ax,
            self.fitted_mems,
            color="C1",
            ylabel="Memory",
            title=f"{self.function_data.function_name}: {self.fitted_mems[0]}",
        )


class CheckBounds(Analysis):
    def __init__(
        self,
        function_data: FunctionData,
        time_bound: str | None = None,
        mem_bound: str | None = None,
    ):
        super().__init__()
        self.function_data = function_data
        self.time_bound = models.get_model(time_bound) if time_bound else None
        self.mem_bound = models.get_model(mem_bound) if mem_bound else None

    def run(self):
        with log_timer(
            "Infer time and memory models for {self.function_data.function_name}"
        ):
            with log_timer("Check time models"):
                if self.time_bound:
                    self.time_check = models.check_bound(
                        self.function_data.lengths,
                        self.function_data.times,
                        self.time_bound,
                    )
                else:
                    self.time_check = None
            with log_timer("Infer memory models"):
                if self.mem_bound:
                    self.mem_check = models.check_bound(
                        self.function_data.lengths,
                        self.function_data.mems,
                        self.mem_bound,
                    )
                else:
                    self.mem_check = None

    def message(self) -> str:
        message = [ f"Bounds for {self.function_data.function_name}" ]
        if self.time_check:
            message += [
                f"Declared Time Bound: {self.time_bound}",
                f"Models with better fits: ",
                str(
                    self.time_check.better_models[["model", "pvalue"]].to_string(
                        index=False
                    )
                ),
            ]
        if self.mem_check:
            message += [
                f"Declared Memory Bound: {self.mem_bound}",
                f"Models with better fits: ",
                str(self.mem_check.better_models),
            ]
        return "\n".join(message)

    def plot_fits(
        self, ax, check_result: models.CheckBoundResult, color, ylabel, title
    ):
        declared_fit = check_result.declared_bound_fit
        sns.scatterplot(
            x=declared_fit.n,
            y=declared_fit.y,
            ax=ax,
            color=color,
            label="Data (outliers removed)",
        )
        sns.lineplot(
            x=declared_fit.n,
            y=declared_fit.predict(declared_fit.n),
            ax=ax,
            color=color,
            label=f"{declared_fit}: {declared_fit}",
        )
        for i, (_, row) in enumerate(check_result.better_models.take(np.arange(0,4)).iterrows()):
            (model, pvalue) = row["model"], row["pvalue"]
            sns.lineplot(
                x=model.n,
                y=model.predict(model.n),
                ax=ax,
                label=f"{model} (p={pvalue:.3f})",
                color="red",
                linewidth=2 if i == 0 else 1,
                alpha=(1 - i * 0.2),
            )

        ax.set_xlabel("Input Size (n)")
        ax.set_ylabel(ylabel)

        ax.set_title(title, fontsize=12)
        ax.legend()

    def plot(self, fig: SubFigure):
        time_ax, mem_ax, extra = fig.subplots(1, 3)
        extra.axis("off")
        if self.time_check:
            self.plot_fits(
                time_ax,
                self.time_check,
                color="C0",
                ylabel="Time (s)",
                title=f"{self.function_data.function_name}: Time Bound",
            )
        if self.mem_check:
            self.plot_fits(
                mem_ax,
                self.mem_check,
                color="C1",
                ylabel="Memory",
                title=f"{self.function_data.function_name}: Mem Bound",
            )


class ABTest(Analysis):
    def __init__(
        self,
        a: FunctionData,
        b: FunctionData,
        metric: str
    ):
        self.a = a
        self.b = b
        self.metric = metric
        print(f"AB Test: A is {a.function_name}; B is {b.function_name}")

    def run(self):
        with log_timer(
            "Infer time and memory models for {self.function_data.function_name}"
        ):
            combined_labels = np.concatenate([["A"] * len(self.a.lengths), ["B"] * len(self.b.lengths)])
            combined_lengths = np.concatenate([self.a.lengths, self.b.lengths])
            combined_T = np.concatenate([self.a.times if self.metric == 'time' else self.a.mems, 
                                         self.b.times if self.metric == 'time' else self.b.mems])
            self.combined_df = pd.DataFrame({
                "label": combined_labels,
                "n": combined_lengths,
                "T": combined_T,
            })
            self.ab_results = segmented_permutation_test(
                self.combined_df,
                num_permutations=1000,
                num_points=100,
            )

    def message(self) -> str:
        message = [ f"AB Test for {self.metric}: A is {self.a.function_name}; B is {self.b.function_name}" ]
        for report in self.ab_results:
            message += [ str(report) ]
        return "\n".join(message)


    def plot(self, fig: SubFigure):
        num_figs = min(1 + len(self.ab_results), 4)
        axes = fig.subplots(1, num_figs)

        # ----------------------------------------
        # 4.1. Scatter Plots with LOESS Curves (First Visualization)
        # ----------------------------------------

        df_A = self.combined_df[self.combined_df.label == "A"]
        df_B = self.combined_df[self.combined_df.label == "B"]

        n_A, T_A = non_parametric_fit(df_A)
        n_B, T_B = non_parametric_fit(df_B)

        sns.scatterplot(
            data=df_A,
            x="n",
            y="T",
            label=f"A: {self.a.function_name}",
            alpha=0.7,
            ax=axes[0],
        )

        sns.scatterplot(
            data=df_B,
            x="n",
            y="T",
            label=f"B: {self.b.function_name}",
            alpha=0.7,
            ax=axes[0],
        )

        sns.lineplot(
            x=n_A,
            y=T_A,
            color="C0",
            ax=axes[0],
        )

        sns.lineplot(
            x=n_B,
            y=T_B,
            color="C1",
            ax=axes[0],
        )

        for index, result in enumerate(self.ab_results):
            if index > 0:
                axes[0].axvline(
                    result.n_common.min(),
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )

        axes[0].set_xlabel("Input Size (n)")
        axes[0].set_ylabel(f"{self.metric.title()} (T)")
        axes[0].set_title(f"{self.a.function_name} vs. {self.b.function_name}: {self.metric.title()}")
        axes[0].legend()
        axes[0].grid(True)

        # ----------------------------------------
        # 4.2. Difference in Running Times and Permutation Test by Segment
        # ----------------------------------------

        for segment_number, result in enumerate(self.ab_results[-3:]):
            index = segment_number

            valid_perm_stats = result.perm_stats[np.isfinite(result.perm_stats)]

            # Histogram of the permutation test distribution
            sns.histplot(
                valid_perm_stats,
                stat="percent",
                bins=50,
                color="C7",
                alpha=0.8,
                label="Permutation Distribution",
                ax=axes[index + 1],
            )

            axes[index + 1].axvline(
                result.observed_stat,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Observed Statistic",
            )

            axes[index + 1].set_xlabel("Signed Area Between LOESS Curves")
            axes[index + 1].set_ylabel("Frequency")
            axes[index + 1].set_title(
                f"{result.n_common.min():.2f} <= n <= {result.n_common.max():.2f}\n{result.faster} is better (p-value={result.p_value:.3f})"
            )
            axes[index + 1].legend()
            axes[index + 1].grid(True)


@click.command()
@click.option(
    "--output-file",
    "output_file",
    default=None,
    help="Specify the output file to process.",
)
def main(output_file: Optional[str]):
    file_name = output_file or f"{system_name}_data.json"
    with log_timer("Loading data"):
        with open(file_name, "r") as f:
            data = json.load(f)

    with log_timer("Building work items"):
        entries = {}
        work_items: List[Analysis] = []
        for key, function_record in data.items():
            key_str = key.strip("()")
            parts = key_str.split(",")
            function_name = parts[0].strip().strip("'\"")
            file_name = parts[1].strip().strip("'\"")

            records = function_record["observations"]
            log(f"{function_name}: {len(records)} records...")
            lengths = [r["length"] for r in records]
            times = [r["time"] for r in records]
            mems = [r["memory"] for r in records]

            function_data = FunctionData(
                function_name=function_name,
                file_name=file_name,
                lengths=np.array(lengths),
                times=np.array(times),
                mems=np.array(mems),
            )

            entries[key] = function_data

        used_in_tests = set()
        for key, function_record in data.items():
            function_data = entries[key]
            time_bound = function_record["tests"].get("time_bound", None)
            mem_bound = function_record["tests"].get("mem_bound", None)
            ab_test = function_record["tests"].get("abtest", None) 
            if time_bound or mem_bound:
                work_items += [CheckBounds(function_data, time_bound, mem_bound)]
                used_in_tests.add(key)
            elif ab_test:
                alt, metrics = ab_test
                if "time" in metrics:
                    work_items += [ABTest(function_data, entries[alt], "time")]
                if "memory" in metrics:
                    work_items += [ABTest(function_data, entries[alt], "memory")]
                used_in_tests.add(key)
                used_in_tests.add(alt)
            
        for key, function_record in data.items():
            if key not in used_in_tests:
                work_items += [InferPerformance(function_data)]

    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    fig = plt.figure(constrained_layout=True, figsize=(12, 4 * len(work_items)))
    gs_main = gridspec.GridSpec(nrows=len(work_items), ncols=1, figure=fig)
    gs_main.tight_layout(fig, rect=(0, 0, 1, 1))

    for index, item in enumerate(work_items):
        subfig = fig.add_subfigure(gs_main[index, 0])
        item.run()
        message(textwrap.indent(item.message(), "  "))
        message("")
        item.plot(subfig)

    filename = f"{system_name}.pdf"
    plt.savefig(filename)
    print(f"{filename} written.")


if __name__ == "__main__":
    main()

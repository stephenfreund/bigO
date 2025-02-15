import abc
import base64
from dataclasses import dataclass
import io
import json
import os
import textwrap
from typing import List, Optional

from matplotlib import gridspec, pyplot as plt
from matplotlib.figure import Figure, SubFigure
import pandas as pd
from bigO.abtest import non_parametric_fit, segmented_permutation_test
import bigO.models as models

from bigO.output import log, timer, set_debug
import click
import numpy as np
import seaborn as sns

import webbrowser

system_name = "bigO"


@dataclass
class FunctionData:
    function_name: str
    file_name: str
    lengths: np.ndarray
    times: np.ndarray
    mems: np.ndarray


@dataclass
class Result:
    success: bool
    message: str
    warnings: List[str]


class Analysis(abc.ABC):

    @abc.abstractmethod
    def title(self) -> str:
        pass

    @abc.abstractmethod
    def run(self) -> Result:
        pass

    @abc.abstractmethod
    def plot(self, fig: Figure | SubFigure | None = None):
        pass


class InferPerformance(Analysis):

    def __init__(self, function_data: FunctionData):
        self.function_data = function_data

    def title(self) -> str:
        return f"Infer performance for {self.function_data.function_name}"

    def run(self):
        with timer(self.title()):
            self.fitted_times = models.infer_bound(
                self.function_data.lengths, self.function_data.times
            )
            self.fitted_mems = models.infer_bound(
                self.function_data.lengths, self.function_data.mems
            )

        best_time_model = (
            self.fitted_times.models[0] if self.fitted_times.models else None
        )
        best_mem_model = self.fitted_mems.models[0] if self.fitted_mems.models else None
        return Result(
            success=True,
            message=(
                f"Inferred Bounds for {self.function_data.function_name}\n"
                f"Best time model: {best_time_model}\nBest memory model: {best_mem_model}"
            ),
            warnings=self.fitted_times.warnings + self.fitted_mems.warnings,
        )

    def plot_fits(self, ax, models: List[models.FittedModel], color, ylabel, title):
        best_fit = models[0]
        sns.scatterplot(
            x=best_fit.n,
            y=best_fit.y,
            ax=ax,
            color=color,
            label="Data",
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

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        time_ax, mem_ax, extra = fig.subplots(1, 3)
        extra.axis("off")
        self.plot_fits(
            time_ax,
            self.fitted_times.models,
            color="C0",
            ylabel="Time (s)",
            title=f"{self.function_data.function_name}: {self.fitted_times.models[0]}",
        )
        self.plot_fits(
            mem_ax,
            self.fitted_mems.models,
            color="C1",
            ylabel="Memory",
            title=f"{self.function_data.function_name}: {self.fitted_mems.models[0]}",
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

    def title(self) -> str:
        return (
            f"Check bounds for {self.function_data.function_name}."
            + (f" Time: {self.time_bound}." if self.time_bound else "")
            + (f" Mem: {self.mem_bound}." if self.mem_bound else "")
        )

    def run(self) -> Result:
        with timer(self.title()):
            if self.time_bound:
                with timer("Check time models"):
                    self.time_check = models.check_bound(
                        self.function_data.lengths,
                        self.function_data.times,
                        self.time_bound,
                    )
            else:
                self.time_check = None
            if self.mem_bound:
                with timer("Check memory models"):
                    self.mem_check = models.check_bound(
                        self.function_data.lengths,
                        self.function_data.mems,
                        self.mem_bound,
                    )
            else:
                self.mem_check = None

        success = True
        message = []
        if self.time_check:
            success = success and len(self.time_check.better_models) == 0
            message += [
                f"Declared Time Bound for {self.function_data.function_name} is {self.time_bound}, ",
                f"but these models with worse performance better fit the data: ",
                str(self.time_check.better_models.to_string(index=False)),
            ]
        if self.mem_check:
            success = success and len(self.mem_check.better_models) == 0
            message += [
                f"Declared Memory Bound for {self.function_data.function_name}: {self.mem_bound}, ",
                f"but these models with worse performance better fit the data: ",
                str(
                    self.mem_check.better_models[["model", "pvalue"]].to_string(
                        index=False
                    )
                ),
            ]
        warnings = (self.time_check.warnings if self.time_check else []) + (
            self.mem_check.warnings if self.mem_check else []
        )
        return Result(success=success, message="\n".join(message), warnings=warnings)

    def plot_fits(
        self, ax, check_result: models.CheckBoundResult, color, ylabel, title
    ):
        declared_fit = check_result.declared_bound_fit
        sns.scatterplot(
            x=declared_fit.n,
            y=declared_fit.y,
            ax=ax,
            color=color,
            label="Data",
        )
        sns.lineplot(
            x=declared_fit.n,
            y=declared_fit.predict(declared_fit.n),
            ax=ax,
            color=color,
            label=f"{declared_fit}: {declared_fit}",
        )
        for i, (_, row) in enumerate(check_result.better_models.iterrows()):
            if i < 4:
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

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))
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


class CheckLimits(Analysis):
    def __init__(
        self,
        function_data: FunctionData,
        time_limit: float | None = None,
        mem_limit: float | None = None,
        length_limit: int | None = None,
    ):
        super().__init__()
        self.function_data = function_data
        self.time_limit = time_limit
        self.mem_limit = mem_limit
        self.length_limit = length_limit

    def title(self) -> str:
        return (
            f"Check limits for {self.function_data.function_name}. "
            + (f"Time: {self.time_limit}." if self.time_limit else "")
            + (f"Mem: {self.mem_limit}." if self.mem_limit else "")
            + (f"Length: {self.length_limit}." if self.length_limit else "")
        )

    def run(self) -> Result:

        with timer(self.title()):
            success = True
            message = []
            if self.time_limit:
                if not (self.function_data.times.max() < self.time_limit):
                    message += [
                        f"Declared time limit for {self.function_data.function_name} is {self.time_limit}, ",
                        f"but the largest observed time is {self.function_data.times.max()}.",
                    ]
                    success = False
            if self.mem_limit:
                if not (self.function_data.mems.max() < self.mem_limit):
                    message += [
                        f"Declared memory limit for {self.function_data.function_name} is {self.mem_limit}, ",
                        f"but the largest observed memory is {self.function_data.mems.max()}.",
                    ]
                    success = False
            if self.length_limit:
                if not (self.function_data.lengths.max() < self.length_limit):
                    message += [
                        f"Declared length limit for {self.function_data.function_name} is {self.length_limit}, ",
                        f"but the largest observed length is {self.function_data.lengths.max()}.",
                    ]
                    success = False

            warnings = []
            return Result(
                success=success, message="\n".join(message), warnings=warnings
            )

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        time_ax, mem_ax, extra = fig.subplots(1, 3)
        extra.axis("off")
        if self.time_limit:
            sns.histplot(
                x=self.function_data.times,
                color="C0",
                ax=time_ax,
            )
            time_ax.title.set_text(f"{self.function_data.function_name}: Time (s)")
            time_ax.axvline(self.time_limit, color="red")
            time_ax.legend(["Declared limit", "Data"])

        if self.mem_limit:
            sns.histplot(
                self.function_data.mems,
                color="C1",
                title=f"{self.function_data.function_name}: Memory",
                ax=mem_ax,
            )
            mem_ax.axvline(self.mem_limit, color="red")
            mem_ax.legend(["Declared limit", "Data"])


class ABTest(Analysis):
    def __init__(self, a: FunctionData, b: FunctionData, metric: str):
        self.a = a
        self.b = b
        self.metric = metric

    def title(self) -> str:
        return f"AB Test: {self.a.function_name} vs. {self.b.function_name}"

    def run(self):
        with timer(self.title()):
            combined_labels = np.concatenate(
                [["A"] * len(self.a.lengths), ["B"] * len(self.b.lengths)]
            )
            combined_lengths = np.concatenate([self.a.lengths, self.b.lengths])
            combined_T = np.concatenate(
                [
                    self.a.times if self.metric == "time" else self.a.mems,
                    self.b.times if self.metric == "time" else self.b.mems,
                ]
            )
            self.combined_df = pd.DataFrame(
                {
                    "label": combined_labels,
                    "n": combined_lengths,
                    "T": combined_T,
                }
            )
            self.ab_results = segmented_permutation_test(
                self.combined_df,
                num_permutations=1000,
                num_points=100,
            )

        message = [
            f"AB Test for {self.metric}: A is {self.a.function_name}; B is {self.b.function_name}"
        ]
        for report in self.ab_results.segments:
            message += [f"  {x}" for x in str(report).splitlines()]
        return Result(
            success=True, message="\n".join(message), warnings=self.ab_results.warnings
        )

    def plot(self, fig: Figure | SubFigure | None = None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=(12, 4))

        num_figs = min(1 + len(self.ab_results.segments), 4)
        axes = fig.subplots(1, num_figs)

        # ----------------------------------------
        # 4.1. Scatter Plots with Curves (First Visualization)
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

        for index, result in enumerate(self.ab_results.segments):
            if index > 0:
                axes[0].axvline(
                    result.n_common.min(),
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )

        axes[0].set_xlabel("Input Size (n)")
        axes[0].set_ylabel(f"{self.metric.title()} (T)")
        axes[0].set_title(
            f"{self.a.function_name} vs. {self.b.function_name}: {self.metric.title()}"
        )
        axes[0].legend()
        axes[0].grid(True)

        # ----------------------------------------
        # 4.2. Difference in Running Times and Permutation Test by Segment
        # ----------------------------------------

        for segment_number, result in enumerate(self.ab_results.segments[-3:]):
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

            axes[index + 1].set_xlabel("Signed Area Between Smoothed Curves")
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
@click.option(
    "--debug",
    "debug",
    is_flag=True,
    help="Show debug output.",
)
@click.option(
    "--open-report",
    "open_report",
    is_flag=True,
    help="Open the output in a web browser.",
)
@click.option(
    "--html",
    "html",
    is_flag=True,
    help="Generate an HTML report instead of a PDF.",
)
def main(output_file: Optional[str], debug: bool, open_report: bool, html: bool):

    set_debug(debug)

    file_name = output_file or f"{system_name}_data.json"
    with timer("Loading data"):
        with open(file_name, "r") as f:
            data = json.load(f)

    with timer("Building work items"):
        entries = {}
        work_items: List[Analysis] = []
        for key, function_record in data.items():
            key_str = key.strip("()")
            parts = key_str.split(",")
            function_name = parts[1].strip().strip("'\"")
            file_name = parts[0].strip().strip("'\"")

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
            time_limit = function_record["tests"].get("time_limit", None)
            mem_limit = function_record["tests"].get("mem_limit", None)
            ab_test = function_record["tests"].get("abtest", None)
            if time_bound or mem_bound:
                work_items += [CheckBounds(function_data, time_bound, mem_bound)]
                used_in_tests.add(key)
            if time_limit or mem_limit:
                work_items += [CheckLimits(function_data, time_limit, mem_limit)]
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
                function_data = entries[key]
                if all(function_data.lengths >= 0):
                    work_items += [InferPerformance(function_data)]

    if html:
        filename = run_html(work_items)
    else:
        filename = run_pdf(work_items)
    if open_report:
        webbrowser.open(f"file://{os.getcwd()}/{filename}")


def run_pdf(work_items) -> str:
    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    fig = plt.figure(constrained_layout=True, figsize=(12, 4 * len(work_items)))
    gs_main = gridspec.GridSpec(nrows=len(work_items), ncols=1, figure=fig, hspace=0.1)
    gs_main.tight_layout(fig, rect=(0, 0, 1, 1))

    for index, item in enumerate(work_items):
        print()
        print(item.title())
        print("-" * len(item.title()))
        subfig = fig.add_subfigure(gs_main[index, 0])
        result = item.run()
        print(textwrap.indent(result.message, "  "))
        print("")
        if result.warnings:
            print(textwrap.indent("\n".join(["Warnings:"] + result.warnings), "  "))
            print()
        item.plot(subfig)
        if result.success:
            subfig.suptitle(f"{item.title()}")
        else:
            subfig.suptitle(f"FAILED -- {item.title()}", color="red")

    filename = f"{system_name}.pdf"
    plt.savefig(filename)
    print(f"{filename} written.")
    return filename


def run_html(work_items) -> str:
    # Assume these are defined elsewhere:
    # - work_items: a list of objects, each with methods .title(), .run(), and .plot(ax)
    # - system_name: a string used for naming the output file (without extension)
    # - open_plots: a boolean flag to open the result in a web browser

    # Set up the plotting style and palette.
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

    # Prepare an HTML page as a list of strings.
    html_lines = [
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Work Items Output</title>",
        "</head>",
        "<body>",
        "  <h1>BigO Report</h1>",
    ]

    # Process each work item separately.
    for item in work_items:
        title = item.title()
        print(f"{title}...")
        result = item.run()
        mark = (
            "<span style='color:blue;'>&#10003;</span>"
            if result.success
            else "<span style='color:red;'>&#10007;</span>"
        )
        html_lines.append(f"<h2>{mark} {title}</h2>")

        # Create a new figure and axis for this work item.
        fig_item = Figure(figsize=(12, 4))

        # Run the work item and capture its result.
        result = item.run()

        # Add the result message.
        html_lines.append("<pre>")
        html_lines.append(textwrap.indent(result.message, "  "))
        html_lines.append("</pre>")

        # Let the work item create its plot on the current axis.
        item.plot(fig_item)

        # Save the figure as a PNG image into a bytes buffer.
        buf = io.BytesIO()
        fig_item.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Embed the PNG image (as a Base64 data URL) in the HTML.
        html_lines.append(
            f'<img src="data:image/png;base64,{image_base64}" alt="Plot for {title}">'
        )

        # Close the figure to free memory.
        plt.close(fig_item)

        # If there are warnings, include them.
        if result.warnings:
            warnings_text = "\n".join(["Warnings:"] + result.warnings)
            html_lines.append("<pre>")
            html_lines.append(textwrap.indent(warnings_text, "  "))
            html_lines.append("</pre>")

    html_lines.append("</body>")
    html_lines.append("</html>")

    # Write the HTML content to a file.
    html_filename = f"{system_name}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    print(f"{html_filename} written.")
    return html_filename


if __name__ == "__main__":
    main()

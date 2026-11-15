"""Build the 200-chunk retrieval test fixture (tests/fixtures/chunks_200.jsonl).

Usage: python scripts/make_chunk_fixture.py [--out tests/fixtures/chunks_200.jsonl]

The chunks are synthetic, written by us from general API knowledge (no benchmark material). They cover docs and
Stack Overflow, six libraries, two versions each, and API changes across major versions (removed, renamed or
deprecated functions) so the BM25, dense, fusion and version-filter tests have realistic material. Output is
deterministic: re-running produces a byte-identical file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import typer

from snakepp.schemas import Chunk, make_chunk_id

N_CHUNKS = 200
DEFAULT_OUT = Path("tests/fixtures/chunks_200.jsonl")

DOCS_URL = {
    "pandas": "https://pandas.pydata.org/pandas-docs/version/{v}/reference/api/{s}.html",
    "numpy": "https://numpy.org/doc/{v}/reference/generated/{s}.html",
    "matplotlib": "https://matplotlib.org/{v}/api/_as_gen/{s}.html",
    "scikit-learn": "https://scikit-learn.org/{v}/modules/generated/{s}.html",
    "scipy": "https://docs.scipy.org/doc/scipy-{v}/reference/generated/{s}.html",
    "python": "https://docs.python.org/{v}/library/{s}.html",
}
DOCS_LICENSE = {
    "pandas": "BSD-3-Clause",
    "numpy": "BSD-3-Clause",
    "matplotlib": "PSF-based Matplotlib License",
    "scikit-learn": "BSD-3-Clause",
    "scipy": "BSD-3-Clause",
    "python": "PSF-2.0",
}
# Approximate release dates, used to date Stack Overflow answers inside each version's era.
RELEASED = {
    ("pandas", "1.5.3"): date(2023, 1, 19),
    ("pandas", "2.2.2"): date(2024, 4, 10),
    ("numpy", "1.26.4"): date(2024, 2, 5),
    ("numpy", "2.0.0"): date(2024, 6, 16),
    ("matplotlib", "3.7.0"): date(2023, 2, 13),
    ("matplotlib", "3.9.0"): date(2024, 5, 15),
    ("scikit-learn", "1.2.2"): date(2023, 3, 8),
    ("scikit-learn", "1.5.0"): date(2024, 5, 21),
    ("scipy", "1.11.4"): date(2023, 11, 18),
    ("scipy", "1.13.0"): date(2024, 4, 2),
    ("python", "3.11"): date(2022, 10, 24),
    ("python", "3.12"): date(2023, 10, 2),
}


@dataclass(frozen=True)
class Api:
    library: str
    symbol: str
    summary: str
    example: str
    notes: dict[str, str]  # version -> version-specific sentence; docs exist only for these versions


APIS: list[Api] = [
    # pandas
    Api("pandas", "pandas.concat", "Concatenate pandas objects along a particular axis.",
        "df = pd.concat([df1, df2], ignore_index=True)",
        {"1.5.3": "", "2.2.2": "Recommended replacement for the removed DataFrame.append."}),
    Api("pandas", "pandas.DataFrame.append", "Append rows of other to the end of caller, returning a new object.",
        "df = df.append(other, ignore_index=True)",
        {"1.5.3": "Deprecated since 1.4.0; use pandas.concat instead."}),
    Api("pandas", "pandas.DataFrame.iteritems", "Iterate over (column name, Series) pairs.",
        "for name, col in df.iteritems():\n    print(name, col.sum())",
        {"1.5.3": "Deprecated since 1.5.0; use DataFrame.items instead."}),
    Api("pandas", "pandas.DataFrame.items", "Iterate over (column name, Series) pairs.",
        "for name, col in df.items():\n    print(name, col.sum())", {"1.5.3": "", "2.2.2": ""}),
    Api("pandas", "pandas.DataFrame.groupby", "Group DataFrame using a mapper or by a Series of columns.",
        "df.groupby('team')['score'].mean()",
        {"1.5.3": "numeric_only defaults to None for mean.",
         "2.2.2": "numeric_only now defaults to False; non-numeric columns raise TypeError in mean."}),
    Api("pandas", "pandas.DataFrame.merge", "Merge DataFrame or named Series objects with a database-style join.",
        "pd.merge(left, right, on='key', how='left')", {"1.5.3": "", "2.2.2": ""}),
    Api("pandas", "pandas.to_datetime", "Convert argument to datetime.",
        "pd.to_datetime(df['date'], format='%Y-%m-%d')",
        {"1.5.3": "Without format, each element is parsed independently.",
         "2.2.2": "Without format, the format is inferred from the first element and applied to all."}),
    Api("pandas", "pandas.DataFrame.pivot_table", "Create a spreadsheet-style pivot table as a DataFrame.",
        "df.pivot_table(index='region', columns='year', values='sales', aggfunc='sum')",
        {"1.5.3": "", "2.2.2": ""}),
    Api("pandas", "pandas.Series.str.contains", "Test if pattern or regex is contained within a string of a Series.",
        "df[df['name'].str.contains('ann', case=False, na=False)]", {"1.5.3": "", "2.2.2": ""}),
    Api("pandas", "pandas.DataFrame.map", "Apply a function to a DataFrame elementwise.",
        "df.map(lambda x: round(x, 2))", {"2.2.2": "Added in 2.1.0 as the new name of DataFrame.applymap."}),
    Api("pandas", "pandas.DataFrame.applymap", "Apply a function to a DataFrame elementwise.",
        "df.applymap(lambda x: round(x, 2))",
        {"1.5.3": "", "2.2.2": "Deprecated since 2.1.0; use DataFrame.map instead."}),
    # numpy
    Api("numpy", "numpy.concatenate", "Join a sequence of arrays along an existing axis.",
        "np.concatenate([a, b], axis=0)", {"1.26.4": "", "2.0.0": ""}),
    Api("numpy", "numpy.product", "Return the product of array elements over a given axis.",
        "np.product(arr, axis=1)", {"1.26.4": "Deprecated alias of numpy.prod."}),
    Api("numpy", "numpy.prod", "Return the product of array elements over a given axis.",
        "np.prod(arr, axis=1)", {"1.26.4": "", "2.0.0": "numpy.product was removed; use numpy.prod."}),
    Api("numpy", "numpy.in1d", "Test whether each element of a 1-D array is also present in a second array.",
        "mask = np.in1d(a, b)",
        {"1.26.4": "", "2.0.0": "Deprecated since 2.0; use numpy.isin instead."}),
    Api("numpy", "numpy.isin", "Calculates element in test_elements, broadcasting over element only.",
        "mask = np.isin(a, b)", {"1.26.4": "", "2.0.0": ""}),
    Api("numpy", "numpy.random.Generator.integers", "Return random integers from low (inclusive) to high.",
        "rng = np.random.default_rng(0)\nx = rng.integers(0, 10, size=5)", {"1.26.4": "", "2.0.0": ""}),
    Api("numpy", "numpy.where", "Return elements chosen from x or y depending on condition.",
        "np.where(arr > 0, arr, 0)", {"1.26.4": "", "2.0.0": ""}),
    Api("numpy", "numpy.round_", "Round an array to the given number of decimals.",
        "np.round_(arr, 2)", {"1.26.4": "Deprecated alias of numpy.round."}),
    # matplotlib
    Api("matplotlib", "matplotlib.pyplot.subplots", "Create a figure and a set of subplots.",
        "fig, ax = plt.subplots(2, 2, figsize=(8, 6))", {"3.7.0": "", "3.9.0": ""}),
    Api("matplotlib", "matplotlib.cm.get_cmap", "Get a colormap instance by name.",
        "cmap = matplotlib.cm.get_cmap('viridis')",
        {"3.7.0": "Deprecated since 3.7; use matplotlib.colormaps[name] instead."}),
    Api("matplotlib", "matplotlib.colormaps", "Container of all registered colormaps, indexable by name.",
        "cmap = matplotlib.colormaps['viridis']",
        {"3.7.0": "", "3.9.0": "matplotlib.cm.get_cmap was removed in 3.9; index matplotlib.colormaps."}),
    Api("matplotlib", "matplotlib.pyplot.style.use", "Use Matplotlib style settings from a style specification.",
        "plt.style.use('seaborn-v0_8-whitegrid')",
        {"3.7.0": "The old 'seaborn' style names were renamed to 'seaborn-v0_8-<style>'.", "3.9.0": ""}),
    Api("matplotlib", "matplotlib.axes.Axes.bar_label", "Label a bar plot with the bar values.",
        "bars = ax.bar(x, heights)\nax.bar_label(bars, fmt='%.1f')", {"3.7.0": "", "3.9.0": ""}),
    # scikit-learn
    Api("scikit-learn", "sklearn.preprocessing.OneHotEncoder", "Encode categorical features as a one-hot array.",
        "enc = OneHotEncoder(sparse_output=False)\nX_enc = enc.fit_transform(X)",
        {"1.2.2": "The sparse parameter is deprecated in favour of sparse_output.",
         "1.5.0": "The sparse parameter was removed in 1.4; use sparse_output."}),
    Api("scikit-learn", "sklearn.metrics.mean_squared_error", "Mean squared error regression loss.",
        "mse = mean_squared_error(y_true, y_pred)",
        {"1.2.2": "Pass squared=False to get the root mean squared error.",
         "1.5.0": "The squared parameter is deprecated; use root_mean_squared_error."}),
    Api("scikit-learn", "sklearn.metrics.root_mean_squared_error", "Root mean squared error regression loss.",
        "rmse = root_mean_squared_error(y_true, y_pred)", {"1.5.0": "Added in 1.4."}),
    Api("scikit-learn", "sklearn.model_selection.train_test_split", "Split arrays into random train and test subsets.",
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)",
        {"1.2.2": "", "1.5.0": ""}),
    Api("scikit-learn", "sklearn.pipeline.Pipeline", "A sequence of transformers with an optional final predictor.",
        "pipe = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression())])",
        {"1.2.2": "", "1.5.0": "Supports set_output(transform='pandas') to keep DataFrame outputs."}),
    # scipy
    Api("scipy", "scipy.integrate.simps", "Integrate y(x) using samples along the given axis and Simpson's rule.",
        "area = simps(y, x=x)", {"1.11.4": "Deprecated alias of scipy.integrate.simpson."}),
    Api("scipy", "scipy.integrate.simpson", "Integrate y(x) using samples along the given axis and Simpson's rule.",
        "area = simpson(y, x=x)", {"1.11.4": "", "1.13.0": "x and dx are now keyword-only."}),
    Api("scipy", "scipy.stats.ttest_ind", "Calculate the T-test for the means of two independent samples.",
        "res = ttest_ind(a, b, equal_var=False)\nprint(res.pvalue)", {"1.11.4": "", "1.13.0": ""}),
    Api("scipy", "scipy.optimize.minimize", "Minimization of a scalar function of one or more variables.",
        "res = minimize(f, x0=[0.0, 0.0], method='BFGS')", {"1.11.4": "", "1.13.0": ""}),
    # python stdlib
    Api("python", "itertools.batched", "Batch data from an iterable into tuples of length n.",
        "for batch in batched(range(10), 3):\n    print(batch)", {"3.12": "Added in 3.12."}),
    Api("python", "tomllib", "Parse TOML files.",
        "with open('pyproject.toml', 'rb') as f:\n    data = tomllib.load(f)", {"3.11": "Added in 3.11.", "3.12": ""}),
    Api("python", "functools.lru_cache", "Decorator to wrap a function with a memoizing callable.",
        "@lru_cache(maxsize=128)\ndef fib(n): ...", {"3.11": "", "3.12": ""}),
    Api("python", "datetime.datetime.utcnow", "Return the current UTC date and time, with tzinfo None.",
        "now = datetime.utcnow()",
        {"3.11": "", "3.12": "Deprecated since 3.12; use datetime.now(timezone.utc)."}),
]

SO_AUTHORS = ["data_wrangler", "np_fan", "plot_twist", "fit_predict", "sci_guy", "pyuser42", "kernel_panic"]


def _docs_chunks() -> list[Chunk]:
    chunks = []
    for api in APIS:
        for version, note in api.notes.items():
            url = DOCS_URL[api.library].format(v=version, s=api.symbol)
            sections = {
                "summary": f"{api.symbol}\n\n{api.summary} {note}".strip(),
                "example": f"{api.symbol} example\n\n```python\n{api.example}\n```",
            }
            for section, text in sections.items():
                chunks.append(Chunk(
                    id=make_chunk_id(url, section), text=text, source="docs", library=api.library,
                    version=version, url=url, title=f"{api.symbol} — {section}", license=DOCS_LICENSE[api.library],
                ))
    return chunks


def _so_chunk(i: int, api: Api) -> Chunk:
    versions = list(api.notes)
    version = versions[i % len(versions)]
    created = date.fromordinal(RELEASED[(api.library, version)].toordinal() + 7 + (i * 37) % 300)
    unknown = i % 10 == 9  # some posts where version inference fails
    note = api.notes[version] or "Works as documented."
    url = f"https://stackoverflow.com/a/{70_000_000 + i * 1_013}"
    text = (
        f"Q: How do I use {api.symbol.rsplit('.', 1)[-1]} in {api.library}?\n\n"
        f"A: {api.summary} {note}\n\n```python\n{api.example}\n```"
    )
    return Chunk(
        id=make_chunk_id(url), text=text, source="stackoverflow", library=api.library,
        version=None if unknown else version, version_inferred=not unknown, url=url,
        title=f"How to use {api.symbol}", score=(i * 17) % 120 - 5, created=created,
        license="CC BY-SA 4.0", author=SO_AUTHORS[i % len(SO_AUTHORS)],
    )


def build_chunks(n: int = N_CHUNKS) -> list[Chunk]:
    """All docs chunks, then Stack Overflow answers cycling over the APIs until there are exactly n chunks."""
    chunks = _docs_chunks()
    if len(chunks) > n:
        raise ValueError(f"{len(chunks)} docs chunks already exceed n={n}")
    chunks += [_so_chunk(i, APIS[i % len(APIS)]) for i in range(n - len(chunks))]
    return chunks


def write_jsonl(chunks: list[Chunk], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8", newline="\n")


def main(out: Path = DEFAULT_OUT) -> None:
    chunks = build_chunks()
    write_jsonl(chunks, out)
    typer.echo(f"wrote {len(chunks)} chunks to {out}")


if __name__ == "__main__":
    typer.run(main)

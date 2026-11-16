# Contributing

Bug reports, questions and pull requests are welcome.

## Reporting a problem

Open an issue at <https://github.com/k-sakel/ordinal-networks/issues> with

- the version (`python -c "import ordinal_networks as on; print(on.__version__)"`),
  operating system and Python version;
- the smallest series and call that show the problem (a seed for
  `numpy.random.default_rng` or a `systems.generate(...)` call is ideal);
- what you expected and what you got.

For a result you believe is numerically wrong, a comparison with another
implementation or an analytic value helps most.

## Development setup

You need Python ≥ 3.10 and a Rust toolchain (<https://rustup.rs>; on Windows also the
"Desktop development with C++" workload of Visual Studio Build Tools, or the
`x86_64-pc-windows-gnu` toolchain).

```
git clone https://github.com/k-sakel/ordinal-networks
cd ordinal-networks
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install maturin
pip install -e ".[dev]"                            # builds the Rust core in release mode
```

After changing Rust code, rebuild with `maturin develop --release`. Python changes
take effect immediately (editable install).

## Checks

```
cargo test --release --lib        # Rust unit tests
pytest                            # Python tests, incl. cross-checks against ordpy
ruff check python tests benchmarks examples paper
mkdocs build --strict             # documentation
```

CI runs the same on Linux, macOS and Windows.

## Guidelines

- **Every fix comes with a test** that fails without it. Numerical code is checked
  against a brute-force reference, another package, or an analytic value.
- **Results must not depend on performance settings.** Chunk size, thread count,
  memory budget and spilling may change speed and memory only; tests assert this.
- **Fail loudly.** Bad input raises `ValueError`/`TypeError` with a message that says
  what to change; silent fallbacks are not accepted.
- **State units and conventions** in the docstring: bits or nats, which ranking,
  how ties and non-finite values are treated, and the reference for the definition.
- Public API changes go in `CHANGELOG.md` under the unreleased version.

## Releasing (maintainers)

1. Update the version in `pyproject.toml`, `Cargo.toml` and `CITATION.cff`, and date
   the changelog entry.
2. Tag `vX.Y.Z` and push the tag. CI builds wheels and the sdist, creates the GitHub
   release, and publishes to PyPI through trusted publishing.
3. Zenodo archives the GitHub release and mints a DOI; add it to `CITATION.cff` and
   the README.

## Licence

By contributing you agree that your contribution is released under the MIT licence.

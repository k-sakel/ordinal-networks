# Paper

`manuscript.tex` is the software paper (AIP REVTeX 4.2, target journal *Chaos*).
No number in it is typed by hand: figures come from `make_figures.py`, and every
measured value is a macro in `numbers.tex`, written by `make_numbers.py` from the
JSON files in `results/`.

## Regenerate everything

```
pip install -e ".[paper]"
python benchmarks/compare.py --out paper/results/benchmark.json   # timings vs ordpy, thread scaling
python paper/make_figures.py                                      # figures/*.pdf, results/*.json
python paper/make_numbers.py                                      # numbers.tex
cd paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`make_figures.py` runs at reduced sizes by default (largest series 10^8 points, about
1 GB on disk under `paper/data/`, which is git-ignored). `--full` uses 10^9 points for
the scale figure and needs about 8 GB of disk and a machine with more than 16 GB of
memory for comfortable page caching. `--only <name>` regenerates one item.

Timings depend on the machine; `results/benchmark.json` records the platform,
processor count and package versions it was measured with. Re-run the benchmark
before quoting numbers for a different machine.

## Files

| file | content |
|---|---|
| `manuscript.tex`, `references.bib` | the paper; all bibliography entries were checked against Crossref/OpenAlex |
| `numbers.tex` | generated macros for measured values |
| `make_figures.py`, `make_numbers.py` | generation scripts |
| `figures/` | generated figures (PDF and PNG) |
| `results/` | generated numbers (JSON), including the benchmark record |

## Open items before submission

Marked `\TODO{...}` in the manuscript:

- co-authors, affiliation, corresponding e-mail, funding;
- Zenodo DOI of the tagged release;
- real-data case study (data set to be chosen);
- comparison table and benchmark extended to antropy, EntropyHub, pyunicorn, teaspoon
  and ComplexityMeasures.jl; peak-memory measurements; a second machine;
- the AI-assistance disclosure wording required by the journal.

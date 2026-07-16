# RAG Evaluation Design

## Scope

The evaluation checks two parts of the RAG pipeline:

1. whether the expected filing chunks are retrieved; and
2. whether the final answer is relevant, supported and correctly cited.



## Evaluation set

`rag_questions.json` contains 91 questions built from `data/filings`:

| Question type                       | Count |
| ----------------------------------- | ----: |
| Single-filing factual: 10-K Item 1A |    21 |
| Section-specific: 10-Q Item 2 MD&A  |    23 |
| Earnings-call transcript            |    23 |
| Cross-filing comparison             |    16 |
| Unanswerable                        |     8 |

There are 83 answerable questions, 8 unanswerable questions and 99 gold
sources. Each gold source stores the expected filing metadata, chunk ID and
evidence text.

The set can be rebuilt with:

```powershell
python eval\build_rag_evaluation_set.py
```

Only filters stated in the question are used during retrieval. These can
include ticker, filing form, date and Item. Gold-source metadata is used after
retrieval for scoring and is not passed to the retriever as a hidden hint.

## Evaluation modes

The CLI has three answer modes:

| Mode                | Command flag   | Behaviour                                                                                                                                          |
| ------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Retrieval-only      | No mode flag   | Runs the normal retrieval and reranking pipeline, then creates a local extractive answer. It never calls OpenAI or Gemini.                         |
| RAG answer          | `--with-llm`   | Runs the normal pipeline and generates an answer. It tries OpenAI first, then Gemini, then an extractive fallback if both are unavailable or fail. |
| Extractive baseline | `--extractive` | Uses direct index search and a simple extractive answer. It is mainly used with `--bm25-only`.                                                     |

`--with-llm` and `--extractive` are mutually exclusive. Results from different
modes should be reported separately.


## Metric count

The evaluator calculates 19 named fields internally:

- 5 main metric fields;
- 14 diagnostic fields;
- 10 of these 19 fields are used in the answerable-case overall score; and
- 9 have zero weight and are used only for diagnosis or reporting.

`recall_at_k` and `retrieval_hit` are aliases of contextual recall.
`hallucination_rate` is calculated from faithfulness. They are not independent
measurements.

The summary JSON contains all 19 fields. The per-question JSON and CSV contain
a smaller selected set together with the overall score, pass result and failure
reason.

## Answerable-question score

The following formula is the current implementation in `rag_eval.py`:

```text
overall score =
    contextual recall       × 0.30
  + contextual precision    × 0.15
  + MRR                     × 0.15
  + evidence hit            × 0.12
  + exact chunk hit         × 0.08
  + metadata hit            × 0.05
  + answer correctness      × 0.05
  + answer relevancy        × 0.03
  + faithfulness            × 0.05
  + citation correctness    × 0.02
```

The first six metrics are retrieval metrics and total 85%. The last four are
answer metrics and total 15%. The 85/15 split applies only to answerable
questions.

| Metric               | Weight | Implementation                                                                           |
| -------------------- | -----: | ---------------------------------------------------------------------------------------- |
| Contextual recall    |    30% | Gold sources matched by at least one retrieved chunk, divided by all gold sources.       |
| Contextual precision |    15% | Retrieved chunks that match any gold source, divided by all retrieved chunks.            |
| MRR                  |    15% | Reciprocal rank of the first relevant chunk. Rank 1 gives 1.0 and rank 2 gives 0.5.      |
| Evidence hit         |    12% | Gold evidence passages covered by retrieved text, divided by all gold evidence passages. |
| Exact chunk hit      |     8% | Gold sources whose exact chunk ID appears in the retrieved list.                         |
| Metadata hit         |     5% | Gold sources matched on ticker, form, date, Item and section metadata.                   |
| Answer correctness   |     5% | Word and normalised-number coverage against the gold answer and evidence.                |
| Answer relevancy     |     3% | Coverage of the meaningful question terms, normalised against the gold references.       |
| Faithfulness         |     5% | Answer sentences supported by the retrieved chunks.                                      |
| Citation correctness |     2% | `[n]` citations that point to relevant retrieved chunks, divided by all citations.       |

## Unanswerable-question score

Unanswerable questions use a different diagnostic formula:

```text
diagnostic overall score =
    abstention accuracy × 0.55
  + answer relevancy    × 0.25
  + faithfulness        × 0.20
```

However, the actual pass decision does not use the `0.77` answerable-question
threshold. An unanswerable question passes only when `abstention_accuracy` is
`1.0`.

The abstention check looks for an explicit statement that the requested
information is not available, cannot be found or cannot be confirmed. A reply
that invents an answer does not pass.


## Thresholds and defaults

### CLI defaults

| Setting          | Code default | Meaning                                                                         |
| ---------------- | -----------: | ------------------------------------------------------------------------------- |
| `top_k`          |            5 | Number of retrieved chunks scored for each question.                            |
| `pass_threshold` |         0.77 | Minimum overall score for an answerable question.                               |
| `min_pass_rate`  |      Not set | The CLI does not enforce a full-run pass rate unless this argument is supplied. |
| `with_llm`       |        False | Default mode does not call an external LLM.                                     |
| `extractive`     |        False | The direct-search baseline is not enabled by default.                           |
| `bm25_only`      |        False | The normal run uses the persistent hybrid index.                                |
| `rebuild_index`  |        False | Existing embeddings are reused by default.                                      |

The project-level acceptance target is 85%. It becomes an enforced CLI check
only when the command includes `--min-pass-rate 0.85`. If this flag is omitted,
the result files are still written but the CLI does not fail solely because the
full-run pass rate is below 85%.


## Recommended run commands

The explicit parameters below are recommended for a recorded project result,
even though `top_k = 5` and `pass_threshold = 0.77` are already code defaults.

Default retrieval-only evaluation without OpenAI or Gemini:

```powershell
python src\finsight\rag_eval.py --top-k 5 --pass-threshold 0.77 --min-pass-rate 0.85
```

Generated-answer evaluation. OpenAI is tried first, then Gemini, then the
extractive fallback:

```powershell
python src\finsight\rag_eval.py --with-llm --top-k 5 --pass-threshold 0.77 --min-pass-rate 0.85
```

BM25 extractive baseline:

```powershell
python src\finsight\rag_eval.py --extractive --bm25-only --top-k 5 --pass-threshold 0.77 --min-pass-rate 0.85
```

A full run writes result files before checking `min_pass_rate`. A command can
therefore return exit code 1 for a low pass rate and still produce the report.

`--rebuild-index` should only be added when the corpus, embedding model or
chunk settings have intentionally changed, or when no valid persistent index
exists. 

## Output files

Each full run creates:

- `rag_eval_results.json`: selected per-question metrics and failure details;
- `rag_eval_results.csv`: table version of the same per-question results;
- `rag_eval_summary.json`: all metric averages, pass rates and run metadata;
  and
- `rag_eval_report.md`: readable summary, failed cases and reproduction
  commands.

The provider, generation model, evaluation mode, date, `top_k`, embedding model
and index fingerprint should be kept with any result used in the final report.

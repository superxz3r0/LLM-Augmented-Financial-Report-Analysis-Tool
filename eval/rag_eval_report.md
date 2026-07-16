# RAG Evaluation Report

## Executive Summary

| Metric | Result |
|---|---:|
| Total cases | 91 |
| Passed | 82 |
| Failed | 9 |
| Pass rate | 90.1% |
| Primary score | Retrieval 85% + answer 15% |
| Answerable pass rate | 89.2% |
| Unanswerable pass rate | 100.0% |
| Evaluation mode | rag_answer |
| Actual answer backend(s) | openai: 91 |
| Fallbacks after provider failure | 0 |
| Retrieval backend | hybrid(bm25+bge) |
| Existing index reused | Yes |
| Embeddings regenerated | No |
| Evaluation timestamp | 2026-07-15T22:54:41+00:00 |

Pass rate: [######################--] 90.1%

## Retrieval Metrics

| Metric | Result |
|---|---:|
| Recall@1 | 92.2% |
| Recall@3 | 97.6% |
| Recall@5 | 98.2% |
| MRR | 0.976 |
| Evidence hit rate | 97.6% |
| Exact chunk hit rate | 93.4% |
| Contextual precision | 38.1% |
| Contextual recall | 98.2% |

## Answer Metrics

| Metric | Result |
|---|---:|
| Answer correctness | 37.8% |
| Answer relevancy | 98.4% |
| Faithfulness | 59.0% |
| Citation correctness | 46.7% |
| Hallucination rate | 41.0% |
| Abstention accuracy | 100.0% |

## Results By Question Type

| Question type | Total | Passed | Failed | Pass rate |
|---|---:|---:|---:|---:|
| cross-filing comparison | 16 | 11 | 5 | 68.8% |
| section-specific | 23 | 23 | 0 | 100.0% |
| single-filing factual | 21 | 19 | 2 | 90.5% |
| transcript QA | 23 | 21 | 2 | 91.3% |
| unanswerable | 8 | 8 | 0 | 100.0% |

## Failure Category Summary

| Category | Cases |
|---|---:|
| Overall quality failure | 8 |
| Retrieval failure | 1 |

## Failed Cases

| ID | Type | Overall score | Failed metrics | Retrieved chunk(s) | Gold chunk(s) | Reason | Next action |
|---|---|---:|---|---|---|---|---|
| RAG_Q012 | single-filing factual | 0.763 | overall_score=0.763 (<0.77) | META_10-K_2026-01-29#80;META_10-K_2026-01-29#251;META_10-K_2026-01-29#250;MET... | META_10-K_2026-01-29#80;META_10-K_2026-01-29#79;META_10-K_2026-01-29#81 | overall_score=0.763 (<0.77) | Review the retrieved chunks and the generated answer. |
| | Question | | According to Item 1A of Meta's 2026-01-29 10-K, what does the filing say about customer demand or sales channels? | | | | |
| RAG_Q016 | single-filing factual | 0.755 | overall_score=0.755 (<0.77) | PEP_10-K_2026-02-03#77;PEP_10-K_2026-02-03#69;PEP_10-K_2026-02-03#50;PEP_10-K... | PEP_10-K_2026-02-03#69;PEP_10-K_2026-02-03#68;PEP_10-K_2026-02-03#70 | overall_score=0.755 (<0.77) | Review the retrieved chunks and the generated answer. |
| | Question | | According to Item 1A of PepsiCo's 2026-02-03 10-K, what does the filing say about customer demand or sales channels? | | | | |
| RAG_Q047 | transcript QA | 0.765 | overall_score=0.765 (<0.77) | AMZN_TRANSCRIPT_2026-04-29#31;AMZN_TRANSCRIPT_2026-04-29#26;AMZN_TRANSCRIPT_2... | AMZN_TRANSCRIPT_2026-04-29#26;AMZN_TRANSCRIPT_2026-04-29#25;AMZN_TRANSCRIPT_2... | overall_score=0.765 (<0.77) | Review the retrieved chunks and the generated answer. |
| | Question | | In Amazon's earnings call transcript dated 2026-04-29, what was said about revenue or net sales? | | | | |
| RAG_Q059 | transcript QA | 0.128 | overall_score=0.128 (<0.77); contextual_recall=0.000 (<0.60); contextual_prec... | MSFT_TRANSCRIPT_2026-04-29#10;MSFT_TRANSCRIPT_2026-04-29#38;MSFT_TRANSCRIPT_2... | MSFT_TRANSCRIPT_2026-04-29#16;MSFT_TRANSCRIPT_2026-04-29#15;MSFT_TRANSCRIPT_2... | overall_score=0.128 (<0.77); contextual_recall=0.000 (<0.60); contextual_precision=0.000 (<0.20); evidence_hit=0.000... | Review chunking, metadata filters, and gold evidence coverage. |
| | Question | | In Microsoft's earnings call transcript dated 2026-04-29, what was said about revenue or net sales? | | | | |
| RAG_Q069 | cross-filing comparison | 0.723 | overall_score=0.723 (<0.77) | AMZN_10-K_2024-02-02#72;AMZN_10-K_2024-02-02#89;AMZN_10-K_2024-02-02#42;AMZN_... | AMZN_10-K_2024-02-02#72;AMZN_10-K_2024-02-02#71;AMZN_10-K_2024-02-02#73;AMZN_... | overall_score=0.723 (<0.77) | Review the retrieved chunks and the generated answer. |
| | Question | | Comparing Item 1A of Amazon's 10-K filings from 2024-02-02 and 2025-02-07, what changed or stayed important regarding liquidity or capital risk? | | | | |
| RAG_Q074 | cross-filing comparison | 0.543 | overall_score=0.543 (<0.77); contextual_recall=0.500 (<0.60); evidence_hit=0.... | GOOGL_10-K_2025-02-05#116;GOOGL_10-K_2025-02-05#78;GOOGL_10-K_2024-01-31#80;G... | GOOGL_10-K_2024-01-31#136;GOOGL_10-K_2024-01-31#135;GOOGL_10-K_2024-01-31#137... | overall_score=0.543 (<0.77); contextual_recall=0.500 (<0.60); evidence_hit=0.500 (<0.60) | Review the retrieved chunks and the generated answer. |
| | Question | | Comparing Item 1A of Alphabet's 10-K filings from 2024-01-31 and 2025-02-05, what changed or stayed important regarding liquidity or capital risk and customer demand or sales ch... | | | | |
| RAG_Q078 | cross-filing comparison | 0.740 | overall_score=0.740 (<0.77) | META_10-K_2024-02-02#96;META_10-K_2025-01-30#262;META_10-K_2024-02-02#84;META... | META_10-K_2024-02-02#96;META_10-K_2024-02-02#95;META_10-K_2024-02-02#97;META_... | overall_score=0.740 (<0.77) | Review the retrieved chunks and the generated answer. |
| | Question | | Comparing Item 1A of Meta's 10-K filings from 2024-02-02 and 2025-01-30, what changed or stayed important regarding customer demand or sales channels? | | | | |
| RAG_Q079 | cross-filing comparison | 0.759 | overall_score=0.759 (<0.77); evidence_hit=0.500 (<0.60) | MSFT_10-K_2023-07-27#9;MSFT_10-K_2024-07-30#11;MSFT_10-K_2023-07-27#8;MSFT_10... | MSFT_10-K_2023-07-27#10;MSFT_10-K_2023-07-27#9;MSFT_10-K_2023-07-27#11;MSFT_1... | overall_score=0.759 (<0.77); evidence_hit=0.500 (<0.60) | Review the retrieved chunks and the generated answer. |
| | Question | | Comparing Item 1A of Microsoft's 10-K filings from 2023-07-27 and 2024-07-30, what changed or stayed important regarding customer demand or sales channels and regulation or lega... | | | | |
| RAG_Q080 | cross-filing comparison | 0.753 | overall_score=0.753 (<0.77) | NFLX_10-K_2024-01-26#150;NFLX_10-K_2025-01-27#120;NFLX_10-K_2024-01-26#135;NF... | NFLX_10-K_2024-01-26#150;NFLX_10-K_2024-01-26#149;NFLX_10-K_2024-01-26#151;NF... | overall_score=0.753 (<0.77) | Review the retrieved chunks and the generated answer. |
| | Question | | Comparing Item 1A of Netflix's 10-K filings from 2024-01-26 and 2025-01-27, what changed or stayed important regarding customer demand or sales channels? | | | | |

## Lowest-Scoring Passed Cases

| ID | Type | Overall score | Failed/weak metrics |
|---|---|---:|---|
| RAG_Q081 | cross-filing comparison | 0.771 |  |
| RAG_Q062 | transcript QA | 0.772 |  |
| RAG_Q058 | transcript QA | 0.780 |  |
| RAG_Q032 | section-specific | 0.782 |  |
| RAG_Q076 | cross-filing comparison | 0.785 |  |
| RAG_Q025 | section-specific | 0.789 |  |
| RAG_Q030 | section-specific | 0.794 |  |
| RAG_Q031 | section-specific | 0.800 |  |
| RAG_Q035 | section-specific | 0.800 |  |
| RAG_Q014 | single-filing factual | 0.801 |  |

## Index Reuse Information

| Field | Value |
|---|---:|
| Index path | D:\UCD\CSNL\Summer\LLM-Augmented-Financial-Report-Analysis-Tool\data\index |
| Index backend | chroma |
| Index status | existing index loaded |
| Manifest version | 1 |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Embedding dimension | 384 |
| Chunk configuration | 900 / 150 |
| Existing index reused | Yes |
| Rebuild requested | No |
| Rebuild performed | No |
| Rebuild reason |  |
| Index load time | 16.236 s |
| Evaluation runtime | 275.173 s |
| Restored read-only mtime touches | chroma.sqlite3 |
| Unrestored index timestamp changes |  |

## Reproduction Commands

PowerShell:

```powershell
python -m pip install -r requirements.txt
python -c "import sys; sys.path.insert(0, 'src'); from finsight.llm import api_key_status; print(api_key_status('openai', 'OPENAI_API_KEY').message)"
pytest -q
python src\finsight\rag_eval.py
python src\finsight\rag_eval.py --min-pass-rate 0.85
python src\finsight\rag_eval.py --with-llm
python src\finsight\rag_eval.py --rebuild-index
code eval\rag_eval_report.md
```

Bash:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -c "from finsight.llm import api_key_status; print(api_key_status('openai', 'OPENAI_API_KEY').message)"
pytest -q
python src/finsight/rag_eval.py
python src/finsight/rag_eval.py --min-pass-rate 0.85
python src/finsight/rag_eval.py --with-llm
python src/finsight/rag_eval.py --rebuild-index
${EDITOR:-vi} eval/rag_eval_report.md
```

# B2a-0 CPU context-length audit

Status: **BLOCKED_B2A_LONG_DOCUMENT_HANDLING_REQUIRED**

Pairs: **431200** (5600 queries × 77 documents)
Tokenizer: `5.14.1`, revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`
Prefix/suffix tokens: 39 / 9

All lengths are exact tokenizer counts for the official Qwen prefix + formatted `<Instruct>/<Query>/<Document>` content + suffix. No labels, Fold0, or CUDA were used.

## Distribution

| statistic | tokens |
|---|---:|
| minimum | 84 |
| p50 | 18455.000 |
| p90 | 64539.000 |
| p95 | 95289.000 |
| p99 | 158503.000 |
| p99_5 | 198726.000 |
| p99_9 | 362369.000 |
| maximum | 2145502 |

Selected max_length: **None**
Truncated pairs: **not selected (p99 > 32768)**

## Per fold

| fold | pairs | p99 | max |
|---|---:|---:|---:|
| F1 | 107800 | 158499.000 | 2145494 |
| F2 | 107800 | 158573.200 | 2145502 |
| F3 | 107800 | 158499.000 | 2145495 |
| F4 | 107800 | 158512.000 | 2145490 |

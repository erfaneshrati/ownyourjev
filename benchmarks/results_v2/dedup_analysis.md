# Duplicate-exclusion robustness analysis

Robustness analysis (not a model change): re-score each task's frozen predictions with evaluation inputs
that exactly duplicate (after normalisation) any text the experiment could see -- seeds, dev and the training
pool (for cross-fit tasks: seeds and dev only, since the pool *is* the unlabeled eval set) -- removed.
Jev is re-scored on the same subset. Reads test labels, so it is logged to TEST_ACCESS.log as 'analysis'.

| task | duplicated eval inputs removed | ours | Jev | ours − Jev |
|---|---|---|---|---|
| sms_spam | 82 / 5354 | 0.9574 | 0.9356 | +0.0218 |
| paws | 0 / 8000 | 0.8845 | 0.8499 | +0.0346 |
| boolq | 1 / 3270 | 0.8553 | 0.9131 | -0.0578 |
| prompt_injections | 0 / 116 | 0.9569 | 0.7414 | +0.2155 |
| toxic_chat | 307 / 5083 | 0.7754 | 0.7903 | -0.0150 |
| banking77 | 25 / 3076 | 0.8597 | 0.7958 | +0.0639 |
| clinc150 | 19 / 5500 | 0.9340 | 0.8947 | +0.0392 |
| ag_news | 15 / 7600 | 0.8931 | 0.8852 | +0.0079 |
| financial_phrasebank | 4 / 970 | 0.7950 | 0.7319 | +0.0631 |
| emotion | 11 / 2000 | 0.8145 | 0.5867 | +0.2278 |
| stsb | 7 / 1379 | 0.8929 | 0.8898 | +0.0031 |
| sst5 | 2 / 2210 | 0.5833 | 0.5793 | +0.0041 |
| toxigen | 177 / 940 | 0.8453 | 0.8794 | -0.0341 |
| summeval | 0 / 1376 | 0.4716 | 0.5591 | -0.0875 |
| helpsteer2 | 3 / 1038 | 0.3114 | 0.4096 | -0.0982 |

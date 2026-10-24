# MorphQC

Spot-level image–RNA consistency QC for spatial transcriptomics (HEST-1k).

## Bedrock: clean group split

`bedrock/` builds a leakage-safe split of the 1038 eligible HEST slides from metadata only.

- **Groups.** Slides are merged into one group when they share a study, a duplicate-section / shared-H&E link, a patient within a study, or (10x Genomics demo data) species and organ. A group never crosses sets.
- **OmiCLIP overlap.** Decided per slide. A slide seen in OmiCLIP pretraining, or a close relative of one (same duplicate / H&E link, same patient, same 10x series, or same study + species + organ), is never scored but still trains in other folds. Groups with no scorable slide are train-only.
- **Final holdout.** About 20% of the evaluable groups, stratified by species × platform, opened once.
- **Cross-validation.** The other evaluable groups get 5 stratified group folds, and each fold has a 10% checkpoint-selection subset.

```bash
python bedrock/build_split.py --split data/inputs/split_v3.2_2026-10-19.csv --meta HEST_v1_3_0.csv --out data/splits/bedrock_split_2026-10-24.csv
python bedrock/check_split.py data/splits/bedrock_split_2026-10-24.csv
python bedrock/one_organ_split.py --split data/splits/bedrock_split_2026-10-24.csv --out data/splits/bedrock_one_organ_mouse_brain_visium_2026-10-24.csv
```

`HEST_v1_3_0.csv` is the public HEST metadata table.

| File | md5 |
|---|---|
| `data/splits/bedrock_split_2026-10-24.csv` | `b6397176f38f7946e9649f8d896f68f4` |
| `data/splits/bedrock_one_organ_mouse_brain_visium_2026-10-24.csv` | `74b9ab3835445229d049c0366684d217` |

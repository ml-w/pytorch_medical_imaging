## File structure

```
.
├── asset/
│   ├── pmi_config/
│   │   ├── BM_nyul_v2.ini
│   │   └── NPC_seg.ini
│   ├── trained_states/
│   │   ├── deeplearning/
│   │   │   ├── segmen_checkpoint.pt
│   │   │   └── dl_diag_checkpoint.pt
│   │   └── radiomics/
│   │       └── (Nil)
│   ├── Logo.jpg
│   ├── t2w_normalization.yaml
│   ├── v1_seg_transform.yaml
│   └── v1_swran_transform.yaml
├── example_data/
│   ├── benign_case
│   ├── doubtful_case
│   ├── npc_case
│   └── ...
└── npc_report_gen/
    └── (Code directory)
```

## Output file structure



## Possible outputs

### NPC

* DL score >= threshold
* Radiomics score >= threshold
* Total volume >= 0.5 cm^3 

### Benign hyperplasia or normal

* DL score < threshold
* Radiomics score < threshold
* Total volume >= 3 cm ^3

### Normal

* DL score < threshold (Radiomics skipped because no lesion)
* Total volume < 0.5 cm^3

### Indeterminate

* Otherwise



## Note

* There should be no 'space bar' in all the directories
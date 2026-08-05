# FIIS — Fisher Information Imbalance Signature

**Beyond Class Imbalance: Fisher Information Diagnostic for Imbalance Remedy Selection**  
[Mabrouka Salmi et al. 2026]
*National High School of Statistics and Applied Economics, Algeria · University of Córdoba, Spain*

---

## Overview

This repository contains the code and supplementary materials for the FIIS framework — a principled diagnostic tool that reframes class imbalance as an information geometry problem. Rather than relying on the imbalance ratio (IR) alone, FIIS decomposes the Fisher information ratio into four interpretable components:

$$R = R_n \cdot R_p \cdot R_g \cdot C$$

separating **sampling deficit** ($R_n$), **boundary-proximity asymmetry** ($R_p$), **feature-energy asymmetry** ($R_g$), and **coupling correction** ($C$). From this decomposition, the framework derives a compensation theorem, a two-layer diagnostic (archetype + compensation regime), and hypothesis-driven remedy selection.

---

## 📄 Supplementary Materials

Full experimental results, per-dataset FIIS characterisations, per-classifier analyses, pairwise statistical tests, and the Vertebral Column case study are available at:

**[https://bouka12.github.io/FIIS/](https://bouka12.github.io/FIIS/)**

---

## Repository Structure

```
FIIS/
├── docs/                        # GitHub Pages (supplementary materials webpage)
│   ├── index.html
│   ├── assets/                  # Figures
│   └── data/                    # Result CSV files
├── src/
│   ├── real_data_pipeline.py    # Main experimental pipeline
│   ├── archetype_discovery.py   # Archetype assignment
│   ├── remedy_effectiveness_revised.py
│   ├── wilcoxon_analysis.py
│   ├── dgp_s.py                 # DGP-S synthetic validation
│   ├── dgp_g.py                 # DGP-G synthetic validation
│   ├── dgp_k.py                 # DGP-K compensation validation
│   ├── dgp_utils.py
│   ├── vertebral_pipeline.py    # Case study pipeline
│   └── vertebral_diagnostic.py  # Case study diagnostic
└── README.md
```

---

## Key Results

| | Result |
|---|---|
| Datasets | 94 (KEEL + imblearn, IR 1.82–129.53) |
| Median R / R_n gap | **6.6×** |
| Archetype A (pure sampling deficit) | 57 datasets (60.6%) |
| Archetype C (geometric deficit) | 19 datasets (20.2%) |
| H1 (Archetype A + Deficiency) | ✓ Confirmed — 14/15 Wilcoxon significant |
| H2 (Archetype C + Deficiency) | ~ Partially confirmed |
| H3 (Balance / Overbalance) | ✓ Confirmed — 0/15 Wilcoxon significant |

---

## Requirements

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib scipy
```

---

## Citation

```bibtex
@article{salmi2026fiis,
  title   = {Beyond Class Imbalance: Fisher Information Diagnostic
             for Imbalance Remedy Selection},
  author  = {Salmi, Mabrouka and Atif, Dalia and Ventura, Sebastian},
  journal = {},
  year    = {2026}
}
```

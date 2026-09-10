# FIIS — Fisher Information Imbalance Signature

**Beyond Class Imbalance: Fisher Information Diagnostic for Imbalance Remedy Selection**

**Salmi et al. (2026)**
National High School of Statistics and Applied Economics, Algeria · University of Córdoba, Spain

## Overview

This repository contains code and supplementary results for the Fisher Information Imbalance Signature (FIIS), a logistic-model-based, trace-based diagnostic of the relative information contributions of the two classes in binary classification.

FIIS decomposes the class information ratio as

$$
R = R_n \cdot R_p \cdot R_g \cdot R_c,
$$

where:

* **$R_n$** is the minority-to-majority sample-count ratio;
* **$R_p$** describes boundary-proximity asymmetry relative to the fitted logistic model;
* **$R_g$** describes feature-energy asymmetry in the preprocessed representation;
* **$R_c$** captures coupling between boundary proximity and feature energy.

The decomposition provides algebraic compensation conditions and two diagnostic dimensions: empirical archetype and compensation regime. Their association with remedy effectiveness is evaluated experimentally to support **diagnosis-informed remedy guidance**.

FIIS depends on the diagnostic model and feature representation. Its empirical archetypes are not an exhaustive taxonomy, and the experiments do not establish a universal or held-out remedy-selection policy. Some source files and figures use `C` for the coupling component denoted $R_c$ here.

## Supplementary materials

The companion webpage provides benchmark diagnostics, per-classifier comparisons, downloadable results and the illustrative Vertebral Column case study:

**[FIIS supplementary materials](https://bouka12.github.io/FIIS/)**

The paper focuses on G-mean; classification-result CSVs retain additional performance metrics. Synthetic validation code is available under `synthetic_validation/`.

## Repository structure

| Path                    | Contents                                                                                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `fiis_core.py`          | Core FIIS calculations and supporting functions                                                                                        |
| `real_data_validation/` | Benchmark pipelines, characterisation, archetype discovery, remedy-effectiveness analysis, Wilcoxon tests and the Vertebral case study |
| `synthetic_validation/` | DGP-S, DGP-G and DGP-K validation code and plotting scripts                                                                            |
| `docs/index.html`       | Companion webpage                                                                                                                      |
| `docs/assets/`          | Supplementary figures                                                                                                                  |
| `docs/data/`            | Downloadable result CSVs                                                                                                               |

Principal scripts include `real_data_pipeline_cpu.py`, `archetype_discovery.py`, `remedy_effectiveness.py`, `extra_remedy_effectiveness_per_classifier.py`, `wilcoxon_analysis.py`, `wilcoxon_analysis_per_classifier.py`, `vertebral_pipeline.py` and `vertebral_diagnostic.py` under `real_data_validation/`; and `dgp_s.py`, `dgp_g.py` and `dgp_k.py` under `synthetic_validation/`.

## Experimental scope and corrected results

| Item                                          | Summary                                                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Benchmark tasks                               | 94: 67 KEEL and 27 imbalanced-learn                                                              |
| Imbalance ratio                               | Majority/minority count; range 1.82–129.53                                                       |
| Evaluated classifiers in the reported results | Logistic regression, random forest and XGBoost                                                   |
| Remedies                                      | No correction, threshold correction, ROS, SMOTE, Borderline SMOTE and class weighting            |
| Benchmark evaluation                          | Six repetitions of five-fold cross-validation: 30 evaluations per dataset, classifier and remedy |
| Median $R/R_n$                                | 6.6, calculated from the dataset-level component means                                           |
| Consistently A                                | 55 datasets (58.5%)                                                                              |
| Consistently C                                | 19 datasets (20.2%)                                                                              |
| Mixed AC                                      | 14 datasets (14.9%)                                                                              |
| Mixed CA                                      | 6 datasets (6.4%)                                                                                |

“Consistently” means the same archetype in every evaluation. Mixed labels are ordered by frequency; archetype ties are resolved in favour of C. The resulting dominant labels are A for 69 datasets and C for 25 datasets. These are empirical diagnostic labels; A is interpreted as sampling-dominated and C as a feature-energy-deficit regime.

### Hypothesis groups and compensation labels

The supplied remedy-effectiveness results use these dataset-level groups:

* **H1:** dominant archetype A and mean $R<1$ — **58 datasets**.
* **H2:** dominant archetype C and mean $R<1$ — **25 datasets**.
* **H3:** mean $R\geq1$ — **11 datasets**.

These groups must be distinguished from evaluation-level compensation labels. With the operational tolerance $\varepsilon=0.05$, those labels are deficiency ($R<0.95$), balance ($0.95\leq R\leq1.05$) and overbalance ($R>1.05$). A modal compensation label and a group based on mean $R$ can differ.

### Statistical comparisons

Cross-dataset tests use **one mean G-mean value per dataset and remedy**; folds and repetitions are not treated as independent datasets. The aggregate analysis averages over classifiers, while per-classifier analysis retains separate classifier results.

Friedman tests provide omnibus comparisons, Nemenyi comparisons use average ranks, and paired two-sided Wilcoxon tests compare remedies. Bonferroni adjustment applies to the 15 Wilcoxon comparisons **within each classifier–hypothesis group**.

| Classifier          |    H1 |    H2 |   H3 |
| ------------------- | ----: | ----: | ---: |
| Logistic regression |  6/15 |  7/15 | 0/15 |
| Random forest       | 14/15 | 12/15 | 0/15 |
| XGBoost             | 12/15 |  9/15 | 0/15 |

Entries are the numbers of Bonferroni-significant Wilcoxon comparisons. Arithmetic mean G-mean differences provide descriptive magnitudes and directions. No detected difference does not establish equivalence. H2 is only partially supported; interpretation of H3 should consider the small group and possible ceiling effects. Some benchmark tasks share source datasets, which limits independence and generalisation.

For statistical background, see [Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html) and the [SciPy Friedman test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).

## Requirements

Core dependencies:

```bash
python -m pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib scipy
```

This installs dependencies without pinning versions. Exact reproduction also requires the software versions, random seeds, model settings and preprocessing used for the reported experiments. Check the relevant script's input and output paths before execution.



## Citation

Manuscript citation:

```bibtex
@unpublished{salmi2026fiis,
  title  = {Beyond Class Imbalance: Fisher Information Diagnostic
            for Imbalance Remedy Selection},
  author = {Salmi, Mabrouka and Atif, Dalia and Ventura, Sebastian},
  year   = {2026}
}
```

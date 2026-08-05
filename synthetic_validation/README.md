# **Synthetic Validation**
***Synthetic data is generated to validate the interpretation layer of the Fisher Information based Imbalance Diagnosis, mimick each archetype to validate it and test the remedy selection hypothesis***

**Structure of code of in "FIIS/synthetic_validation/"**

FIIS/
├── fiis_core.py                    ✓ done
│
└── synthetic_validation/
    ├── dgp_utils.py                ← shared generation helpers
    ├── dgp_s.py                    ← DGP-S
    ├── dgp_b.py                    ← DGP-B
    ├── dgp_g.py                    ← DGP-G
    ├── dgp_c.py                    ← DGP-C
    ├── dgp_k.py                    ← DGP-K
    ├── run_all.py                  ← runs all DGPs
    ├── figures.py                  ← figures for all results
    └── results/                    ← output CSVs

## Data Generation Processes
***Data generation protocol is shared, but changes are made to parameters to make the data satisfy the archetype conditions. As such, for each archetype A-D and F we explain its details separately (parameter grids).***

### DGP-S
### DGP-B
### DGP-G
### DGP-C
### DGP-K

## Component Validation
this is both examine the interpreation layer, and how for each archetype changes within parrameters changes the related dominance criterion maintaining same diagnosis. answering Is change in data change the corresponding value in the FIIS (Fisher Information Imbalance Signature)?
for each factor we notice its variation in terms of variation of related parameters to validate the interpretation of the components of the FIIS  vector (R_n, R_p, R_g, C), or its logarithm (log R_n, log R_p, log R_g, log C)

## Hypothesis Validation
***Remedy selection hypotheses are tested, where each archetype (A-D and F) has some hypothesis regarding class imbalance correction. Thus, some class imbalance corrections are tested in each synthetic data***

## Sensitivity Analysis 
***This is to check if changes in dimensionality, data size, etc affect the interpretation of the components and the hypothesis validation.***
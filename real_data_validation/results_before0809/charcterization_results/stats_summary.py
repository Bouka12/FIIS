"""
this script used to get some necessary statistics for the results section, particularly, `dataset characterization`
"""
import pandas as pd
import numpy as np

log_fiis_df = pd.read_csv(r"real_data_validation\results\charcterization_results\table_log_fiis_components.csv")
print(log_fiis_df.head())

# median of the decoupling D = log(R) - log(R_n)
log_fiis_df['decoupling_R_Rn'] = log_fiis_df['log_fiis_R'] - log_fiis_df['log_fiis_R_n']
print(f" median of the decoupling between R and R_n is {np.median(log_fiis_df['decoupling_R_Rn'].values)}")
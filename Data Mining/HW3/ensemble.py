import pandas as pd
# First file
FILE_A = "submission_combined.csv" 
# Second file
FILE_B = "submission_loadembed_normalized.csv" 
# Third file
FILE_C = "submission_feature_engineering.csv"

output_filename = "submission_3way_ensemble.csv"

df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)
df_c = pd.read_csv(FILE_C)

# Verify IDs match exactly for ALL files (Safety check)
assert df_a['id'].equals(df_b['id']) and df_a['id'].equals(df_c['id']), "Error: IDs do not match!"

# Create Ensemble
df_ensemble = df_a.copy()
pred_cols = [f'p{i}' for i in range(1, 16)]

print("Calculating weighted average...")
for col in pred_cols:
    # 3-Way Weighted Average (equals)
    df_ensemble[col] = (df_a[col] + df_b[col]  + df_c[col]) / 3

# Save
df_ensemble.to_csv(output_filename, index=False)
print(f"Success! Saved ensemble to {output_filename}")
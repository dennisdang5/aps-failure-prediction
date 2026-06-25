import pandas as pd
import json
import sys
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_THRESHOLD = 0.5

# Load reference and current data
X_train = pd.read_parquet('data/processed/X_train.parquet')
X_test = pd.read_parquet('data/processed/X_test.parquet')

# Create drift report
report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=X_train, current_data=X_test)

# Extract drift data and create an alert
result_dict = result.dict()
drift_metric = result_dict['metrics'][0]['value']
drifted_count = drift_metric['count']
drifted_share = drift_metric['share']
drifted_detected = drifted_share > DRIFT_THRESHOLD

summary = {
    'drifted_columns': int(drifted_count),
    'drift_share': drifted_share,
    'dataset_drift_detected': drifted_detected
}

# Save as HTML
result.save_html('results/drift_report.html')

# Save as JSON
with open('results/drift_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Drift share: {drifted_share:.2%}')
print(f'Dataset drifted detected: {drifted_detected}')

if drifted_detected:
    print('WARNING: Significant drift detected. Consider retraining.')
    sys.exit(1)

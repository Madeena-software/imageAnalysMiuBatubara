#!/usr/bin/env python3
"""Generate a sample latest PDF at public/latest_circle_report.pdf for browser testing."""
import sys, os, base64
sys.path.insert(0, os.path.join(os.getcwd(), 'public', 'image-analysis-miu-batubara'))
from pdf_exporter import generate_circle_detection_pdf

result = {
    'count': 1,
    'circles': [
        {'grid_pos': [1,1], 'center': [120, 80], 'diameter': 12.3, 'mean_value': 1234.5, 'classification': 'Coal'}
    ]
}
params = {'threshold_value': 100, 'min_diameter': 5, 'max_diameter': 50, 'min_circularity': 0.8, 'min_solidity': 0.9, 'expected_count': 1}

diagonal_result = {'summary': {
    'upper_mu_avg': 0.0025,
    'lower_mu_avg': 0.0021,
    'upper_mu_std': 0.0001,
    'lower_mu_std': 0.00012,
    'lower_avg_mean': 1200.0,
    'lower_avg_median': 1195.0,
    'lower_std_means': 2.5,
    'upper_avg_mean': 1250.0,
    'upper_avg_median': 1248.0,
    'upper_std_means': 3.1,
    'mean_difference': 50.0
}}

out = generate_circle_detection_pdf('sample.tiff', result, params, diagonal_result=diagonal_result)

if 'pdf_base64' in out:
    pdf_bytes = base64.b64decode(out['pdf_base64'])
    out_path = os.path.join(os.getcwd(), 'public', 'latest_circle_report.pdf')
    with open(out_path, 'wb') as f:
        f.write(pdf_bytes)
    print('WROTE', out_path)
else:
    print('ERROR', out)

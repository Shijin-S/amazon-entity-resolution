# amazon-entity-resolution

Business entity resolution across 3 noisy data sources for competition record linkage.

## Setup Instructions

1. Create a virtual environment (Python 3.10+ required):
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline

- Raw TSV data ingestion
- Data preprocessing
- Name, address, and country normalization
- Candidate generation and blocking
- Feature extraction and similarity calculation
- Matching model training and validation
- Threshold optimization using F0.5 score
- Final prediction output (matching_results.tsv)

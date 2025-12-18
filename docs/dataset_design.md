# Dataset Design

Goals
- Realistic transaction logs with sender/receiver, timestamps, amounts, channels, countries, and labels for suspicious patterns.
- Target size 50k–200k transactions with 1–3% suspicious to mimic AML rarity.
- Use the same schema across train/val/test while preserving imbalance.

Files produced
- transaction.csv: tx_id, timestamp, sender_id, receiver_id, amount, currency, channel, country, tx_type, label, scenario_id, evidence.
- accounts.csv: account_id, account_type, created_at, initial_balance.
- edges.csv: sender_id, receiver_id, tx_count, total_amount (aggregated for graph models).
- train.csv, val.csv, test.csv: time-ordered splits of transaction.csv with suspicious examples present in each.

Scenario definitions
- Structuring: many small deposits (50–200) into one account within 24–72h that sum to a large value; label "structuring" and evidence describes the burst.
- Circular transactions: directed cycles of 3–6 accounts with amounts 5k–40k; label "circular" and evidence lists the path.
- Layering: chains of 10–16 hops across countries/channels within 6–36h; label "layering" and evidence contains the path.

Generation approach
- Create an account pool (individual/company/shell) with random created_at dates and initial balances.
- Simulate normal activity with log-normal amounts, diurnal timestamp patterns, and mixed channels/countries.
- Inject structuring, circular, and layering scenarios according to a sampled suspicious ratio between 1–3%.
- Shuffle, aggregate edges, then split by time (70/15/15). If val/test lack suspicious cases, move a few labeled records from train to keep coverage.

Usage
- Run scripts/run_generate.sh (or python scripts/generate_dataset.py) to write outputs into data/raw.

from __future__ import annotations

import argparse
import json

from .cleaning import clean_all
from .eda import build_eda_report
from .features import build_all_features
from .modeling import train_all_models
from .validation import validate_processed


def run_pipeline(skip_models: bool = False):
    clean_all(write_outputs=True)
    validation = validate_processed()
    if not validation["test_summary"]["all_passed"]:
        raise RuntimeError("Hard data validation checks failed. See reports/validation_report.json")
    build_all_features()
    eda = build_eda_report()
    models = None if skip_models else train_all_models()
    return {"validation": validation["test_summary"], "eda": str(eda), "models": models}


def main():
    parser = argparse.ArgumentParser(description="Run the full Zomato BI + ML pipeline")
    parser.add_argument("--skip-models", action="store_true", help="Run ingestion, cleaning, validation, features and EDA only")
    args = parser.parse_args()
    result = run_pipeline(skip_models=args.skip_models)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

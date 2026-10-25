"""Run the whole pipeline: data -> EDA figures -> model training/evaluation.

    python run_pipeline.py
"""
from src import data_generation, eda, train

if __name__ == "__main__":
    print("== 1/3 Building dataset ==")
    data_generation.main()
    print("\n== 2/3 Exploratory analysis ==")
    eda.main()
    print("\n== 3/3 Training model ==")
    train.main()

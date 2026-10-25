"""Exploratory data analysis: saves the figures to img/Sleeping-Quality-Prediction-Optimization/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from src.preprocessing import load_final_dataset

TARGET = config.TARGET
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(config.IMG_DIR / name, dpi=130)
    plt.close()


def quality_distribution(df):
    plt.figure(figsize=(8, 5))
    hue = df[config.SYNTHETIC_FLAG].map({0: "Real", 1: "Synthetic"})
    ax = sns.countplot(x=TARGET, hue=hue, data=df, palette=["#ff7a59", "#6c63ff"])
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
    plt.title("Distribution of Sleep Quality Levels")
    plt.legend(title="")
    save("eda_quality_distribution.png")


def correlation_heatmap(df):
    corr = df[config.NUMERIC_FEATURES + [TARGET]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save("eda_correlation_heatmap.png")


def boxplots(df):
    cols = ["Sleep Duration", "Stress Level", "Physical Activity Level", "Caffeine Intake", "Screen Time", "Heart Rate"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(x=TARGET, y=col, data=df, hue=TARGET, legend=False, palette="viridis", ax=ax)
        ax.set_title(f"{col} vs Sleep Quality")
    save("eda_boxplots.png")


def disorder_vs_quality(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Sleep Disorder", hue=TARGET, data=df, palette="viridis")
    plt.title("Sleep Disorder vs Quality of Sleep")
    save("eda_sleep_disorder.png")


def mean_features(df):
    cols = ["Sleep Duration", "Stress Level", "Physical Activity Level", "Caffeine Intake",
            "Daily Work Hours", "Screen Time", "Heart Rate"]
    means = df.groupby(TARGET)[cols].mean().reset_index().melt(id_vars=TARGET, var_name="Feature", value_name="Average")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=means, x="Feature", y="Average", hue=TARGET, palette="viridis")
    plt.xticks(rotation=40, ha="right")
    plt.title("Average Feature Values per Sleep Quality Level")
    save("eda_mean_features.png")


def real_vs_synthetic(df):
    """Why the data source matters: the same feature behaves differently in real and synthetic rows."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["Sleep Duration", "Caffeine Intake"]):
        sns.scatterplot(data=df, x=col, y=TARGET, hue=df[config.SYNTHETIC_FLAG].map({0: "Real", 1: "Synthetic"}),
                        palette=["#ff7a59", "#6c63ff"], alpha=0.35, s=18, ax=ax)
        ax.set_title(f"{col} vs quality: real vs synthetic")
    save("eda_real_vs_synthetic.png")


def main():
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_final_dataset()
    print("Shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))
    print(df.describe().round(2).T.to_string())
    for fn in (quality_distribution, correlation_heatmap, boxplots, disorder_vs_quality, mean_features, real_vs_synthetic):
        fn(df)
    print("Saved EDA figures to", config.IMG_DIR)


if __name__ == "__main__":
    main()

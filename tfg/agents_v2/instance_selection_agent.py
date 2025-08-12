# instance_selection_agent_structured_multi_tools.py
from __future__ import annotations
from typing import Optional, Any, Dict, Type, List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

from dotenv import load_dotenv
import os

# =====================================================================
# LLM Setup
# =====================================================================
load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    custom_llm_provider="gemini",
)

# =====================================================================
# Helpers
# =====================================================================

def _get_df(tool: BaseTool) -> pd.DataFrame:
    df = getattr(tool, "dataset", None)
    if df is None:
        raise ValueError("No dataset assigned to tool. Set `tool.dataset = your_dataframe` before running.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`dataset` must be a pandas DataFrame.")
    return df


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    X_num = df.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        raise ValueError("No numeric columns available for this operation.")
    return X_num


# =====================================================================
# Core Sampling Tools
# =====================================================================

class StratifiedSampleInput(BaseModel):
    target: str = Field(..., description="Target column name for stratification")
    sample_size: float = Field(0.5, gt=0.0, le=1.0, description="Proportion (0,1] of rows to keep")
    random_state: Optional[int] = Field(42, description="Random seed")


class StratifiedSampleTool(BaseTool):
    name: str = "stratified_sample"
    description: str = (
        "Subsample the dataset using stratified sampling on the target. "
        "Requires: target. Optional: sample_size, random_state. Saves to pipeline_data/dataset.csv"
    )
    args_schema: Type[BaseModel] = StratifiedSampleInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, target: str, sample_size: float = 0.5, random_state: Optional[int] = 42) -> str:
        try:
            from sklearn.model_selection import train_test_split
            from collections import Counter

            df = _get_df(self)
            if target not in df.columns:
                return f"Target '{target}' not found. Available: {list(df.columns)}"

            X = df.drop(columns=[target])
            y = df[target]

            class_counts = y.value_counts(dropna=False).to_dict()
            if any(cnt < 2 for cnt in class_counts.values()):
                return (
                    "Cannot apply stratified sampling: at least one class has < 2 samples. "
                    "Use 'random_sample' or 'class_balanced_sample' instead."
                )

            X_res, _, y_res, _ = train_test_split(
                X, y, train_size=sample_size, random_state=random_state, stratify=y
            )
            reduced_df = pd.concat([X_res, y_res], axis=1)

            os.makedirs("pipeline_data", exist_ok=True)
            final_path = "pipeline_data/dataset.csv"
            reduced_df.to_csv(final_path, index=False)

            return (
                "Stratified sampling applied.\n"
                f"Original size: {df.shape[0]} → Reduced size: {reduced_df.shape[0]}\n"
                f"Class distribution (original): {class_counts}\n"
                f"Saved to: {final_path}"
            )
        except Exception as e:
            return f"StratifiedSampleTool failed: {type(e).__name__}: {e}"


class RandomSampleInput(BaseModel):
    sample_size: float = Field(0.5, gt=0.0, le=1.0, description="Proportion (0,1] of rows to keep")
    random_state: Optional[int] = Field(42, description="Random seed")


class RandomSampleTool(BaseTool):
    name: str = "random_sample"
    description: str = (
        "Subsample the dataset using random (non-stratified) sampling. "
        "Optional: sample_size, random_state. Saves to pipeline_data/dataset.csv"
    )
    args_schema: Type[BaseModel] = RandomSampleInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, sample_size: float = 0.5, random_state: Optional[int] = 42) -> str:
        try:
            from sklearn.model_selection import train_test_split

            df = _get_df(self)
            idx = np.arange(len(df))
            train_idx, _ = train_test_split(idx, train_size=sample_size, random_state=random_state)
            reduced_df = df.iloc[train_idx].copy()

            os.makedirs("pipeline_data", exist_ok=True)
            final_path = "pipeline_data/dataset.csv"
            reduced_df.to_csv(final_path, index=False)

            return (
                "Random sampling applied.\n"
                f"Original size: {df.shape[0]} → Reduced size: {reduced_df.shape[0]}\n"
                f"Saved to: {final_path}"
            )
        except Exception as e:
            return f"RandomSampleTool failed: {type(e).__name__}: {e}"


class ClassBalancedSampleInput(BaseModel):
    target: str = Field(..., description="Target column name")
    per_class: Optional[int] = Field(None, gt=0, description="Number of samples to draw per class (if set)")
    max_total: Optional[int] = Field(None, gt=0, description="Overall cap; per-class quota is derived as floor(max_total / n_classes)")
    random_state: Optional[int] = Field(42, description="Random seed")
    shuffle_within_class: bool = Field(True, description="Shuffle rows within each class before selecting")


class ClassBalancedSampleTool(BaseTool):
    name: str = "class_balanced_sample"
    description: str = (
        "Build a class-balanced subset by sampling up to 'per_class' rows from each class. "
        "Alternatively set 'max_total' to derive per-class quota. Saves to pipeline_data/dataset.csv"
    )
    args_schema: Type[BaseModel] = ClassBalancedSampleInput
    dataset: Type[pd.DataFrame] = None

    def _run(
        self,
        target: str,
        per_class: Optional[int] = None,
        max_total: Optional[int] = None,
        random_state: Optional[int] = 42,
        shuffle_within_class: bool = True,
    ) -> str:
        try:
            rng = np.random.default_rng(random_state)
            df = _get_df(self)
            if target not in df.columns:
                return f"Target '{target}' not found. Available: {list(df.columns)}"

            n_classes = df[target].nunique(dropna=False)
            if per_class is None and max_total is None:
                return "Provide either 'per_class' or 'max_total'."
            if per_class is None and max_total is not None:
                per_class = max(1, max_total // max(1, n_classes))

            parts = []
            class_counts = {}
            for cls, sub in df.groupby(target, dropna=False):
                idx = sub.index.to_numpy()
                if shuffle_within_class:
                    rng.shuffle(idx)
                take = min(per_class, len(idx))
                chosen = sub.loc[idx[:take]]
                parts.append(chosen)
                class_counts[cls] = int(take)

            reduced_df = pd.concat(parts, axis=0).reset_index(drop=True)
            os.makedirs("pipeline_data", exist_ok=True)
            final_path = "pipeline_data/dataset.csv"
            reduced_df.to_csv(final_path, index=False)

            return (
                "Class-balanced sampling applied.\n"
                f"Per-class quota: {per_class}\n"
                f"Class sample counts: {class_counts}\n"
                f"Total: {reduced_df.shape[0]} rows. Saved to: {final_path}"
            )
        except Exception as e:
            return f"ClassBalancedSampleTool failed: {type(e).__name__}: {e}"


class ClusteredSampleInput(BaseModel):
    n_clusters: int = Field(..., ge=1, description="Number of clusters to form")
    samples_per_cluster: int = Field(1, ge=1, description="Number of representatives to select per cluster")
    random_state: Optional[int] = Field(42, description="Random seed for KMeans")
    use_features: Optional[List[str]] = Field(None, description="Optional subset of feature columns to use; numeric enforced")


class ClusteredSampleTool(BaseTool):
    name: str = "clustered_sample"
    description: str = (
        "Diversity-based sampling via KMeans on numeric features; selects the closest points to centroids "
        "(or multiple per cluster). Saves to pipeline_data/dataset.csv"
    )
    args_schema: Type[BaseModel] = ClusteredSampleInput
    dataset: Type[pd.DataFrame] = None

    def _run(
        self,
        n_clusters: int,
        samples_per_cluster: int = 1,
        random_state: Optional[int] = 42,
        use_features: Optional[List[str]] = None,
    ) -> str:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances_argmin_min

            df = _get_df(self)
            X = df if use_features is None else df[use_features]
            X_num = _numeric_df(X).to_numpy()

            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
            labels = km.fit_predict(X_num)

            chosen_idx: List[int] = []
            for cl in range(n_clusters):
                cluster_idx = np.where(labels == cl)[0]
                if cluster_idx.size == 0:
                    continue
                # Choose representatives closest to centroid
                cent = km.cluster_centers_[cl].reshape(1, -1)
                _, nn_idx = pairwise_distances_argmin_min(X_num[cluster_idx], cent)
                # First representative
                sel = cluster_idx[nn_idx[0]]
                chosen_idx.append(sel)
                # If more per cluster required, sample additional nearest (without replacement)
                if samples_per_cluster > 1:
                    # Distances to centroid for ordering
                    dists = np.linalg.norm(X_num[cluster_idx] - km.cluster_centers_[cl], axis=1)
                    order = np.argsort(dists)
                    extra = [cluster_idx[i] for i in order if cluster_idx[i] != sel][: max(0, samples_per_cluster - 1)]
                    chosen_idx.extend(extra)

            reduced_df = df.iloc[sorted(set(chosen_idx))].copy()
            os.makedirs("pipeline_data", exist_ok=True)
            final_path = "pipeline_data/dataset.csv"
            reduced_df.to_csv(final_path, index=False)

            return (
                "Clustered sampling applied (KMeans).\n"
                f"Clusters: {n_clusters}, representatives/cluster: {samples_per_cluster}.\n"
                f"Total selected: {reduced_df.shape[0]} rows. Saved to: {final_path}"
            )
        except Exception as e:
            return f"ClusteredSampleTool failed: {type(e).__name__}: {e}"


# =====================================================================
# Splitting Tools (independent of sampling)
# =====================================================================

class TrainValTestSplitInput(BaseModel):
    target: Optional[str] = Field(None, description="Optional target for stratified splitting")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Proportion for test set")
    val_size: float = Field(0.1, ge=0.0, lt=1.0, description="Proportion for validation set (from remaining after test)")
    stratify: bool = Field(True, description="Whether to stratify using target if provided")
    random_state: Optional[int] = Field(42, description="Random seed")


class TrainValTestSplitTool(BaseTool):
    name: str = "train_val_test_split"
    description: str = (
        "Create reproducible train/val/test splits. If 'target' given and 'stratify' True, use stratified split. "
        "Saves to pipeline_data/train.csv, val.csv, test.csv"
    )
    args_schema: Type[BaseModel] = TrainValTestSplitInput
    dataset: Type[pd.DataFrame] = None

    def _run(
        self,
        target: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: Optional[int] = 42,
    ) -> str:
        try:
            from sklearn.model_selection import train_test_split

            df = _get_df(self)
            if target is not None and target not in df.columns:
                return f"Target '{target}' not found. Available: {list(df.columns)}"

            # First split: train+val vs test
            strat = df[target] if (target and stratify and target in df.columns) else None
            trainval_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=strat
            )

            # Second split: train vs val
            effective_val = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
            strat2 = trainval_df[target] if (target and stratify and target in df.columns) else None
            train_df, val_df = train_test_split(
                trainval_df, test_size=effective_val, random_state=random_state, stratify=strat2
            )

            os.makedirs("pipeline_data", exist_ok=True)
            train_path = "pipeline_data/train.csv"
            val_path = "pipeline_data/val.csv"
            test_path = "pipeline_data/test.csv"
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            return (
                "Train/Val/Test split created.\n"
                f"Sizes → train: {train_df.shape[0]}, val: {val_df.shape[0]}, test: {test_df.shape[0]}\n"
                f"Saved to: {train_path}, {val_path}, {test_path}"
            )
        except Exception as e:
            return f"TrainValTestSplitTool failed: {type(e).__name__}: {e}"


class TimeSeriesSplitInput(BaseModel):
    time_column: str = Field(..., description="Timestamp or date column name")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Proportion for test set (last fraction in time)")
    val_size: float = Field(0.1, ge=0.0, lt=1.0, description="Proportion for validation set (from remaining after test)")


class TimeSeriesSplitTool(BaseTool):
    name: str = "time_series_split"
    description: str = (
        "Chronological train/val/test split for time series (no leakage). "
        "Saves to pipeline_data/train.csv, val.csv, test.csv"
    )
    args_schema: Type[BaseModel] = TimeSeriesSplitInput
    dataset: Type[pd.DataFrame] = None

    def _run(self, time_column: str, test_size: float = 0.2, val_size: float = 0.1) -> str:
        try:
            df = _get_df(self)
            if time_column not in df.columns:
                return f"Time column '{time_column}' not found. Available: {list(df.columns)}"

            # Ensure datetime and sort
            tmp = df.copy()
            tmp[time_column] = pd.to_datetime(tmp[time_column], errors="coerce")
            tmp = tmp.sort_values(time_column)
            tmp = tmp[tmp[time_column].notna()]

            n = len(tmp)
            n_test = max(1, int(round(n * test_size)))
            n_trainval = n - n_test
            n_val = max(0, int(round(n_trainval * val_size)))
            n_train = n_trainval - n_val

            train_df = tmp.iloc[:n_train]
            val_df = tmp.iloc[n_train:n_train + n_val]
            test_df = tmp.iloc[n_train + n_val:]

            os.makedirs("pipeline_data", exist_ok=True)
            train_path = "pipeline_data/train.csv"
            val_path = "pipeline_data/val.csv"
            test_path = "pipeline_data/test.csv"
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            return (
                "Time-series split created (chronological).\n"
                f"Sizes → train: {train_df.shape[0]}, val: {val_df.shape[0]}, test: {test_df.shape[0]}\n"
                f"Saved to: {train_path}, {val_path}, {test_path}"
            )
        except Exception as e:
            return f"TimeSeriesSplitTool failed: {type(e).__name__}: {e}"


# =====================================================================
# Agent + Task wiring
# =====================================================================

def build_agent() -> Agent:
    return Agent(
        role="Instance Selection & Splitting Expert",
        goal=(
            "Understand the user's request and call only the necessary tool: stratified/random/class-balanced/clustered "
            "sampling or dataset splitting (train/val/test, time series). Ask a short question if required params are missing."
        ),
        backstory=(
            "You are a data sampling specialist who minimizes bias and preserves representativeness. "
            "You never run unnecessary tools and you always report dataset sizes and save locations."
        ),
        tools=[
            StratifiedSampleTool(),
            RandomSampleTool(),
            ClassBalancedSampleTool(),
            ClusteredSampleTool(),
            TrainValTestSplitTool(),
            TimeSeriesSplitTool(),
        ],
        verbose=True,
        llm=llm,
    )


def build_task(agent: Agent) -> Task:
    return Task(
        description=(
            "User request: {user_request}\n\n"
            "Use ONLY the necessary tool to answer. If a required parameter (e.g., target, time_column) is missing, "
            "ask one brief clarifying question and stop."
        ),
        expected_output=(
            "A concise answer stating: method applied, parameters used, size(s) obtained, and save location(s)."
        ),
        agent=agent,
    )

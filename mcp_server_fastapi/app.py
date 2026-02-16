"""
MCP Visualization Server – LLM-supervised, Tool-driven (Stable)
"""

import ast
import base64
import io
import json
import os
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic


# FASTAPI

app = FastAPI(title="MCP Visualization Server", version="6.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MODELS

class VisualizationRequest(BaseModel):
    prompt: str
    dataset_info: Dict[str, Any]
    output_format: str = "png"


# HELPERS

def get_llm() -> Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    return Anthropic(api_key=key)


def load_df(dataset_info: Dict[str, Any]) -> pd.DataFrame:
    if "sample_data" not in dataset_info:
        raise HTTPException(400, "dataset_info.sample_data missing")
    df = pd.DataFrame(dataset_info["sample_data"])
    if df.empty:
        raise HTTPException(400, "Dataset is empty")
    return df


def fig_to_base64(fig: plt.Figure, fmt: str = "png") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return out


def pretty(col: str) -> str:
    return col.replace("_", " ").title()


def normalize_column(col: str, df: pd.DataFrame) -> str:
    key = col.lower().replace(" ", "").replace("_", "")
    for c in df.columns:
        if c.lower().replace(" ", "").replace("_", "") == key:
            return c
    raise KeyError(f"Column not found: {col}")


def normalize_params(params: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        if isinstance(v, str):
            out[k] = normalize_column(v, df)
        else:
            out[k] = v
    return out


sns.set_theme(style="whitegrid", palette="Set2")

# Visualization tools

def dual_axes_chart(df: pd.DataFrame, x: str, y1: str, y2: str, fmt: str):
    agg = df.groupby(x, as_index=False)[[y1, y2]].mean()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(agg[x], agg[y1], marker="o", label=pretty(y1))
    ax1.set_xlabel(pretty(x))
    ax1.set_ylabel(pretty(y1))
    ax2 = ax1.twinx()
    ax2.plot(agg[x], agg[y2], marker="s", color="orange", label=pretty(y2))
    ax2.set_ylabel(pretty(y2))
    plt.title(f"{pretty(y1)} a {pretty(y2)} v čase ({pretty(x)})")
    fig.autofmt_xdate()
    return fig_to_base64(fig, fmt)


def violin_chart(df: pd.DataFrame, cat: str, num: str, fmt: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x=cat,
        y=num,
        ax=ax,
        inner="quartile",
        cut=0
    )
    ax.set_xlabel(pretty(cat))
    ax.set_ylabel(pretty(num))
    ax.set_title(f"Rozdělení {pretty(num)} podle {pretty(cat)}")
    return fig_to_base64(fig, fmt)



def heatmap_chart(df: pd.DataFrame, fmt: str):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        raise ValueError("Not enough numeric columns for heatmap")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Korelační heatmapa")
    return fig_to_base64(fig, fmt)


TOOL_REGISTRY = {
    "dual_axes": dual_axes_chart,
    "violin": violin_chart,
    "heatmap": heatmap_chart,
}

# LLM analysis prompt



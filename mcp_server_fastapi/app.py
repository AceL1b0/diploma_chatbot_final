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



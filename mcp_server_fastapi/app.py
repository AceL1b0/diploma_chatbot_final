"""
MCP Visualization Server
"""

import base64
import io
import os
import textwrap
from typing import Dict, Any, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic


# FastAPI

app = FastAPI(title="MCP Visualization Server", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models

class VisualizationRequest(BaseModel):
    prompt: str
    dataset_info: Dict[str, Any]
    output_format: str = "png"


# Helpers

def get_llm() -> Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    return Anthropic(api_key=key)


def get_model() -> str:
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL not set")
    return model


def load_df(dataset_info: Dict[str, Any]) -> pd.DataFrame:
    if "sample_data" not in dataset_info:
        raise HTTPException(400, "dataset_info.sample_data missing")
    df = pd.DataFrame(dataset_info["sample_data"])
    if df.empty:
        raise HTTPException(400, "Dataset is empty")
    return df


def fig_to_base64(fig: plt.Figure, fmt: str = "png") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=120)
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return out


def build_schema(df: pd.DataFrame) -> Dict:
    """Bohaté schéma datasetu pro LLM - kardinalita, typy, ukázky."""
    schema = {}
    for col in df.columns:
        if col.lower().startswith("unnamed"):
            continue
        nunique = int(df[col].nunique())
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(5).tolist()
        info = {"dtype": dtype, "nunique": nunique, "sample": sample}
        if pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(df[col].min())
            info["max"] = float(df[col].max())
            info["mean"] = round(float(df[col].mean()), 3)
        schema[col] = info
    return schema


# Tool schema (plan)

PLAN_TOOL = {
    "name": "create_dashboard_plan",
    "description": "Vytvoří plán dashboardu - insight a seznam 3-4 grafů s popisem co každý má ukázat.",
    "input_schema": {
        "type": "object",
        "properties": {
            "insight": {
                "type": "string",
                "description": "Hlavní datový insight v jedné větě"
            },
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Nadpis grafu"},
                        "description": {"type": "string", "description": "Co graf ukazuje a proč je zajímavý"},
                        "chart_type": {
                            "type": "string",
                            "enum": ["line", "bar", "scatter", "histogram", "violin", "dual_axes"]
                        },
                        "columns_used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Přesné názvy sloupců použité v grafu"
                        }
                    },
                    "required": ["title", "description", "chart_type", "columns_used"]
                },
                "minItems": 3,
                "maxItems": 4
            }
        },
        "required": ["insight", "charts"]
    }
}


# Prompts

PLAN_SYSTEM = """
Jsi zkušený datový analytik. Tvým úkolem je navrhnout dashboard s 3-4 grafy.

Pravidla pro výběr grafů:
- line: pouze pro datum/čas nebo pořadové hodnoty (nunique > 20)
- bar: pro kategorie s nunique 2-25, zobraz top hodnoty seřazené sestupně
- scatter: pro vztah dvou numerických sloupů, přidej regresní linii
- histogram: pro distribuci jednoho numerického sloupce, přidej průměr a medián
- violin: pro distribuci čísla podle kategorie (nunique kategorie < 15)
- dual_axes: pouze pokud chceš srovnat 2 metriky s velmi různými škálami

KRITICKÁ PRAVIDLA:
- Nepoužívej sloupce začínající "Unnamed"
- bar NIKDY pro sloupce s nunique > 25
- violin NIKDY pro kategorie s nunique > 15
- Každý graf musí přinést JINOU informaci
- Nepoužívej stejný typ grafu dvakrát
"""

CODE_SYSTEM = """
Jsi expert na Python vizualizace s matplotlib a seaborn.

Napiš Python kód pro JEDEN konkrétní graf.

Pravidla:
- DataFrame je dostupný jako proměnná `df` (již načtený)
- Figure je dostupný jako proměnná `fig` a `ax` (již vytvořený: fig, ax = plt.subplots(...))
- NEPIŠ: import, plt.subplots(), plt.show(), plt.savefig(), plt.close()
- Kresli pouze na `ax`
- Používej sns nebo ax přímé volání
- Přidej popisné osy a title
- Zpracuj data správně (agregace, filtrování, konverze typů)
- Pro datetime: pd.to_datetime() a resample("ME").mean()
- Pro bar s mnoha kategoriemi: zobraz jen top 15 podle hodnoty, horizontálně
- Pro scatter: přidej regresní linii přes sns.regplot(scatter=False)
- Pro histogram: přidej ax.axvline pro průměr a medián
- Kód musí být robustní: dropna(), pd.to_numeric(errors='coerce') kde je potřeba

Napiš POUZE spustitelný Python kód, bez vysvětlení, bez markdown.
"""


# Step 1: Plan (tool_use)

def create_plan(prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
    """LLM navrhne strukturovaný plán dashboardu přes tool_use."""
    llm = get_llm()
    schema = build_schema(df)

    user_msg = f"""
Požadavek: {prompt}

Schéma datasetu ({len(df)} řádků):
{schema}

Navrhni 3-4 různé grafy pro dashboard.
"""

    resp = llm.messages.create(
        model=get_model(),
        max_tokens=1000,
        system=PLAN_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "create_dashboard_plan"},
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == "create_dashboard_plan":
            return block.input  # již Python dict, bez json.loads()

    raise HTTPException(500, "LLM did not return tool_use block")


# Step 2: Code per chart

def generate_chart_code(chart: Dict[str, Any], df: pd.DataFrame) -> str:
    """LLM napíše matplotlib kód na míru pro jeden konkrétní graf."""
    llm = get_llm()
    schema = build_schema(df)

    # Ukázka dat pro relevantní sloupce
    cols = chart.get("columns_used", [])
    valid_cols = [c for c in cols if c in df.columns]
    sample_data = df[valid_cols].head(10).to_string() if valid_cols else df.head(5).to_string()

    user_msg = f"""
Graf: {chart['title']}
Typ: {chart['chart_type']}
Popis: {chart['description']}
Použité sloupce: {chart['columns_used']}

Schéma datasetu:
{schema}

Ukázka dat:
{sample_data}

Napiš Python kód pro tento graf. Kresli na proměnnou `ax`, data jsou v `df`.
"""

    resp = llm.messages.create(
        model=get_model(),
        max_tokens=800,
        system=CODE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    code = resp.content[0].text.strip()

    # Odstranění markdown pokud LLM přidá
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    return code


# Step 3: Execute code

def execute_chart_code(code: str, df: pd.DataFrame, fmt: str) -> str:
    """Spustí kód grafu a vrátí base64 obrázek."""
    sns.set_theme(style="whitegrid", palette="Set2")
    fig, ax = plt.subplots(figsize=(10, 6))

    exec_globals = {
        "df": df.copy(),
        "fig": fig,
        "ax": ax,
        "plt": plt,
        "pd": pd,
        "sns": sns,
        "np": np,
    }

    exec(textwrap.dedent(code), exec_globals)  # noqa: S102

    return fig_to_base64(fig, fmt)


# Endpoint

@app.post("/advanced-visualization")
def advanced_visualization(req: VisualizationRequest):
    df = load_df(req.dataset_info)
    fmt = req.output_format

    # Krok 1: strukturovaný plán přes tool_use
    plan = create_plan(req.prompt, df)
    print(f"Plan: insight='{plan.get('insight')}', charts={[c['title'] for c in plan.get('charts', [])]}")

    images = {}
    errors = []

    # Krok 2+3: pro každý graf LLM napíše kód
    for chart in plan.get("charts", [])[:4]:
        title = chart.get("title", "chart")
        print(f"Generating code for: {title} ({chart.get('chart_type')})")

        try:
            code = generate_chart_code(chart, df)
            print(f"Code for '{title}':\n{code}\n---")

            img = execute_chart_code(code, df, fmt)

            key = title.lower().replace(" ", "_")[:30]
            counter = 1
            while key in images:
                key = f"{key}_{counter}"
                counter += 1
            images[key] = img

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error for '{title}': {tb}")
            errors.append(f"{title}: {str(e)}")

    if not images:
        raise HTTPException(500, f"No visualizations generated. Errors: {errors}")

    return {
        "success": True,
        "insight": plan.get("insight"),
        "visualization": next(iter(images.values())),
        "visualizations": images,
        "chart_count": len(images),
        "tool_errors": errors,
        "llm_plan": plan,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )

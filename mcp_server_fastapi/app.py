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


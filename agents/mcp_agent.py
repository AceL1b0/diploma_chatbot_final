"""
MCP Agent (HTTP) – volá FastAPI MCP server (např. HF Spaces) pro pokročilé vizualizace.
"""
import base64
import os
import tempfile
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from agents.evaluation_agent import EvaluationAgent

load_dotenv()

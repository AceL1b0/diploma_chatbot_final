"""
Visualization Agent - generuje a spouští Python skripty pro vizualizaci dat
"""
import os
import sys
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()


class VisualizationAgent:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.sandbox_dir = None

    def create_sandbox_environment(self) -> str:
        """
        Vytvoří izolované sandbox prostředí pro vykreslování grafů

        Vrací:
            Cesta k sandbox adresáři
        """
        self.sandbox_dir = tempfile.mkdtemp(prefix="viz_sandbox_")

        # Vytvoření requirements.txt pro sandbox
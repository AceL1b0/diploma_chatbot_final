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
        sandbox_requirements = os.path.join(self.sandbox_dir,
                                            "requirements.txt")
        with open(sandbox_requirements, 'w') as f:
            f.write(
                """pandas>=2.0.0
                numpy>=1.24.0
                matplotlib>=3.7.0
                seaborn>=0.12.0
                """
            )

        return self.sandbox_dir


    def generate_visualization_script(self, instructions: Dict[str, Any]) -> str:
        """
        Generuje Python skript pro vizualizaci dat

        Args:
            instructions: Instrukce od Main Agent

        Vrací:
            Python skript jako string
        """
        dataset_info = instructions["dataset_info"]
        graphs = instructions["graphs"]
        viz_type = instructions["visualization_type"]

        # Prompt pro generování vizualizačního skriptu
        system_prompt = f"""
Vytvořte Python skript pro datovou vizualizaci.

Dataset: {dataset_info['columns']} ({dataset_info['shape'][0]} řádků)
Grafy: {graphs}

POVINNÉ:
- Použijte pouze matplotlib a seaborn
- Načtěte data: df = pd.read_csv('data.csv')
- Uložte grafy: plt.savefig('graf1.png'), plt.savefig('graf2.png')
- Použijte sns.set_style() nebo sns.set_theme() místo plt.style.use()
- Zkontrolujte sloupce před použitím
- Žádné try/except/if/for/while bloky – pište pouze sekvenční kód (řádek za řádkem)

KRITICKÉ – ZÁVORKY (jinak skript spadne se SyntaxError):
- Každá otevírací závorka musí mít uzavírací: každé ( musí mít ), každé [ musí mít ].
- U volání jako .barh(...), .plot(...) pište VŠECHNY argumenty a uzavírací ) na JEDEN řádek, např. axes[1,0].barh(x, y, color='blue'). Nikdy neukončujte řádek čárkou a otevřenou závorkou – na dalším řádku musí být uzavírací ).
- Pokud volání rozdělíte na více řádků, na konci posledního řádku s argumentem musí být uzavírací ).

Vytvořte jednoduchý, funkční skript bez Plotly.
"""

        try:
            model = os.getenv("LLM_MODEL")
            if not model:
                return "# Chyba: LLM_MODEL není nastaveno v .env/Secrets"
            response = self.client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": system_prompt}]
            )

            script_text = response.content[0].text

            # Odstranění markdown bloků
            if "```python" in script_text:
                start_marker = "```python"
                end_marker = "```"
                start_idx = script_text.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = script_text.find(end_marker, start_idx)
                    if end_idx != -1:
                        script_text = script_text[start_idx:end_idx].strip()
                    else:
                        # Pokud nenajde konec, zkus obecný markdown
                        script_text = script_text[start_idx:].strip()
            elif "```" in script_text:
                # Obecný markdown blok
                start_idx = script_text.find("```") + 3
                end_idx = script_text.find("```", start_idx)
                if end_idx != -1:
                    script_text = script_text[start_idx:end_idx].strip()
                else:
                    # Pokud nenajde konec, vezmi vše od začátku
                    script_text = script_text[start_idx:].strip()

            return script_text

        except Exception as e:
            return f"# Chyba při generování skriptu: {str(e)}"


    def execute_visualization_script(self, script: str, dataset_path: str, sandbox_dir: str) -> Dict[str, Any]:
        """
         Spustí vizualizační skript v sandbox prostředí

         Args:
             script: Python skript
             dataset_path: Cesta k datasetu
             sandbox_dir: Sandbox adresář

         Returns:
             Dict s výsledky spuštění
         """

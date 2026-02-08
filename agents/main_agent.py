"""
Main Agent - řídí chat a komunikaci s uživatelem
"""
import os
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import json
import re

load_dotenv()


class MainAgent:
    def __init__(self):
        """Inicializace Main Agent"""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.conversation_history = []
        self.current_dataset = None
        self.dataset_info = None

    def process_uploaded_file(self, file_path: str) -> Dict[str, Any]:
        """
        Zpracuje nahraný soubor a extrahuje informace o datasetu

        Args:
            file_path: Cesta k nahranému souboru

        Returns:
            Dict s informacemi o datasetu
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Nepodporovaný typ souboru")

            self.current_dataset = df

            # Analýza datasetu
            dataset_info = {
                'shape': list(df.shape),
                'columns': df.columns.tolist(),
                'dtypes': {str(k): str(v) for k, v in
                           df.dtypes.to_dict().items()},
                'missing_values': {str(k): int(v) for k, v in
                                   df.isnull().sum().to_dict().items()},
                'numeric_columns': df.select_dtypes(
                    include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(
                    include=['object', 'category']).columns.tolist(),
                'sample_data': {
                    str(k): v.tolist() if hasattr(v, 'tolist') else v for k, v
                    in df.head().to_dict().items()}
            }

            self.dataset_info = dataset_info
            return dataset_info

        except Exception as e:
            return {"error": f"Chyba při zpracování souboru: {str(e)}"}

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Try to parse JSON from raw LLM text with common fallbacks."""
        if not text:
            raise ValueError("Empty response")
        # Strip markdown fences if present
        if "```" in text:
            # Prefer ```json block
            start_marker = "```json"
            if start_marker in text:
                start = text.find(start_marker) + len(start_marker)
                end = text.find("```", start)
                candidate = text[start:end].strip() if end != -1 else text
                return json.loads(candidate)
            # Fallback to first fenced block
            start = text.find("```") + 3
            end = text.find("```", start)
            candidate = text[start:end].strip() if end != -1 else text
            try:
                return json.loads(candidate)
            except Exception:
                pass
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            # Find first JSON object via regex
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            raise

    def interpret_user_request(self, user_message: str) -> Dict[str, Any]:
        """
        Interpretuje požadavek uživatele a rozhoduje o typu vizualizace

        Args:
            user_message: Zpráva od uživatele

        Returns:
            Dict s rozhodnutím o vizualizaci
        """
        if not self.dataset_info:
            return {"error": "Nejdříve nahrajte dataset"}

        # Prompt pro analýzu požadavku uživatele
        system_prompt = f"""
        Jste expert na analýzu dat a vizualizaci. Máte k dispozici dataset s následujícími informacemi:

        Dataset info:
        - Počet řádků: {self.dataset_info['shape'][0]}
        - Počet sloupců: {self.dataset_info['shape'][1]}
        - Sloupce: {self.dataset_info['columns']}
        - Numerické sloupce: {self.dataset_info['numeric_columns']}
        - Kategorické sloupce: {self.dataset_info['categorical_columns']}

        Uživatel napsal: "{user_message}"

        Rozhodněte:
        1. Zda uživatel specifikoval konkrétní grafy (specific_graphs: true/false)
        2. Pokud ano, jaké grafy chce (graph_types: [])
        3. Pokud ne, doporučte 3 profesionální grafy (default_graphs: [])

        Odpovězte ve formátu JSON a pouze JSON (bez Markdownu, bez komentářů), minifikovaně na jeden řádek:
        {{"specific_graphs": true/false, "graph_types": ["typ1", "typ2"], "default_graphs": ["typ1", "typ2", "typ3"], "reasoning": "..."}}
        """

        try:
            model = os.getenv("LLM_MODEL")
            if not model:
                raise ValueError("LLM_MODEL není nastaveno v .env/Secrets")
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": system_prompt}]
            )

            response_text = response.content[
                0].text if response and response.content else ""
            decision = self._extract_json(response_text)

            # Přidání do historie konverzace
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": decision.get("reasoning", "")
            })

            return decision

        except Exception as e:
            return {"error": f"Chyba při interpretaci požadavku: {str(e)}",
                    "raw": response_text if 'response_text' in locals() else ""}

    def generate_visualization_instructions(self, decision: Dict[str, Any]) -> \
    Dict[str, Any]:
        """
        Generuje instrukce pro Visualization Agent

        Args:

            decision: Rozhodnutí o typu vizualizace

        Returns:
            Dict s instrukcemi pro vizualizaci
        """
        if decision.get("error"):
            return decision

        instructions = {
            "dataset_info": self.dataset_info,
            "dataset_path": None,
            "visualization_type": "specific" if decision.get(
                "specific_graphs") else "default",
            "graphs": decision.get("graph_types", []) if decision.get(
                "specific_graphs") else decision.get("default_graphs", []),
            "reasoning": decision.get("reasoning", "")
        }

        return instructions

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Vrátí historii konverzace"""
        return self.conversation_history

    def reset_conversation(self):
        """Resetuje konverzaci a dataset"""
        self.conversation_history = []
        self.current_dataset = None
        self.dataset_info = None

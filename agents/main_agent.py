"""
Main Agent - řídí chat a komunikaci s uživatelem
"""
import os
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()


class MainAgent:
    INTERPRET_TOOL = {
        "name": "interpret_request",
        "description": "Interpretuje požadavek uživatele a rozhodne o typech grafů pro vizualizaci.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specific_graphs": {
                    "type": "boolean",
                    "description": "True pokud uživatel specifikoval konkrétní grafy, jinak False."
                },
                "graph_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Grafy požadované uživatelem (pouze pokud specific_graphs=true)."
                },
                "default_graphs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3 doporučené grafy pro dataset (pouze pokud specific_graphs=false)."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Stručné zdůvodnění výběru grafů."
                }
            },
            "required": ["specific_graphs", "graph_types", "default_graphs", "reasoning"]
        }
    }

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

    def interpret_user_request(self, user_message: str) -> Dict[str, Any]:
        if not user_message or not user_message.strip():
            user_message = "Vytvoř doporučené grafy"

        if not self.dataset_info:
            return {"error": "Nejdříve nahrajte dataset"}

        system_prompt = f"""Jste expert na analýzu dat a vizualizaci.

Dataset:
- Řádky: {self.dataset_info['shape'][0]}, Sloupce: {self.dataset_info['shape'][1]}
- Sloupce: {self.dataset_info['columns']}
- Numerické: {self.dataset_info['numeric_columns']}
- Kategorické: {self.dataset_info['categorical_columns']}

Rozhodněte o grafech pro požadavek uživatele pomocí nástroje interpret_request."""

        try:
            model = os.getenv("LLM_MODEL")
            if not model:
                raise ValueError("LLM_MODEL není nastaveno v .env/Secrets")

            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                tools=[self.INTERPRET_TOOL],
                tool_choice={"type": "tool", "name": "interpret_request"},
                messages=[{"role": "user", "content": user_message}]
            )

            decision = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "interpret_request":
                    decision = block.input
                    break

            if decision is None:
                return {"error": "LLM nevrátil tool_use blok"}

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": decision.get("reasoning", "")})

            return decision

        except Exception as e:
            return {"error": f"Chyba při interpretaci požadavku: {str(e)}"}

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

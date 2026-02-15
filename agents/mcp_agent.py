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


class MCPAgent:
    def __init__(self, evaluation_agent: EvaluationAgent, server_url: Optional[str] = None):
        self.eval = evaluation_agent
        self.server_url = server_url or os.getenv("MCP_SERVER_URL", "")

    def is_available(self) -> bool:
        if not self.server_url:
            print("❌ MCP_SERVER_URL není nastaven")
            return False
        try:
            health_url = f"{self.server_url.rstrip('/')}/health"
            print(f"🔍 Kontroluji MCP server: {health_url}")
            r = requests.get(health_url, timeout=5)
            print(f"📡 HTTP status: {r.status_code}")
            if r.status_code == 200:
                response = r.json()
                print(f"📋 Response: {response}")
                return response.get("status") == "ok"
            else:
                print(f"❌ HTTP error: {r.status_code}")
                return False
        except Exception as e:
            print(f"❌ Chyba při připojení k MCP server: {e}")
            return False

    def should_activate(self, user_request: str, visualization_type: str) -> bool:
        print(f"🔍 MCP should_activate: user_request='{user_request}', type='{visualization_type}'")
        # Aktivace MCP při nízkém skóre
        if self.eval.should_use_mcp():
            print("✅ MCP aktivován kvůli nízkému skóre")
            return True
        advanced_keywords = [
            "interaktivní", "3d", "animace", "real-time", "streaming",
            "machine learning", "plotly", "sankey", "treemap",
            # Klíčová slova
            "torch", "pytorch", "cuda", "gpu", "accelerator", "nvidia", "tensor",
        ]
        low = user_request.lower()
        for keyword in advanced_keywords:
            if keyword in low:
                print(f"✅ MCP aktivován kvůli klíčovému slovu: '{keyword}'")
                return True
        simple_keywords = ["histogram", "pie", "koláč", "box", "boxplot", "heatmap"]
        if any(k in low for k in simple_keywords):
            print("❌ MCP neaktivován - jednoduchá vizualizace")
            return False
        print("❌ MCP neaktivován - žádné pokročilé klíčové slovo")
        return False

    def generate_advanced(self, user_request: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            print("Nepodařilo se vytvořit vizualizaci")
            return {
                "success": False,
                "error": "MCP server není dostupný.",
                "generated_files": [],
                "execution_log": {},
            }

        try:
            payload = {
                "prompt": user_request,
                "dataset_info": dataset_info,
                "visualization_type": "advanced",
                "output_format": "png",
            }
            timeout = int(os.getenv("MCP_REQUEST_TIMEOUT", "180"))
            r = requests.post(
                f"{self.server_url.rstrip('/')}/advanced-visualization",
                json=payload,
                timeout=timeout,
            )
            if r.status_code != 200:
                print("Nepodařilo se vytvořit vizualizaci")
                return {
                    "success": False,
                    "error": f"HTTP {r.status_code}: {r.text[:500] if r.text else ''}",
                    "generated_files": [],
                    "execution_log": {},
                }

            data = r.json()
            if not data.get("success"):
                error_msg = data.get("error", "Neznámá chyba MCP")
                logs = data.get("logs", {})
                stderr = logs.get("stderr", "")
                stdout = logs.get("stdout", "")
                print("Nepodařilo se vytvořit vizualizaci")
                print(f"📋 stderr: {stderr}")
                print(f"📋 stdout: {stdout}")
                return {
                    "success": False,
                    "error": error_msg,
                    "generated_files": [],
                    "execution_log": {"stderr": stderr, "stdout": stdout},
                }

            gallery = data.get("visualizations_multi") or []
            saved_files = []
            try:
                if gallery:
                    for b64img in gallery:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                            f.write(base64.b64decode(b64img))
                            saved_files.append(f.name)
                else:
                    img_b64 = data.get("visualization", "")
                    if not img_b64:
                        print("Nepodařilo se vytvořit vizualizaci")
                        return {
                            "success": False,
                            "error": "Prázdný výstup z MCP serveru.",
                            "generated_files": [],
                            "execution_log": data.get("logs", {}),
                        }
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(base64.b64decode(img_b64))
                        saved_files.append(f.name)

                return {
                    "success": True,
                    "generated_files": saved_files,
                    "script": data.get("script", ""),
                    "execution_log": data.get("logs", {}),
                }
            except Exception as decode_error:
                print("Nepodařilo se vytvořit vizualizaci")
                return {
                    "success": False,
                    "error": f"Chyba při dekódování výstupu: {decode_error}",
                    "generated_files": [],
                    "execution_log": {},
                }
        except Exception as e:
            print("Nepodařilo se vytvořit vizualizaci")
            return {
                "success": False,
                "error": f"Chyba při komunikaci s MCP serverem: {e}",
                "generated_files": [],
                "execution_log": {},
            }

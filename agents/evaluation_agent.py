"""
Evaluation Agent - ukládání vizualizačních skriptů a hodnocení, výpočet skóre
"""
import base64
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

class EvaluationAgent:
    # Persist: False - skore pouze na 1 session
    # Persist: True - skore na disku (evaluation_data.json)
    def __init__(self, data_file: str = "evaluation_data.json", persist: bool = False):
        self.data_file = data_file
        self.persist = persist
        self.evaluations: List[Dict[str, Any]] = self._load()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[Anthropic] = Anthropic(api_key=api_key) if api_key else None

    def _load(self) -> List[Dict[str, Any]]:
        if not self.persist:
            return []
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self) -> None:
        if not self.persist:
            return
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.evaluations, f, ensure_ascii=False, indent=2)

    def _llm_model(self) -> str:
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise ValueError("LLM_MODEL není nastaveno v .env/Secrets")
        return model

    # Max rozměr obrázku pro Anthropic API (limit 8000 px; 1568 doporučeno pro rychlost)
    _MAX_IMAGE_DIM = 1568

    def _build_image_blocks(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Připraví content bloky pro Anthropic messages (text+image).
        Obrázky větší než _MAX_IMAGE_DIM px se zmenší, aby API nepřekročilo limit 8000 px.
        Podporuje pouze PNG/JPEG/WEBP. Ostatní typy (SVG/HTML) přeskočí.
        """
        blocks: List[Dict[str, Any]] = []
        for p in image_paths or []:
            if not p or not os.path.exists(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            media_type = None
            if ext == ".png":
                media_type = "image/png"
            elif ext in {".jpg", ".jpeg"}:
                media_type = "image/jpeg"
            elif ext == ".webp":
                media_type = "image/webp"
            else:
                continue

            try:
                if os.path.getsize(p) > 8 * 1024 * 1024:
                    continue
                with Image.open(p) as img:
                    img = img.convert("RGB" if img.mode not in ("RGB", "RGBA") else img.mode)
                    w, h = img.size
                    if max(w, h) > self._MAX_IMAGE_DIM:
                        ratio = self._MAX_IMAGE_DIM / max(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    fmt = "PNG" if media_type == "image/png" else "JPEG"
                    if fmt == "JPEG" and img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(buf, format=fmt, quality=85)
                    data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                blocks.append({"type": "text", "text": f"Soubor: {os.path.basename(p)}"})
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data_b64},
                    }
                )
            except Exception:
                continue
        return blocks

    def _call_llm_with_images(self, prompt: str, image_paths: List[str], max_tokens: int = 1800) -> str:
        if not self._client:
            return "⚠️ Nelze spustit auto-hodnocení/vysvětlení: chybí `ANTHROPIC_API_KEY`."
        try:
            model = self._llm_model()
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend(self._build_image_blocks(image_paths))
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text if resp and resp.content else ""
        except Exception as e:
            return f"⚠️ Auto-hodnocení/vysvětlení selhalo: {str(e)}"

    def save_script(
        self,
        script: str,
        user_request: str,
        visualization_type: str,
        graphs: List[str],
        auto_explanations: Optional[str] = None,
    ) -> str:
        eid = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rec = {
            "id": eid,
            "timestamp": datetime.now().isoformat(),
            "user_request": user_request,
            "visualization_type": visualization_type,
            "graphs": graphs,
            "script": script,
            "user_rating": None,
            "auto_explanations": auto_explanations,
        }
        self.evaluations.append(rec)
        self._save()
        return eid

    def add_rating(self, evaluation_id: str, rating: int) -> bool:
        for r in self.evaluations:
            if r.get("id") == evaluation_id:
                r["user_rating"] = 1 if rating == 1 else 0
                r["rating_timestamp"] = datetime.now().isoformat()
                self._save()
                return True
        return False

    def score_percentage(self) -> float:
        rated = [r for r in self.evaluations if r.get("user_rating") is not None]
        if not rated:
            return 100.0
        s = sum(r["user_rating"] for r in rated)
        return round(100.0 * s / len(rated), 1)

    def stats(self) -> Dict[str, Any]:
        rated = [r for r in self.evaluations if r.get("user_rating") is not None]
        good = sum(1 for r in rated if r.get("user_rating") == 1)
        return {
            "total": len(self.evaluations),
            "rated": len(rated),
            "good": good,
            "bad": len(rated) - good,
            "score": self.score_percentage(),
        }

    def should_use_mcp(self, threshold: float = 50.0) -> bool:
        rated = [r for r in self.evaluations if r.get("user_rating") is not None]
        if not rated:
            return False
        return self.score_percentage() < threshold

    def explain(
        self,
        *,
        user_request: str,
        dataset_info: Dict[str, Any],
        visualization_type: str,
        requested_graphs: List[str],
        generated_files: List[str],
    ) -> str:
        """
        LLM: vysvětlí uživateli, co jednotlivé grafy znamenají a jak je číst.
        Vrací markdown text (v bodech, pro každý graf zvlášť).
        """
        dataset_summary = {
            "shape": dataset_info.get("shape"),
            "columns": dataset_info.get("columns"),
            "numeric_columns": dataset_info.get("numeric_columns"),
            "categorical_columns": dataset_info.get("categorical_columns"),
            "dtypes": dataset_info.get("dtypes"),
        }
        prompt = f"""
Jsi datový analytik a učitel. Níže dostaneš jednotlivé grafy jako obrázky.

## Kontext
- Uživatelský požadavek: {user_request}
- Typ vizualizace: {visualization_type}
- Požadované grafy (pokud byly specifikovány): {requested_graphs}
- Dataset info (souhrn): {json.dumps(dataset_summary, ensure_ascii=False)}

## Úkol: VYSVĚTLENÍ PRO UŽIVATELE
Napiš vysvětlení **pro každý graf zvlášť** jako seznam bodů. U každého grafu pokryj:
- **Co graf zobrazuje** (co je na ose X/Y, co znamenají barvy/legendy, jednotky pokud jsou vidět)
- **Jak graf číst** (na co se dívat, jak interpretovat trend/rozdíly/rozptyl)
- **Co je hlavní takeaway** (1–3 klíčová zjištění, jen pokud jsou z obrázku odůvodnitelná)
- **Na co si dát pozor** (např. outliery, malý vzorek, korelace ≠ kauzalita, škála os)

Formát:
- Použij nadpisy typu `### Graf 1: <název souboru>` (nebo podobně), a pod tím odrážky.

Pravidla:
- Nehalucinuj konkrétní hodnoty ani vztahy, které nejsou z grafu patrné.
- Pokud je graf nečitelný / bez popisků, řekni to a uveď, co chybí.
- VŽDY dokonči vysvětlení všech grafů. Nepřidávej žádný text za posledním grafem.
- Nikdy nezačínej větu nebo bod, který nedokončíš.
"""
        return self._call_llm_with_images(prompt, generated_files, max_tokens=2500)

    def latest_id(self) -> Optional[str]:
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda r: r.get("timestamp", ""))

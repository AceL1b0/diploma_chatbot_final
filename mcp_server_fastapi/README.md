# MCP Visualization Server (FastAPI)

HTTP server pro pokročilé vizualizace. LLM vybere 2–3 nástroje podle schématu dat a požadavku; server je spustí a vrátí obrázky v base64. Nástroje: `dual_axes`, `boxplot`, `heatmap` (matplotlib + seaborn).

## Endpointy

- **GET `/health`** – `{ "status": "ok" }`
- **POST `/advanced-visualization`**
  - **Body:** `{ "prompt": string, "dataset_info": object, "output_format": "png" }`
  - **dataset_info** musí obsahovat **`sample_data`** (data pro vykreslení, např. slovníky nebo řádky).
  - **Odpověď:** `{ "success", "visualization" (base64), "visualizations_multi", "visualizations", "insight", "script", "logs" }`

## Lokální spuštění

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export LLM_MODEL=claude-3-5-sonnet-20241022
uvicorn app:app --host 0.0.0.0 --port 7860
```
## Nasazení na Hugging Face Spaces

1. Vytvořte nový Space (typ: Python, CPU).
2. Nahrajte obsah složky `mcp_server_fastapi/` (zejména `app.py`, `requirements.txt`).
3. V Settings → Secrets přidejte `ANTHROPIC_API_KEY` a volitelně `LLM_MODEL`.
4. Jako App file zadejte `app:app`; příkaz spuštění např. `uvicorn app:app --host 0.0.0.0 --port 7860`.
5. Po buildu bude Space dostupný na `https://<vas-space>.hf.space`.

## Příklad volání

```bash
curl -X POST "$URL/advanced-visualization" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Porovnej sloupce",
    "dataset_info": {
      "sample_data": [
        {"kategorie": "A", "hodnota": 10},
        {"kategorie": "B", "hodnota": 20}
      ]
    },
    "output_format": "png"
  }'
```

## Integrace s MCP agentem

Lokální Gradio app nastaví `MCP_SERVER_URL` na URL tohoto serveru. MCP Agent pošle `POST /advanced-visualization` s `prompt` a `dataset_info` (včetně `sample_data` z nahraného datasetu) a z odpovědi použije `visualization` a `visualizations_multi` jako obrázky.

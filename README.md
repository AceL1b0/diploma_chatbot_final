# Multiagentní chatbot pro vizualizaci dat

Aplikace, ve které uživatel nahraje dataset (CSV/Excel), zadá požadavek a systém vygeneruje grafy. Jednoduché vizualizace běží lokálně (Visualization Agent), pokročilé lze posílat na MCP server. Všechny grafy jsou matplotlib + seaborn (bez Plotly).

## Struktura projektu

```
python_diploma_new/
├── agents/
│   ├── main_agent.py          # interpretace dotazu, tvorba instrukcí, dataset_info
│   ├── visualization_agent.py # generování a běh Python skriptu v sandboxu (lokálně)
│   ├── evaluation_agent.py    # ukládání skriptů, rating, vysvětlení grafů (LLM)
│   └── mcp_agent.py           # rozhodnutí o MCP, volání MCP serveru
├── mcp_server_fastapi/
│   ├── app.py                 # FastAPI server: LLM + nástroje (dual_axes, boxplot, heatmap)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── gradio_app.py              # UI (upload, chat, galerie)
├── main.py                    # spuštění Gradio
├── requirements.txt
└── README.md
```

## Instalace a spuštění (lokální UI)

1. **Virtuální prostředí**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Závislosti**
   ```bash
   pip install -r requirements.txt
   ```

3. **Konfigurace (.env)**
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Volitelně: `LLM_MODEL`, `MCP_SERVER_URL`, `SANDBOX_TIMEOUT`, `MCP_REQUEST_TIMEOUT`.

4. **Spuštění**
   ```bash
   python main.py
   ```
   Aplikace běží na `http://localhost:7862`.

## Jak to funguje

1. **Main Agent** – z nahraného souboru sestaví `dataset_info` (sloupce, typy, vzorek dat). Z textu uživatele určí typ vizualizace a seznam grafů.
2. **Rozhodnutí MCP vs lokál** – MCP Agent aktivuje MCP server při nízkém skóre nebo při pokročilých požadavcích (3D, ML, torch, …). Jednoduché grafy (histogram, box, heatmap, …) dělá lokální Visualization Agent.
3. **Visualization Agent** – vytvoří sandbox, zkopíruje data jako `data.csv`, vygeneruje skript (LLM), spustí ho a posbírá vygenerované obrázky (main.png, graf1.png, …).
4. **MCP server** – dostane `prompt` a `dataset_info` (včetně `sample_data`). LLM vybere 2–3 nástroje (dual_axes, boxplot, heatmap), server je spustí a vrátí obrázky v base64.
5. **Evaluation Agent** – ukládá skripty a metadata, zpracovává rating; může generovat vysvětlení grafů (LLM nad obrázky).

## MCP server (volitelně)

Server pro pokročilé vizualizace. Běží samostatně (lokálně nebo na HF Spaces).

**Lokální spuštění**
```bash
cd mcp_server_fastapi
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export LLM_MODEL=claude-3-5-sonnet-20241022   # nebo jiný model
uvicorn app:app --host 0.0.0.0 --port 7860
```

**Připojení z UI**  
V `.env` nastavte `MCP_SERVER_URL=https://vas-server.example.com` (bez koncového lomítka). Health check: `GET {MCP_SERVER_URL}/health`.

**API**
- `POST /advanced-visualization` – body: `{ "prompt": "", "dataset_info": { "sample_data": ... }, "output_format": "png" }`. Odpověď: `success`, `visualization` (base64), `visualizations_multi`, `visualizations`, `insight`, `logs`.

## Konfigurace (.env)

| Proměnná | Popis |
|----------|--------|
| `ANTHROPIC_API_KEY` | Povinné pro LLM (lokální agenti i MCP server). |
| `LLM_MODEL` | Model pro Main/Visualization/Evaluation agenta a pro MCP (např. `claude-3-5-sonnet-20241022`). |
| `MCP_SERVER_URL` | URL MCP serveru (např. HF Space). Bez nastavení se MCP nevolá. |
| `SANDBOX_TIMEOUT` | Timeout běhu vizualizačního skriptu v sekundách (výchozí 30). |
| `MCP_REQUEST_TIMEOUT` | Timeout HTTP požadavku na MCP (výchozí 180). |

## Troubleshooting

- **„Vizualizace vytvořeny, ale žádné soubory“** – skript neuložil obrázky; zkontrolujte stderr/stdout v odpovědi agenta. Visualization Agent hledá soubory podle názvů (main, graph1, …) a přípon (png, svg, …).
- **„MCP se nespustil“** – ověřte `MCP_SERVER_URL` a že `GET {URL}/health` vrací 200. MCP se neaktivuje pro čistě jednoduché grafy, pokud není nízké skóre (v konzoli je důvod aktivace/ignorace).
- **„Prázdný výstup z MCP serveru“** – server vrátil `success: true`, ale chybí `visualization`. Ověřte, že MCP vrací pole `visualization` (base64) a případně `visualizations_multi`.
- **Hodnocení / skóre** – výchozí režim je in-memory. Perzistenci lze zapnout v kódu (EvaluationAgent).

## Závislosti (kořen projektu)

- anthropic, gradio, pandas, numpy, matplotlib, seaborn, python-dotenv, scikit-learn, Pillow

Vizualizace pouze matplotlib + seaborn (bez Plotly/Kaleido).

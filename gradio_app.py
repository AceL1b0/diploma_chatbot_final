"""
Gradio rozhraní pro multiagentní chatbot pro vizualizaci dat
"""
import gradio as gr
import os
import tempfile
from typing import List, Tuple, Optional, Dict
import shutil

from agents.main_agent import MainAgent
from agents.visualization_agent import VisualizationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.mcp_agent import MCPAgent


class DataVisualizationChatbot:
    def __init__(self):
        """Inicializace chatbotu"""
        self.main_agent = MainAgent()
        self.viz_agent = VisualizationAgent()
        # Skóre platné pouze pro jedno spuštění (in-memory)
        self.eval_agent = EvaluationAgent(persist=False)
        self.mcp_agent = MCPAgent(self.eval_agent)
        self.current_file_path = None
        self.current_evaluation_id = None

    def process_file_upload(self, file) -> Tuple[str, str]:
        """
        Zpracuje nahraný soubor

        Args:
            file: Nahraný soubor

        Vrací:
            Tuple s informacemi o datasetu a stavem
        """
        if file is None:
            return "Nejprve nahrajte dataset", ""

        try:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, file.name.split('/')[-1])
            shutil.copy2(file.name, file_path)

            self.current_file_path = file_path

            # Zpracování souboru
            dataset_info = self.main_agent.process_uploaded_file(file_path)

            if "error" in dataset_info:
                return f"Chyba: {dataset_info['error']}", ""

            info_text = f"""
**Dataset úspěšně nahran!**

📊 **Základní informace:**
- Řádky: {dataset_info['shape'][0]}
- Sloupce: {dataset_info['shape'][1]}

📋 **Sloupce:**
{', '.join(dataset_info['columns'])}

🔢 **Numerické sloupce:**
{', '.join(dataset_info['numeric_columns']) if dataset_info['numeric_columns'] else 'Žádné'}

📝 **Kategorické sloupce:**
{', '.join(dataset_info['categorical_columns']) if dataset_info['categorical_columns'] else 'Žádné'}

❓ **Chybějící hodnoty:**
{sum(dataset_info['missing_values'].values())} celkem

---
**Nyní můžete:**
- Požádat o konkrétní grafy (např. "Vytvoř histogram věku a scatter plot výšky vs váhy")
- Nebo jen napsat "Vytvoř grafy" pro 3 doporučené vizualizace
            """

            return info_text, "Dataset připraven k vizualizaci!"

        except Exception as e:
            return f"Chyba při zpracování souboru: {str(e)}", ""

    def process_user_message(self, message: str,
                             history: List[Dict[str, str]]) -> Tuple[
        str, List[Dict[str, str]], str, List[str]]:
        """
        Zpracuje zprávu uživatele a vrátí odpověď

        Args:
            message: Zpráva od uživatele
            history: Historie konverzace

        Vrací:
            Tuple s odpovědí, aktualizovanou historií a stavem
        """
        if not self.current_file_path:
            return "Nejprve nahrajte dataset!", history, "Chyba: Dataset není nahraný", []

        try:
            decision = self.main_agent.interpret_user_request(message)

            if "error" in decision:
                return f"Chyba: {decision['error']}", history, "Chyba při interpretaci", []

            instructions = self.main_agent.generate_visualization_instructions(
                decision)

            if "error" in instructions:
                return f"Chyba: {instructions['error']}", history, "Chyba při generování instrukcí", []

            # Rozhodnutí mezi MCP Agent a Visualization Agent
            if self.mcp_agent.should_activate(message, instructions[
                "visualization_type"]):
                print("🔄 Aktivace MCP Agent pro pokročilé vizualizace")
                viz_result = self.mcp_agent.generate_advanced(message,
                                                              instructions.get(
                                                                  "dataset_info",
                                                                  {}))
            else:
                print("📊 Použití standardního Visualization Agent")
                viz_result = self.viz_agent.create_visualizations(instructions,
                                                                  self.current_file_path)

            if not viz_result["success"]:
                error_msg = viz_result.get("error", "Neznámá chyba")
                execution_log = viz_result.get("execution_log", {})
                stderr = execution_log.get("stderr", "")
                stdout = execution_log.get("stdout", "")
                print("Nepodařilo se vytvořit vizualizaci")
                if error_msg:
                    print(f"  Chyba: {error_msg}")
                if stderr:
                    print(f"  stderr: {stderr[:500]}")
                if stdout:
                    print(f"  stdout: {stdout[:500]}")
                user_message = "**Nepodařilo se vytvořit vizualizaci.**"
                if error_msg:
                    user_message += f"\n\n{error_msg}"
                if stderr:
                    user_message += f"\n\n**Chyba při spuštění:**\n```\n{stderr[:2000]}\n```"
                if stdout:
                    user_message += f"\n\n**Výstup:**\n```\n{stdout[:1000]}\n```"
                return user_message, history, "Nepodařilo se vytvořit vizualizaci", []

            # Příprava odpovědi
            generated_files = viz_result["generated_files"]

            if generated_files:
                response = f"✅ **Vizualizace úspěšně vytvořeny!**\n\n"
                response += f"📊 **Vytvořené grafy:**\n"
                for i, file_path in enumerate(generated_files, 1):
                    filename = os.path.basename(file_path)
                    response += f"{i}. {filename}\n"

                response += f"\n🎯 **Typ vizualizace:** {'Specifické grafy' if instructions['visualization_type'] == 'specific' else 'Doporučené grafy'}\n"
                response += f"📝 **Počet grafů:** {len(generated_files)}\n"

                # Vysvětlení grafů (LLM) místo původního "reasoning"
                script = viz_result.get("script", "")
                auto_explain = self.eval_agent.explain(
                    user_request=message,
                    dataset_info=instructions.get("dataset_info", {}),
                    visualization_type=instructions.get("visualization_type",
                                                        ""),
                    requested_graphs=instructions.get("graphs", []),
                    generated_files=generated_files,
                )

                if auto_explain:
                    response += "\n\n## 📘 Vysvětlení grafů\n" + auto_explain.strip()

                # Uložení do Evaluation Agent
                if script:
                    self.current_evaluation_id = self.eval_agent.save_script(
                        script=script,
                        user_request=message,
                        visualization_type=instructions["visualization_type"],
                        graphs=instructions.get("graphs", []),
                        auto_explanations=auto_explain,
                    )

                # Aktualizace historie
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})

                return response, history, "Vizualizace vytvořeny úspěšně!", generated_files
            else:
                return "Vizualizace byly vytvořeny, ale nebyly nalezeny žádné soubory.", history, "Upozornění: Žádné soubory", []

        except Exception as e:
            return f"Chyba při zpracování požadavku: {str(e)}", history, "Chyba systému", []

    def rate_visualization(self, rating: int) -> Tuple[str, str]:
        """
        Uloží hodnocení vizualizace

        Args:
            rating: Hodnocení (0 = špatné, 1 = dobré)

        Returns:
            Tuple s potvrzením a aktualizovaným skóre
        """
        if self.current_evaluation_id is None:
            return "❌ Žádná vizualizace k hodnocení", "0%"

        val = str(rating).strip().lower()
        mapped = 1 if val in {"1", "dobré", "✅ dobré", "true", "yes", "y",
                              "good"} else 0

        success = self.eval_agent.add_rating(self.current_evaluation_id,
                                             mapped)
        if not success:
            return "❌ Chyba při ukládání hodnocení", "0%"

        stats = self.eval_agent.stats()
        score_text = f"📊 **Aktuální skóre: {stats['score']}%**\n"
        score_text += f"Celkem hodnocení: {stats['rated']} (Dobré: {stats['good']}, Špatné: {stats['bad']})"

        return f"✅ Hodnocení uloženo ({'Dobré' if rating == 1 else 'Špatné'})", score_text

    def get_evaluation_stats(self) -> str:
        """
        Vrátí aktuální statistiky hodnocení

        Vrací:
            String s aktuálním skóre
        """
        stats = self.eval_agent.stats()
        if stats['rated'] == 0:
            return "📊 **Zatím žádná hodnocení**"

        score_text = f"📊 **Aktuální skóre: {stats['score']}%**\n"
        score_text += f"Celkem hodnocení: {stats['rated']} (Dobré: {stats['good']}, Špatné: {stats['bad']})"
        return score_text

    def reset_conversation(self) -> Tuple[str, str, str, List[str]]:
        """Resetuje konverzaci a dataset"""
        self.main_agent.reset_conversation()
        self.viz_agent.cleanup_sandbox()
        self.current_file_path = None
        return "", [], "Konverzace resetována", []


def create_gradio_interface():
    """Vytvoří a spustí Gradio rozhraní"""
    chatbot = DataVisualizationChatbot()

    with gr.Blocks(
            title="Multiagentní Chatbot pro Vizualizaci Dat",
            theme=gr.themes.Soft(),
            css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as interface:
        gr.Markdown("""
        # 🤖 Multiagentní Chatbot pro Vizualizaci Dat

        Tento chatbot vám pomůže vytvořit profesionální vizualizace z vašich dat pomocí čtyř specializovaných agentů:

        - **Main Agent**: Rozumí vašim požadavkům a rozhoduje o typu vizualizace
        - **Visualization Agent**: Generuje a spouští Python skripty pro vytvoření grafů
        - **Evaluation Agent**: Ukládá skripty a hodnocení
        - **MCP Agent**: Rozhoduje o volání vzdáleného MCP serveru

        ## 🚀 Jak začít:
        1. Nahrajte váš dataset (CSV, Excel)
        2. Napište požadavek na vizualizaci
        3. Chatbot vytvoří grafy!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Upload souboru
                file_upload = gr.File(
                    label="📁 Nahrajte dataset",
                    file_types=[".csv", ".xlsx", ".xls"],
                    type="filepath"
                )

                # Tlačítko pro reset
                reset_btn = gr.Button("🔄 Resetovat", variant="secondary")

                # Status
                status = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    value="Čekám na nahraný dataset..."
                )

                # Hodnocení vizualizací
                with gr.Group():
                    gr.Markdown("### 📊 Hodnocení vizualizací")
                    rating_radio = gr.Radio(
                        choices=[(0, "❌ Špatné"), (1, "✅ Dobré")],
                        label="Hodnocení poslední vizualizace",
                        value=None
                    )
                    rate_btn = gr.Button("💾 Uložit hodnocení",
                                         variant="secondary")
                    score_display = gr.Markdown(
                        value="📊 **Zatím žádná hodnocení**",
                        label="Aktuální skóre"
                    )

            with gr.Column(scale=2):
                # Chat interface
                chatbot_interface = gr.Chatbot(
                    label="💬 Chat s chatbotem",
                    height=400,
                    show_label=True,
                    type="messages"
                )

                # Input pro zprávy
                msg_input = gr.Textbox(
                    label="Napište Váš požadavek",
                    placeholder="Např: 'Vytvoř histogram věku a scatter plot výšky vs váhy' nebo 'Vytvoř grafy'",
                    lines=2
                )

                # Tlačítko pro odeslání
                send_btn = gr.Button("📤 Odeslat", variant="primary")

        # Dataset info
        dataset_info = gr.Markdown(
            value="**Dataset informace se zobrazí zde po nahrání souboru**",
            label="📋 Informace o datasetu"
        )

        # Galerie pro zobrazení vygenerovaných grafů
        gallery = gr.Gallery(
            label="📊 Vygenerované grafy",
            show_label=True,
            elem_id="gallery",
            columns=2,
            rows=2,
            height="auto"
        )

        # Event handlers
        file_upload.change(
            fn=chatbot.process_file_upload,
            inputs=[file_upload],
            outputs=[dataset_info, status]
        )

        send_btn.click(
            fn=chatbot.process_user_message,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface, status, gallery]
        )

        msg_input.submit(
            fn=chatbot.process_user_message,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface, status, gallery]
        )

        reset_btn.click(
            fn=chatbot.reset_conversation,
            outputs=[dataset_info, chatbot_interface, status, gallery]
        )

        # Event handler pro hodnocení
        rate_btn.click(
            fn=chatbot.rate_visualization,
            inputs=[rating_radio],
            outputs=[status, score_display]
        )

        # Zápatí
        gr.Markdown("""
        ---
        **💡 Tipy:**
        - Pro konkrétní grafy: "Vytvoř histogram věku, box plot výšky a scatter plot váhy vs výška"
        - Pro doporučené grafy: "Vytvoř grafy" nebo "Doporuč grafy"
        - Chatbot automaticky vybere nejlepší vizualizace pro váš dataset
        """)

    return interface


if __name__ == "__main__":
    # Kontrola API klíče
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv(
            "ANTHROPIC_API_KEY") == "your_anthropic_api_key_here":
        print("⚠️  UPOZORNĚNÍ: Nastavte ANTHROPIC_API_KEY v .env souboru!")
        print(
            "Zkopírujte Váš API klíč z Anthropic Console a vložte ho do .env souboru.")

    # Spuštění Gradio aplikace
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )

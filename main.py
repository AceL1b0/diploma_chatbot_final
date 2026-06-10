"""
Hlavní spouštěcí soubor pro multiagentní chatbot pro vizualizaci dat
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_requirements():
    """Kontrola, zda jsou splněny všechny požadavky"""
    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here":
            missing_vars.append(var)

    if missing_vars:
        print("❌ Chybějící environment proměnné:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Nastavte je v .env souboru:")
        print("   ANTHROPIC_API_KEY=your_actual_api_key_here")
        return False

    return True


def main():
    """Hlavní funkce pro spuštění aplikace"""
    print("🤖 Multiagentní Chatbot pro Vizualizaci Dat")
    print("=" * 50)

    # Kontrola požadavků
    if not check_requirements():
        print(
            "\n⚠️  Aplikace nemůže být spuštěna bez správně nastavených API klíčů.")
        return

    print("✅ Všechny požadavky splněny!")
    print("🚀 Spouštím Gradio rozhraní...")

    try:
        from gradio_app import create_gradio_interface

        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=False,
            show_error=True,
            inbrowser=True
        )

    except ImportError as e:
        print(f"❌ Chyba při importu: {e}")
        print("💡 Spusťte: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Chyba při spuštění: {e}")


if __name__ == "__main__":
    main()


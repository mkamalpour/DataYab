"""
app.py — Main Entry Point for DataYab
"""

import sys
import time
from pathlib import Path

# Add current directory to path so we can import backend and frontend modules
sys.path.insert(0, str(Path(__file__).parent))


def wait_for_input(prompt):
    """Robust input function that waits properly."""
    try:
        return input(prompt)
    except KeyboardInterrupt:
        return ""
    except EOFError:
        return ""


def launch_tkinter():
    """Launch Tkinter GUI."""
    try:
        from frontend.tinker import DataYabApp
        print("\n🚀 Launching DataYab Tkinter GUI...\n")
        app = DataYabApp()
        app.run()
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure 'frontend/tinker.py' exists in the frontend folder.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def launch_web():
    """Launch FastAPI web interface."""
    try:
        from frontend.web import run_web_app
        run_web_app(host='127.0.0.1', port=8001)
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure FastAPI is installed: pip install fastapi uvicorn")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def launch_cli():
    """Launch CLI mode."""
    try:
        from backend.core import DataYabCore
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure 'backend/core.py' exists in the backend folder.")
        return
    
    print("\n" + "="*50)
    print("📊 DataYab — CLI Mode")
    print("="*50)
    
    path = wait_for_input("\n📁 Dataset path: ")
    if not path:
        print("❌ No path provided")
        return
    
    core = DataYabCore()
    
    if path.endswith('.csv'):
        result = core.load_csv(path)
    elif path.endswith(('.xlsx', '.xls')):
        result = core.load_excel(path)
    elif path.endswith('.json'):
        result = core.load_json(path)
    elif path.endswith('.parquet'):
        result = core.load_parquet(path)
    else:
        print(f"Unsupported: {path}")
        return
    
    if not result["success"]:
        print(f"❌ Error loading file: {result['message']}")
        return
    
    print(f"\n✅ Loaded: {core.df.shape[0]} rows, {core.df.shape[1]} columns")
    
    target = wait_for_input("\n🎯 Target column (or Enter to skip): ")
    
    if target and target in core.df.columns:
        print("\n⏳ Training model...")
        core.auto_clean(target_col=target)
        core.split_data(target, test_size=0.2)
        core.train("Random Forest")
        
        print("\n📈 Results:")
        for k, v in core.get_metrics().items():
            if not isinstance(v, str):
                print(f"   {k}: {v:.4f}")
    elif target:
        print(f"⚠️ Target '{target}' not found in columns. Skipping training.")
    
    print("\n✅ Done!")
    wait_for_input("\nPress Enter to exit...")


def show_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                      DataYab Launcher                        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python app.py              → Interactive menu
    python app.py --tkinter    → Launch GUI directly
    python app.py --web        → Launch Web interface directly
    python app.py --cli        → Launch CLI directly

Files needed:
    • core.py      (backend logic)
    • tinker.py    (Tkinter GUI)
    • web.py       (FastAPI web interface)
    • app.py       (this launcher)

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn pillow joblib openpyxl fastapi uvicorn
""")


def main():
    """Main entry point with loop instead of recursion."""
    time.sleep(0.5)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--tkinter", "-t", "tkinter"]:
            launch_tkinter()
        elif arg in ["--web", "-w", "web"]:
            launch_web()
        elif arg in ["--cli", "-c", "cli"]:
            launch_cli()
        elif arg in ["--help", "-h"]:
            show_help()
        else:
            print(f"Unknown argument: {arg}")
            show_help()
        return
    
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except:
        pass
    
    while True:
        print("\n" + "="*40)
        print("   📊 DataYab — Ultimate Dataset Analyzer")
        print("="*40)
        print("\n[1] 🖥️  Tkinter GUI (Desktop App)")
        print("[2] 🌐 Web Interface (FastAPI + HTML)")
        print("[3] 📟 CLI Mode (Terminal)")
        print("[4] 📖 Help")
        print("[0] ❌ Exit")
        print("-"*40)
        
        try:
            choice = input("\n👉 Enter your choice (1-3): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        
        if choice == "1":
            launch_tkinter()
            break
        elif choice == "2":
            launch_web()
            break
        elif choice == "3":
            launch_cli()
            break
        elif choice == "4":
            show_help()
            wait_for_input("\nPress Enter to return to menu...")
            continue
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice == "":
            print("\n❌ No choice entered. Launching Tkinter GUI by default...")
            launch_tkinter()
            break
        else:
            print(f"\n❌ Invalid choice: '{choice}'. Please enter 1-4 or 0.")
            print("   Launching Tkinter GUI by default...")
            launch_tkinter()
            break


if __name__ == "__main__":
    main()
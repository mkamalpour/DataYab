"""Launch the DataYab local dataset workbench."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="DataYab: explore, clean and model tabular datasets."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--streamlit", "-s", action="store_true", help="Open the Streamlit web UI (recommended)"
    )
    modes.add_argument(
        "--tkinter",
        "-t",
        action="store_true",
        help="Open the desktop application (Tkinter)",
    )
    modes.add_argument(
        "--cli", "-c", action="store_true", help="Inspect a dataset in the terminal"
    )
    parser.add_argument("--file", help="Dataset path for CLI mode")
    parser.add_argument("--target", help="Optional target column for CLI training")
    parser.add_argument(
        "--task", choices=["auto", "classification", "regression"], default="auto"
    )
    args = parser.parse_args()
    try:
        if args.streamlit:
            import subprocess
            import sys
            from pathlib import Path
            streamlit_path = str(Path(__file__).resolve().parent / "frontend" / "streamlit" / "app.py")
            subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_path], check=True)
        elif args.cli:
            from backend.core import DataYabCore

            if not args.file:
                parser.error("--cli requires --file")
            core = DataYabCore()
            result = core.load_file(args.file)
            if not result["success"]:
                raise ValueError(result["message"])
            print(result["message"])
            if args.target:
                core.split_data(args.target, problem_type=args.task)
                core.train("Random Forest")
                print(json.dumps(core.get_metrics(), indent=2, ensure_ascii=True))
        else:
            from frontend.desktop import DataYabApp

            DataYabApp().run()
        return 0
    except (KeyboardInterrupt, EOFError):
        return 130
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "\nDataYab could not start because a required dependency is missing.\n"
            "\nUse the assigned environment and install the project requirements first:\n"
            "  conda activate base\n"
            "  python -m pip install -r requirements.txt\n"
            "\nFor Streamlit UI, also install:\n"
            "  python -m pip install -r requirements-streamlit.txt\n"
            "\nIf you are using the managed workspace, open the project workspace for the assigned interpreter.\n"
            f"Original error: {type(exc).__name__}: {exc}\n"
        )
        return 2
    except Exception as error:
        print(f"DataYab: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

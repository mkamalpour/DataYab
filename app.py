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
        "--web", "-w", action="store_true", help="Open the local web server"
    )
    modes.add_argument(
        "--tkinter",
        "-t",
        action="store_true",
        help="Open the desktop application (default)",
    )
    modes.add_argument(
        "--cli", "-c", action="store_true", help="Inspect a dataset in the terminal"
    )
    parser.add_argument("--file", help="Dataset path for CLI mode")
    parser.add_argument("--target", help="Optional target column for CLI training")
    parser.add_argument(
        "--task", choices=["auto", "classification", "regression"], default="auto"
    )
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    try:
        if args.web:
            from frontend.web import run_web_app

            run_web_app(port=args.port)
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
    except Exception as error:
        print(f"DataYab: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

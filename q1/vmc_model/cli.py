"""Umbrella CLI so you can run `python -m vmc_model train` or `predict`."""
import argparse
from importlib import import_module

_commands = {
    "train": "vmc_model.train:train_best",
    "predict": "vmc_model.predict:main",
}


def _resolve(target: str):
    mod_name, func_name = target.split(":")
    module = import_module(mod_name)
    return getattr(module, func_name)


def main():
    parser = argparse.ArgumentParser(prog="vmc_model", description="VMC calibration pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    pred = sub.add_parser("predict")
    pred.add_argument("--source", required=True)
    pred.add_argument("--model", default="model.joblib")
    pred.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    func = _resolve(_commands[args.command])
    if args.command == "predict":
        func(source=args.source, model=args.model, output=args.output)
    else:
        func()


if __name__ == "__main__":
    main()

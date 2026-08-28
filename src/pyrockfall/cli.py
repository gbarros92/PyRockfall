# pyrockfall/cli.py
from __future__ import annotations

import argparse

from pyrockfall import scripts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyrockfall",
        description="PyRockFall command-line interface.",
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
        metavar="{" + ",".join(scripts.__all__) + "}",
    )

    for name in scripts.__all__:
        script = getattr(scripts, name)
        p = sub.add_parser(
            name,
            help=script.HELP,
            description=script.DESCRIPTION,
        )
        script.add_arguments(p)
        p.set_defaults(func=script.main_from_namespace)

    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())

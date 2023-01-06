import argparse
from script_functions import python_to_cast


def get_args():
    parser = argparse.ArgumentParser(
        "Runs Python to CAST pipeline on input Python source file."
    )
    parser.add_argument(
        "--astpp", help="Dumps Python AST to stdout", action="store_true"
    )
    parser.add_argument(
        "--rawjson",
        help="Dumps out raw JSON contents to stdout",
        action="store_true",
    )
    parser.add_argument(
        "--stdout",
        help="Dumps CAST JSON to stdout instead of a file",
        action="store_true",
    )
    parser.add_argument(
        "--agraph",
        help="Generates visualization of CAST as a PDF file",
        action="store_true",
    )
    parser.add_argument(
        "--legacy",
        help="Generate CAST for GrFN 2.2 pipeline",
        action="store_true",
    )
    parser.add_argument("pyfile_path", help="input Python source file")
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    python_to_cast(
        args.pyfile_path,
        args.agraph,
        args.astpp,
        args.stdout,
        args.rawjson,
        args.legacy,
    )

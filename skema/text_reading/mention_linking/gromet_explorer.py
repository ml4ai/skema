"""Utility to see the mentions with alignments and gromets."""

import json
from pprint import pprint
import argparse


def walk_nested_structure(obj, md_keys):
    ret = list()

    if type(obj) == dict:
        obj = [obj]

    for elem in obj:
        if type(elem) is dict:
            md_ix = elem.get("metadata")

            if md_ix in md_keys:
                ret.append(elem)

            for k, v in elem.items():
                if type(v) in {dict, list}:
                    ret.extend(walk_nested_structure(v, md_keys))

    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument(
        "-p",
        "--print_aligned",
        help="Prints the ports that have a linked text reading mention.",
        action="store_true",
    )
    # TODO
    # group by tr linked mention
    # group by DKG grounded entity
    args = parser.parse_args()

    with open(args.input_path) as f:
        gromet_fn_module = json.load(f)

    # First, retrieve the relevant metadata fields
    text_reading_mds = dict()
    for ix, mds in enumerate(gromet_fn_module["metadata_collection"]):
        tr_mds = [md for md in mds if md["metadata_type"].startswith("text_")]
        if len(tr_mds) > 0:
            text_reading_mds[ix] = tr_mds

    # Now, get the ports with TR metadata associated
    ports = list(
        walk_nested_structure(gromet_fn_module, text_reading_mds.keys())
    )

    if args.print_aligned:
        # Print the "table"
        for port in ports:
            pprint(port, indent=4)
            print()
            for md in text_reading_mds[port["metadata"]]:
                pprint(md, indent=4)
                print()

            pprint("=================================================")
            print()

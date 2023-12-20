import argparse
from skema.program_analysis.JSON2GroMEt import json2gromet
from skema.gromet.metadata import SourceCodeReference

# Ways to expand
# Check loop, condition FN indices
# Check bf call FN indices
# Boxes associated with ports

def disp_wire(wire):
    return f"src:{wire.src}<-->tgt:{wire.tgt}"

def get_length(gromet_item):
    # For any gromet object we can generically retrieve the length, since they all exist
    # in lists
    return len(gromet_item) if gromet_item != None else 0

def check_wire(gromet_wire, src_port_count, tgt_port_count, wire_type = "", metadata=None):
    # The current wiring checks are
    # Checking if the ports on both ends of the wire are below or over the bounds
    error_detected = False
    if gromet_wire.src < 0:
        error_detected = True
        print(f"Gromet Wire {wire_type} {disp_wire(gromet_wire)} has negative src port.")
    if gromet_wire.src == 0:
        error_detected = True
        print(f"Gromet Wire {wire_type} {disp_wire(gromet_wire)} has zero src port.")
    if gromet_wire.src > src_port_count:
        error_detected = True
        print(f"Gromet Wire {wire_type} {disp_wire(gromet_wire)} has a src port that goes over the boundary of {src_port_count} src ports.")

    if gromet_wire.tgt < 0:
        error_detected = True
        print(f"Gromet Wire {wire_type} {disp_wire(gromet_wire)} has negative tgt port.")
    if gromet_wire.tgt == 0:
        error_detected = True
        print(f"Gromet Wire {wire_type} {disp_wire(gromet_wire)} has zero tgt port.")
    if gromet_wire.tgt > tgt_port_count:
        error_detected = True
        print(f"Gromet Wire {wire_type} {disp_wire(gromet_wire)} has a tgt port that goes over the boundary of {tgt_port_count} tgt ports.")


    if error_detected:
        if metadata == None:
            print("No line number information exists for this particular wire!")
        else:
            print(f"Wire is associated with source code lines start:{metadata.line_begin} end:{metadata.line_end}")
        print()

    return error_detected

def find_metadata_idx(gromet_fn):
    """
        Attempts to find a metadata associated with this fn
        If it finds something, return it, otherwise return None
    """
    if gromet_fn.b != None:
        for b in gromet_fn.b:
            if b.metadata != None:
                return b.metadata

    if gromet_fn.bf != None:
        for bf in gromet_fn.bf:
            if bf.metadata != None:
                return bf.metadata

    return None

def analyze_fn_wiring(gromet_fn, metadata_collection):
    # Acquire information for all the ports, if they exist
    pif_length = get_length(gromet_fn.pif)
    pof_length = get_length(gromet_fn.pof)
    opi_length = get_length(gromet_fn.opi)
    opo_length = get_length(gromet_fn.opo)
    pil_length = get_length(gromet_fn.pil)
    pol_length = get_length(gromet_fn.pol)
    pic_length = get_length(gromet_fn.pic)
    poc_length = get_length(gromet_fn.poc)
    
    # Find a SourceCodeReference metadata that we can extract line number information for
    # so we can display some line number information about potential errors in the wiring 
    # NOTE: Can we make this extraction more accurate?
    metadata_idx = find_metadata_idx(gromet_fn)
    metadata = None
    if metadata_idx != None:
        for md in metadata_collection[metadata_idx - 1]:
            if isinstance(md, SourceCodeReference):
                metadata = md

    wopio_length = get_length(gromet_fn.wopio)
    if wopio_length > 0:
        for wire in gromet_fn.wff:
            check_wire(wire, opo_length, opi_length, "wff", metadata)

    ######################## loop (bl) wiring

    wlopi_length = get_length(gromet_fn.wlopi)
    if wlopi_length > 0:
        for wire in gromet_fn.wlopi:
            check_wire(wire, pil_length, opi_length, "wlopi", metadata)

    wll_length = get_length(gromet_fn.wll)
    if wll_length > 0:
        for wire in gromet_fn.wll:
            check_wire(wire, pil_length, pol_length, "wll", metadata)

    wlf_length = get_length(gromet_fn.wlf)
    if wlf_length > 0:
        for wire in gromet_fn.wlf:
            check_wire(wire, pif_length, pol_length, "wlf", metadata)
    
    wlc_length = get_length(gromet_fn.wlc)
    if wlc_length > 0:
        for wire in gromet_fn.wlc:
            check_wire(wire, pic_length, pol_length, "wlc", metadata)

    wlopo_length = get_length(gromet_fn.wlopo)
    if wlopo_length > 0:
        for wire in gromet_fn.wlopo:
            check_wire(wire, opo_length, pol_length, "wlopo", metadata)

    ######################## function (bf) wiring
    wfopi_length = get_length(gromet_fn.wfopi)
    if wfopi_length > 0:
        for wire in gromet_fn.wfopi:
            check_wire(wire, pif_length, opi_length, "wfopi", metadata)

    wfl_length = get_length(gromet_fn.wfl)
    if wfl_length > 0:
        for wire in gromet_fn.wfl:
            check_wire(wire, pil_length, pof_length, "wfl", metadata)

    wff_length = get_length(gromet_fn.wff)
    if wff_length > 0:
        for wire in gromet_fn.wff:
            check_wire(wire, pif_length, pof_length, "wff", metadata)
    
    wfc_length = get_length(gromet_fn.wfc)
    if wfc_length > 0:
        for wire in gromet_fn.wfc:
            check_wire(wire, pic_length, pof_length, "wfc", metadata)

    wfopo_length = get_length(gromet_fn.wfopo)
    if wfopo_length > 0:
        for wire in gromet_fn.wfopo:
            check_wire(wire, opo_length, pof_length, "wfopo", metadata)

    ######################## condition (bc) wiring
    wcopi_length = get_length(gromet_fn.wcopi)
    if wcopi_length > 0:
        for wire in gromet_fn.wcopi:
            check_wire(wire, pic_length, opi_length, "wcopi", metadata)

    wcl_length = get_length(gromet_fn.wcl)
    if wcl_length > 0:
        for wire in gromet_fn.wcl:
            check_wire(wire, pil_length, poc_length, "wcl", metadata)

    wcf_length = get_length(gromet_fn.wcf)
    if wcf_length > 0:
        for wire in gromet_fn.wcf:
            check_wire(wire, pif_length, poc_length, "wcf", metadata)
    
    wcc_length = get_length(gromet_fn.wcc)
    if wcc_length > 0:
        for wire in gromet_fn.wcc:
            check_wire(wire, pic_length, poc_length, "wcc", metadata)

    wcopo_length = get_length(gromet_fn.wcopo)
    if wcopo_length > 0:
        for wire in gromet_fn.wcopo:
            check_wire(wire, opo_length, poc_length, "wcopo", metadata)


def wiring_analyzer(gromet_obj):
    # TODO: Multifiles

    for module in gromet_obj.modules:
        # first_module = gromet_obj.modules[0]
        metadata = []
        # Analyze base FN
        print(f"Analyzing {module.name}")
        analyze_fn_wiring(module.fn, module.metadata_collection)

        # Analyze the rest of the FN_array
        for fn in module.fn_array:
            analyze_fn_wiring(fn, module.metadata_collection) 

def get_args():
    parser = argparse.ArgumentParser(
        "Attempts to analyize GroMEt JSON for issues"
    )
    parser.add_argument(
        "gromet_file_path",
        help="input GroMEt JSON file"
    )

    options = parser.parse_args()
    return options

if __name__ == "__main__":
    args = get_args()
    gromet_obj = json2gromet.json_to_gromet(args.gromet_file_path)

    wiring_analyzer(gromet_obj)

    


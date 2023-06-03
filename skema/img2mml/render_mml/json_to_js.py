def json2js(json_data, output_file, var_name="eqn_src"):
    """
    Helper function to write 'json' version of latex source data as a javascript list of dicts.
    Use json.load() to read the json into python object.
    Args:
        json_data: Assumed format: [ {"src": <string>, "mml": <string>}, ... ]
        output_file: Path to .js output file
        var_name: Name of the variable being assigned the list of dicts in the .js file
    Returns:
    """
    with open(output_file, "w") as fout:
        fout.write(f"{var_name} = [\n")
        for i, datum in enumerate(json_data):
            fout.write("  {\n")
            fout.write(f'    mml1: {repr(datum["mml1"])},\n')
            fout.write(f'    mml2: {repr(datum["mml2"])}\n')
            fout.write("  }")
            if i < len(json_data):
                fout.write(",")
            fout.write("\n")
        fout.write("];")

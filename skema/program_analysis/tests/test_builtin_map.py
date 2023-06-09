from skema.program_analysis.PyAST2CAST.builtin_map import(
    build_map,
    dump_map
)

def test_primitive_map():
    build_result = build_map()
    dump_result = dump_map()

    assert (build_result and dump_result)


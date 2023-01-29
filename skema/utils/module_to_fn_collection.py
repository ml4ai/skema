from skema.gromet.fn import (
    GrometFNModuleCollection,
)

def module_to_fn_collection(gromet_module, sysname=""):
    """ Given a single GroMEt FN module, this function
        puts a 'GrometFNModuleCollection' shell around it
        in order to effectively support multi module GroMEt systems
        as intended from v0.1.5 and on
    """

    module_collection = GrometFNModuleCollection(
            schema_version="0.1.5",
            name=sysname,
            modules=[gromet_module],
            module_index=[],
            executables=[],
    )

    return module_collection
    
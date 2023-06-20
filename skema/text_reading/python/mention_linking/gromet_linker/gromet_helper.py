# This is for basic gromet functions that require two or more Gromet objects.

from skema.gromet.fn import GrometFNModule
from skema.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet


class GrometHelper:
    empty_range = range(0, 0)

    @staticmethod
    def json_to_gromet(gromet_path: str) -> GrometFNModule:
        return json_to_gromet(gromet_path)

    @staticmethod
    def get_element_metadata(gromet_object, gromet_fn_module: GrometFNModule):
        if gromet_object.metadata:
            return gromet_fn_module.metadata_collection[
                gromet_object.metadata - 1
            ]  # minus 1 because it is 1-based indexing
        return None

    @staticmethod
    def get_element_line_numbers(
        gromet_object, gromet_fn_module: GrometFNModule
    ) -> range:
        metadata = GrometHelper.get_element_metadata(
            gromet_object, gromet_fn_module
        )

        if metadata:
            for metadatum in metadata:
                if metadatum.metadata_type == "source_code_reference":
                    return range(
                        metadatum.line_begin, metadatum.line_end + 1
                    )  # [beg, end)
        return GrometHelper.empty_range

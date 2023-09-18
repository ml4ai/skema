from copy import deepcopy
import sys
import os.path
import pprint

from skema.utils.misc import uuid

from functools import singledispatchmethod
from datetime import datetime
from time import time

from skema.program_analysis.CAST2FN.model.cast import StructureType

from skema.gromet.fn import (
    FunctionType,
    GrometBoxConditional,
    GrometBoxFunction,
    GrometBoxLoop,
    GrometFNModule,
    GrometFN,
    GrometPort,
    GrometWire,
    ImportType,
    LiteralValue as GLiteralValue,
    TypedValue,
)

from skema.gromet.metadata import (
    Provenance,
    SourceCodeDataType,
    SourceCodeReference,
    SourceCodeCollection,
    SourceCodePortDefaultVal,
    CodeFileReference,
    GrometCreation,
    ProgramAnalysisRecordBookkeeping,
    SourceCodeBoolAnd,
    SourceCodeBoolOr,
)

from skema.program_analysis.CAST.pythonAST.builtin_map import (
    build_map,
    dump_map,
    check_builtin,
)
from skema.program_analysis.CAST2FN.model.cast.scalar_type import ScalarType

from skema.program_analysis.CAST2FN.ann_cast.annotated_cast import *
from skema.program_analysis.CAST.pythonAST.modules_list import (
    BUILTINS,
    find_func_in_module,
    find_std_lib_module,
)

from skema.gromet.execution_engine.primitive_map import (
    get_shorthand,
    get_inputs,
    get_outputs,
    is_primitive,
)

PYTHON_VERSION = "3.8"


def is_inline(func_name):
    # Tells us which functions should be inlined in GroMEt (i.e. don't make GroMEt FNs for these)
    return func_name == "iter" or func_name == "next" or func_name == "range"


def insert_gromet_object(t: list, obj):
    """Inserts a GroMEt object obj into a GroMEt table t
    Where obj can be
        - A GroMEt Box
        - A GroMEt Port
        - A GroMEt Wire
    And t can be
        - A list of GroMEt Boxes
        - A list of GroMEt ports
        - A list of GroMEt wires

    If the table we're trying to insert into doesn't already exist, then we
    first create it, and then insert the value.
    """

    # Logic for generating port ids
    if isinstance(obj, GrometPort):
        if t == None:
            obj.id = 1
        else:
            current_box = obj.box
            current_box_ports = [port for port in t if port.box == current_box]
            obj.id = len(current_box_ports) + 1

    if t == None:
        t = []
    t.append(obj)

    return t


def generate_provenance():
    timestamp = str(datetime.fromtimestamp(time()))
    method_name = "skema_code2fn_program_analysis"
    return Provenance(method=method_name, timestamp=timestamp)


def is_tuple(node):
    # Checks if an AnnCast Node is a Tuple LiteralValue
    return (
        isinstance(node, AnnCastLiteralValue)
        and node.value_type == StructureType.TUPLE
    )


def retrieve_name_id_pair(node):
    """
    Operand from an AnnCastOperator
        AnnCastName
        AnnCastCall
        AnnCastAttribute
    """

    if isinstance(node, AnnCastOperator):
        return retrieve_name_id_pair(node.operands[0])
    if isinstance(node, AnnCastName):
        return (node.name, node.id)
    if isinstance(node, AnnCastAttribute):
        if isinstance(
            node.value, (AnnCastAttribute, AnnCastName, AnnCastCall)
        ):
            return retrieve_name_id_pair(node.value)
        return (node.attr.name, node.attr.id)
    if isinstance(node, AnnCastCall):
        return retrieve_name_id_pair(node.func)
    return ("", -1)


def comp_name_nodes(n1, n2):
    """Given two AnnCast nodes we compare their name
    and ids to see if they reference the same name
    """
    # If n1 or n2 is not a Name or an Operator node
    if (
        not isinstance(n1, AnnCastName)
        and not isinstance(n1, AnnCastOperator)
        and not isinstance(n1, AnnCastAttribute)
    ):
        return False
    if (
        not isinstance(n2, AnnCastName)
        and not isinstance(n2, AnnCastOperator)
        and not isinstance(n2, AnnCastAttribute)
    ):
        return False
    # LiteralValues can't have 'names' compared
    if isinstance(n1, AnnCastLiteralValue) or isinstance(
        n2, AnnCastLiteralValue
    ):
        return False

    n1_name, n1_id = retrieve_name_id_pair(n1)
    n2_name, n2_id = retrieve_name_id_pair(n2)

    return n1_name == n2_name and n1_id == n2_id


def find_existing_opi(gromet_fn, opi_name):
    idx = 1
    if gromet_fn.opi == None:
        return False, idx

    for opi in gromet_fn.opi:
        if opi_name == opi.name:
            return True, idx
        idx += 1
    return False, idx


def find_existing_pil(gromet_fn, opi_name):
    if gromet_fn.pil == None:
        return -1

    idx = 1
    for pil in gromet_fn.pil:
        if opi_name == pil.name:
            return idx
        idx += 1
    return -1


def get_left_side_name(node):
    if isinstance(node, AnnCastAttribute):
        return node.attr.name
    if isinstance(node, AnnCastName):
        return node.name
    if isinstance(node, AnnCastVar):
        return get_left_side_name(node.val)
    if isinstance(node, AnnCastCall):
        return node.func.name
    return "NO LEFT SIDE NAME"


def get_attribute_name(node):
    """
    Given an AnnCastAttribute node
    """
    if isinstance(node, AnnCastName):
        return str(node.name)
    if isinstance(node, AnnCastAttribute):
        return get_attribute_name(node.value) + "." + str(node.attr)
    if isinstance(node, AnnCastCall):
        return get_attribute_name(node.func)


def get_func_name(node: AnnCastCall):
    if isinstance(node.func, AnnCastName):
        return (node.func.name, f"{node.func.name}_id{node.func.id}")
    if isinstance(node.func, AnnCastAttribute):
        return (
            node.func.attr.name,
            f"{'.'.join(node.func.con_scope)}.{node.func.attr.name}_{node.invocation_index}",
        )
    if isinstance(node.func, str):
        return (node.func, f"{node.func}_id{node.func.id}")


class ToGrometPass:
    def __init__(self, pipeline_state: PipelineState):
        self.pipeline_state = pipeline_state
        self.nodes = self.pipeline_state.nodes

        self.var_environment = {"global": {}, "args": {}, "local": {}}
        self.symbol_table = {
            "functions": {},
            "variables": {"global": {}, "args": {}, "local": {}},
            "records": {},
        }
        # Attribute accesses check this collection
        # to see if we're using an imported item
        # Function calls to imported functions without their attributes will also check here
        self.import_collection = {}

        # creating a GroMEt FN object here or a collection of GroMEt FNs
        # generally, programs are complex, so a collection of GroMEt FNs is usually created
        # visiting nodes adds FNs
        self.gromet_module = GrometFNModule(
            schema="FN",
            schema_version="0.1.7",
            name="",
            fn=None,
            fn_array=[],
            metadata_collection=[],
        )

        # build the built-in map
        build_map()

        # Everytime we see an AnnCastRecordDef we can store information for it
        # for example the name of the class and indices to its functions
        self.record = {}

        # When a record type is initiatied we keep track of its name and record type here
        self.initialized_records = {}

        # Initialize the table of function arguments
        self.function_arguments = {}

        # the fullid of a AnnCastName node is a string which includes its
        # variable name, numerical id, version, and scope
        for node in self.pipeline_state.nodes:
            self.visit(node, parent_gromet_fn=None, parent_cast_node=None)

        pipeline_state.gromet_collection = self.gromet_module

    def symtab_variables(self):
        return self.symbol_table["variables"]

    def symtab_functions(self):
        return self.symbol_table["functions"]

    def symtab_records(self):
        return self.symbol_table["records"]

    def build_function_arguments_table(self, nodes):
        """Iterates through all the function definitions at the module
        level and creates a table that maps their function names to a map
        of its arguments with position values

        NOTE: functions within functions aren't currently supported

        """
        for node in nodes:
            if isinstance(node, AnnCastFunctionDef):
                self.function_arguments[node.name.name] = {}
                for i, arg in enumerate(node.func_args, 1):
                    self.function_arguments[node.name.name][arg.val.name] = i
                self.symbol_table["functions"][node.name.name] = node.name.name

    def wire_from_var_env(self, name, gromet_fn):
        var_environment = self.symtab_variables()

        if name in var_environment["local"]:
            local_env = var_environment["local"]
            entry = local_env[name]
            if isinstance(entry[0], AnnCastLoop):
                gromet_fn.wlf = insert_gromet_object(
                    gromet_fn.wlf,
                    GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
                )
            if isinstance(entry[0], AnnCastModelIf):
                gromet_fn.wfopi = insert_gromet_object(
                    gromet_fn.wfopi,
                    GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
                )
            else:
                gromet_fn.wff = insert_gromet_object(
                    gromet_fn.wff,
                    GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
                )
        elif name in var_environment["args"]:
            args_env = var_environment["args"]
            entry = args_env[name]
            gromet_fn.wfopi = insert_gromet_object(
                gromet_fn.wfopi,
                GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
            )
        elif name in var_environment["global"]:
            global_env = var_environment["global"]
            entry = global_env[name]
            gromet_fn.wff = insert_gromet_object(
                gromet_fn.wff,
                GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
            )

    def create_source_code_reference(self, ref_info):
        # return None # comment this when we want metadata
        if ref_info == None:
            return None

        line_begin = ref_info.row_start
        line_end = ref_info.row_end
        col_begin = ref_info.col_start
        col_end = ref_info.col_end

        # file_uid = str(self.gromet_module.metadata[-1].files[0].uid)
        file_uid = str(
            self.gromet_module.metadata_collection[1][0].files[0].uid
        )
        # file_uid = ""
        return SourceCodeReference(
            provenance=generate_provenance(),
            code_file_reference_uid=file_uid,
            line_begin=line_begin,
            line_end=line_end,
            col_begin=col_begin,
            col_end=col_end,
        )

    def insert_metadata(self, *metadata):
        """
        insert_metadata inserts metadata into the self.gromet_module.metadata_collection list
        Then, the index of where this metadata lives is returned
        The idea is that all GroMEt objects that store metadata will store an index
        into metadata_collection that points to the metadata they stored
        """
        # return None # Uncomment this line if we don't want metadata
        to_insert = []
        for md in metadata:
            to_insert.append(md)
        self.gromet_module.metadata_collection.append(to_insert)
        return len(self.gromet_module.metadata_collection)

    def insert_record_info(self, metadata: ProgramAnalysisRecordBookkeeping):
        """
        insert_record_info inserts a ProgramAnalysisRecordBookkeping metadata
        into the metadata table
        All metadata of this kind lives in the first index of the entire collection
        """
        self.gromet_module.metadata_collection[0].append(metadata)

    def set_index(self):
        """Called after a Gromet FN is added to the whole collection
        Properly sets the index of the Gromet FN that was just added
        """
        return
        idx = len(self.gromet_module.fn_array)
        self.gromet_module._fn_array[-1].index = idx

    def handle_primitive_function(
        self,
        node: AnnCastCall,
        parent_gromet_fn,
        parent_cast_node,
        from_assignment,
    ):
        """Creates an Expression GroMEt FN for the primitive function stored in node.
        Then it gets wired up to its parent_gromet_fn appropriately
        """
        ref = node.source_refs[0]
        metadata = self.create_source_code_reference(ref)

        func_name, qual_func_name = get_func_name(node)

        # primitives that come from something other than an assignment or functions designated to be inlined at all times have
        # special semantics in that they're inlined as opposed to creating their own GroMEt FNs
        if from_assignment or is_inline(func_name):
            inline_func_bf = GrometBoxFunction(
                name=func_name, function_type=FunctionType.LANGUAGE_PRIMITIVE
            )
            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf, inline_func_bf
            )
            inline_bf_loc = len(parent_gromet_fn.bf)

            return inline_bf_loc
        else:
            # Create the Expression FN and its box function
            primitive_fn = GrometFN()
            primitive_fn.b = insert_gromet_object(
                primitive_fn.b,
                GrometBoxFunction(
                    function_type=FunctionType.EXPRESSION,
                    metadata=self.insert_metadata(metadata),
                ),
            )

            # Create the primitive expression bf
            primitive_func_bf = GrometBoxFunction(
                name=func_name, function_type=FunctionType.LANGUAGE_PRIMITIVE
            )
            primitive_fn.bf = insert_gromet_object(
                primitive_fn.bf, primitive_func_bf
            )
            primitive_bf_loc = len(primitive_fn.bf)

            primitive_fn.opo = insert_gromet_object(
                primitive_fn.opo, GrometPort(box=len(primitive_fn.b))
            )

            # Write its pof and wire it to its opo
            primitive_fn.pof = insert_gromet_object(
                primitive_fn.pof, GrometPort(box=len(primitive_fn.bf))
            )
            primitive_fn.wfopo = insert_gromet_object(
                primitive_fn.wfopo,
                GrometWire(
                    src=len(primitive_fn.opo), tgt=len(primitive_fn.pof)
                ),
            )

            # Create FN's opi and and opo
            for arg in node.arguments:
                if (
                    isinstance(arg, AnnCastOperator)
                    or isinstance(arg, AnnCastLiteralValue)
                    or isinstance(arg, AnnCastCall)
                ):
                    self.visit(arg, primitive_fn, parent_cast_node)
                    primitive_fn.pif = insert_gromet_object(
                        primitive_fn.pif, GrometPort(box=primitive_bf_loc)
                    )
                    primitive_fn.wff = insert_gromet_object(
                        primitive_fn.wff,
                        GrometWire(
                            src=len(primitive_fn.pif),
                            tgt=len(primitive_fn.pof),
                        ),
                    )
                else:
                    primitive_fn.opi = insert_gromet_object(
                        primitive_fn.opi, GrometPort(box=len(primitive_fn.b))
                    )
                    primitive_fn.pif = insert_gromet_object(
                        primitive_fn.pif, GrometPort(box=primitive_bf_loc)
                    )
                    primitive_fn.wfopi = insert_gromet_object(
                        primitive_fn.wfopi,
                        GrometWire(
                            src=len(primitive_fn.pif),
                            tgt=len(primitive_fn.opi),
                        ),
                    )

            # Insert it into the overall Gromet FN collection
            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array,
                primitive_fn,
            )
            self.set_index()

            ref = node.source_refs[0]
            metadata = self.create_source_code_reference(ref)
            # Creates the 'call' to this primitive expression which then gets inserted into the parent's Gromet FN
            parent_primitive_call_bf = GrometBoxFunction(
                function_type=FunctionType.EXPRESSION,
                body=len(self.gromet_module.fn_array),
                metadata=self.insert_metadata(metadata),
            )

            # We create the arguments of the primitive expression call here and then
            # We must wire the arguments of this primitive expression appropriately
            # We have an extra check to see if the local came from a Loop, in which
            # case we use a wlf wire to wire the pol to the pif

            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf, parent_primitive_call_bf
            )
            return len(parent_gromet_fn.bf)

    def add_var_to_env(
        self, var_name, var_cast, var_pof, var_pof_idx, parent_cast_node
    ):
        """Adds a variable with name var_name, CAST node var_cast, Gromet pof var_pof
        and pof index var_pof_idx to the overall variable environment.
        This addition to the environment happens in these conditions
            - An assignment at the global (module) level
            - An assignment at the local (function def) level
            - When visiting a function argument (This is done at the function def visitor)
        This environment is used when a reference to a variable and its pof is
        needed in Gromet, this is mostly used when creating wires between outputs
        and inputs
        parent_cast_node allows us to determine if this variable exists within
        """
        var_environment = self.symtab_variables()

        if isinstance(parent_cast_node, AnnCastModule):
            global_env = var_environment["global"]
            global_env[var_name] = (var_cast, var_pof, var_pof_idx)
        elif (
            isinstance(parent_cast_node, AnnCastFunctionDef)
            or isinstance(parent_cast_node, AnnCastModelIf)
            or isinstance(parent_cast_node, AnnCastLoop)
        ):
            local_env = var_environment["local"]
            local_env[var_name] = (parent_cast_node, var_pof, var_pof_idx)
        # else:
        # print(f"error: add_var_to_env: we came from{type(parent_cast_node)}")
        # sys.exit()

    def find_gromet(self, func_name):
        """Attempts to find func_name in self.gromet_module.fn_array
        and will return the index of where it is if it finds it.
        It checks if the attribute is a GroMEt FN.
        It will also return a boolean stating whether or not it found it.
        If it doesn't find it, the func_idx then represents the index at
        the end of the self.gromet_module.fn_array collection.
        """
        func_idx = 0
        found_func = False
        for attribute in self.gromet_module.fn_array:
            gromet_fn = attribute
            if gromet_fn.b != None:
                gromet_fn_b = gromet_fn.b[0]
                if gromet_fn_b.name == func_name:
                    found_func = True
                    break

            func_idx += 1

        return func_idx + 1, found_func

    def retrieve_var_port(self, var_name):
        """Given a variable named var_name in the variable environment
        This function attempts to look up the port in which it's located
        """
        var_environment = self.symtab_variables()
        if var_name in var_environment["local"]:
            local_env = var_environment["local"]
            entry = local_env[var_name]
            return entry[2]
        elif var_name in var_environment["args"]:
            args_env = var_environment["args"]
            entry = args_env[var_name]
            return entry[2]
        elif var_name in var_environment["global"]:
            global_env = var_environment["global"]
            entry = global_env[var_name]
            return entry[2]

        return -1

    def check_var_location(self, var_name, env):
        # Given the name of a variable and the name of an environment,
        # check if that variable is in that environment
        var_environment = self.symtab_variables()
        return var_name in var_environment[env]

    def visit(self, node: AnnCastNode, parent_gromet_fn, parent_cast_node):
        """
        External visit that callsthe internal visit
        Useful for debugging/development.  For example,
        printing the nodes that are visited
        """
        # print current node being visited.
        # this can be useful for debugging
        # class_name = node.__class__.__name__
        # print(f"\nProcessing node type {class_name}")

        # call internal visit
        try:
            return self._visit(node, parent_gromet_fn, parent_cast_node)
        except Exception as e:
            print(
                f"Error in visitor for {type(node)} which has source ref information {node.source_refs}"
            )
            raise e

    def visit_node_list(
        self,
        node_list: typing.List[AnnCastNode],
        parent_gromet_fn,
        parent_cast_node,
    ):
        return [
            self.visit(node, parent_gromet_fn, parent_cast_node)
            for node in node_list
        ]

    @singledispatchmethod
    def _visit(self, node: AnnCastNode, parent_gromet_fn, parent_cast_node):
        """
        Internal visit
        """
        raise NameError(f"Unrecognized node type: {type(node)}")

    # This creates 'expression' GroMEt FNs (i.e. new big standalone colored boxes in the diagram)
    # - The expression on the right hand side of an assignment
    #     - This could be as simple as a LiteralValue (like the number 2)
    #     - It could be a binary expression (like 2 + 3)
    #     - It could be a function call (foo(2))

    def unpack_create_collection_pofs(
        self, tuple_values, parent_gromet_fn, parent_cast_node
    ):
        """When we encounter a case where a tuple has a tuple (or list) inside of it
        we call this helper function to appropriately unpack it and create its pofs
        """
        for elem in tuple_values:
            if isinstance(elem, AnnCastLiteralValue):
                self.unpack_create_collection_pofs(
                    elem.value, parent_gromet_fn, parent_cast_node
                )
            else:
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.val.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.val.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )

    def create_pack(
        self, var, tuple_values, parent_gromet_fn, parent_cast_node
    ):
        """Creates a 'pack' primitive whenever the left hand side
        of a tuple assignment is a single variable, such as:
        x = a,b,c...
        """
        # Make the "pack" literal and insert it in the GroMEt FN
        # TODO: a better way to get the name of this 'pack'
        pack_bf = GrometBoxFunction(
            name="pack", function_type=FunctionType.ABSTRACT
        )

        parent_gromet_fn.bf = insert_gromet_object(
            parent_gromet_fn.bf, pack_bf
        )

        pack_index = len(parent_gromet_fn.bf)

        # Construct the pifs for the pack and wire them
        for port in tuple_values:
            parent_gromet_fn.pif = insert_gromet_object(
                parent_gromet_fn.pif, GrometPort(box=pack_index)
            )

            parent_gromet_fn.wff = insert_gromet_object(
                parent_gromet_fn.wff,
                GrometWire(src=len(parent_gromet_fn.pif), tgt=port),
            )

        # Insert the return value of the pack
        # Which is one variable
        parent_gromet_fn.pof = insert_gromet_object(
            parent_gromet_fn.pof,
            GrometPort(name=get_left_side_name(var), box=pack_index),
        )

        self.add_var_to_env(
            get_left_side_name(var),
            var,
            parent_gromet_fn.pof[-1],
            len(parent_gromet_fn.pof),
            parent_cast_node,
        )

    def create_unpack(self, tuple_values, parent_gromet_fn, parent_cast_node):
        """Creates an 'unpack' primitive whenever the left hand side
        of an assignment is a tuple. Example:
        x,y,z = foo(...)
        Then, an unpack with x,y,z as pofs is created and a pif connecting to the return value of
        foo() is created
        """
        parent_gromet_fn.pof = insert_gromet_object(
            parent_gromet_fn.pof, GrometPort(box=len(parent_gromet_fn.bf))
        )

        # Make the "unpack" literal here
        # And wire it appropriately
        unpack_bf = GrometBoxFunction(
            name="unpack", function_type=FunctionType.ABSTRACT
        )  # TODO: a better way to get the name of this 'unpack'
        parent_gromet_fn.bf = insert_gromet_object(
            parent_gromet_fn.bf, unpack_bf
        )

        # Make its pif so that it takes the return value of the function call
        parent_gromet_fn.pif = insert_gromet_object(
            parent_gromet_fn.pif, GrometPort(box=len(parent_gromet_fn.bf))
        )

        # Wire the pif to the function call's pof
        parent_gromet_fn.wff = insert_gromet_object(
            parent_gromet_fn.wff,
            GrometWire(
                src=len(parent_gromet_fn.pif), tgt=len(parent_gromet_fn.pof)
            ),
        )

        for elem in tuple_values:
            if isinstance(elem, AnnCastLiteralValue):
                self.unpack_create_collection_pofs(
                    elem.value, parent_gromet_fn, parent_cast_node
                )
            elif isinstance(elem, AnnCastCall):
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.func.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.func.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )
            elif isinstance(elem, AnnCastAttribute):
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.attr.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.attr.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )
            else:
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.val.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.val.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )

    def create_implicit_unpack(
        self, tuple_values, parent_gromet_fn, parent_cast_node
    ):
        """
        In some cases, we need to unpack a tuple without using an 'unpack' primitive
        In this case, we directly attach the pofs to the FN instead of going through
        and 'unpack'
        """

        for elem in tuple_values:
            if isinstance(elem, AnnCastLiteralValue):
                self.unpack_create_collection_pofs(
                    elem.value, parent_gromet_fn, parent_cast_node
                )
            elif isinstance(elem, AnnCastCall):
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.func.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.func.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )
            elif isinstance(elem, AnnCastAttribute):
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.attr.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.attr.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )
            else:
                ref = elem.source_refs[0]
                metadata = self.create_source_code_reference(ref)
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=elem.val.name,
                        box=len(parent_gromet_fn.bf),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                pof_idx = len(parent_gromet_fn.pof)
                self.add_var_to_env(
                    elem.val.name,
                    elem,
                    parent_gromet_fn.pof[pof_idx - 1],
                    pof_idx,
                    parent_cast_node,
                )

    def determine_func_type(self, node):
        """
        Determines what kind of function this Call or Attribute node is referring to
        Potential options
        - ABSTRACT
        - LANGUAGE_PRIMITIVE
        - IMPORTED
            - GROMET_FN_MODULE
            - NATIVE
            - OTHER
        - IMPORTED_METHOD
        - UNKNOWN_METHOD

        Return a tuple of
        (FunctionType, ImportType, ImportVersion, ImportSource, SourceLanguage, SourceLanguageVersion)
        """
        func_name, _ = retrieve_name_id_pair(node)

        # print(f"Checking {func_name}...")
        if is_primitive(func_name, "Python"):
            # print(f"{func_name} is a primitive GroMEt function")
            return (FunctionType.ABSTRACT, None, None, None, None, None)

        if isinstance(node, AnnCastCall):
            if func_name in BUILTINS or check_builtin(func_name):
                # print(f"{func_name} is a python builtin")
                if isinstance(node.func, AnnCastAttribute):
                    attr_node = node.func
                    if func_name in self.import_collection:
                        # print(f"Module {func_name} has imported function {attr_node.attr.name}")
                        return (
                            FunctionType.IMPORTED,
                            ImportType.NATIVE,
                            None,
                            None,
                            "Python",
                            PYTHON_VERSION,
                        )

                return (
                    FunctionType.LANGUAGE_PRIMITIVE,
                    None,
                    None,
                    None,
                    "Python",
                    PYTHON_VERSION,
                )
            if isinstance(node.func, AnnCastAttribute):
                attr_node = node.func
                if func_name in self.import_collection:
                    # print(f"Module {func_name} has imported function {attr_node.attr.name}")
                    # Check if it's gromet_fn_module/native/other
                    # TODO: import_version/import_source
                    return (
                        FunctionType.IMPORTED,
                        ImportType.OTHER,
                        None,
                        None,
                        "Python",
                        PYTHON_VERSION,
                    )
            else:
                return (
                    FunctionType.IMPORTED_METHOD,
                    ImportType.OTHER,
                    None,
                    None,
                    "Python",
                    PYTHON_VERSION,
                )
        elif isinstance(node, AnnCastAttribute):
            if func_name in BUILTINS or check_builtin(func_name):
                # print(f"{func_name} is a python builtin")
                if func_name in self.import_collection:
                    # print(f"Module {func_name} has imported function {node.attr.name}")
                    return (
                        FunctionType.IMPORTED,
                        ImportType.NATIVE,
                        None,
                        None,
                        "Python",
                        PYTHON_VERSION,
                    )

                return (
                    FunctionType.LANGUAGE_PRIMITIVE,
                    None,
                    None,
                    None,
                    "Python",
                    PYTHON_VERSION,
                )
            elif func_name in self.import_collection:
                # print(f"Module {func_name} has imported function {node.attr.name}")
                # Check if it's gromet_fn_module/native/other
                # TODO: import_version/import_source
                return (
                    FunctionType.IMPORTED,
                    ImportType.OTHER,
                    None,
                    None,
                    "Python",
                    PYTHON_VERSION,
                )
            # Attribute of a class we don't have access to
            else:
                # print(self.import_collection)
                return (
                    FunctionType.IMPORTED_METHOD,
                    ImportType.OTHER,
                    None,
                    None,
                    "Python",
                    PYTHON_VERSION,
                )

    @_visit.register
    def visit_assignment(
        self, node: AnnCastAssignment, parent_gromet_fn, parent_cast_node
    ):
        # How does this creation of a GrometBoxFunction object play into the overall construction?
        # Where does it go?

        # This first visit on the node.right should create a FN
        # where the outer box is a GExpression (GroMEt Expression)
        # The purple box on the right in examples (exp0.py)
        # Because we don't know exactly what node.right holds at this time
        # we create the Gromet FN for the GExpression here

        # A function call creates a GroMEt FN at the scope of the
        # outer GroMEt FN box. In other words it's incorrect
        # to scope it to this assignment's Gromet FN
        if isinstance(node.right, AnnCastCall):
            # Assignment for
            # x = foo(...)
            # x = a.foo(...)
            # x,y,z = foo(...)
            func_bf_idx = self.visit(node.right, parent_gromet_fn, node)
            # NOTE: x = foo(...) <- foo returns multiple values that get packed
            # Several conditions for this
            # - foo has multiple output ports for returning
            #    - multiple output ports but assignment to a single variable, then we introduce a pack
            #       the result of the pack is a single introduced variable that gets wired to the single
            #       variable
            #    - multiple output ports but assignment to multiple variables, then we wire one-to-one
            #       in order, all the output ports of foo to each variable
            #    - else, if we dont have a one to one matching then it's an error
            # - foo has a single output port to return a value
            #    - in the case of a single target variable, then we wire directly one-to-one
            #    - otherwise if multiple target variables for a single return output port, then it's an error

            # We've made the call box function, which made its argument box functions and wired them appropriately.
            # Now, we have to make the output(s) to this call's box function and have them be assigned appropriately.
            # We also add any variables that have been assigned in this AnnCastAssignment to the variable environment
            if not isinstance(
                node.right.func, AnnCastAttribute
            ) and not is_inline(node.right.func.name):
                # if isinstance(node.right.func, AnnCastName) and not is_inline(node.right.func.name):
                # if isinstance(node.left, AnnCastTuple):
                if is_tuple(node.left):
                    self.create_unpack(
                        node.left.value, parent_gromet_fn, parent_cast_node
                    )
                else:
                    if node.right.func.name in self.record.keys():
                        self.initialized_records[
                            node.left.val.name
                        ] = node.right.func.name

                    ref = node.left.source_refs[0]
                    metadata = self.create_source_code_reference(ref)
                    if func_bf_idx == None:
                        func_bf_idx = len(parent_gromet_fn.bf)
                    if isinstance(node.left, AnnCastAttribute):
                        self.add_var_to_env(
                            node.left.attr.name,
                            node.left,
                            parent_gromet_fn.pof[-1],
                            len(parent_gromet_fn.pof),
                            parent_cast_node,
                        )
                        parent_gromet_fn.pof[
                            len(parent_gromet_fn.pof) - 1
                        ].name = node.left.attr.name

                    elif isinstance(node.left.val, AnnCastAttribute):
                        self.add_var_to_env(
                            node.left.val.attr.name,
                            node.left,
                            parent_gromet_fn.pof[-1],
                            len(parent_gromet_fn.pof),
                            parent_cast_node,
                        )
                        parent_gromet_fn.pof[
                            len(parent_gromet_fn.pof) - 1
                        ].name = node.left.val.attr.name
                    else:
                        self.add_var_to_env(
                            get_left_side_name(node.left),
                            node.left,
                            parent_gromet_fn.pof[-1],
                            len(parent_gromet_fn.pof),
                            parent_cast_node,
                        )
                        parent_gromet_fn.pof[
                            len(parent_gromet_fn.pof) - 1
                        ].name = get_left_side_name(node.left)
            else:
                if is_tuple(node.left):
                    if (
                        isinstance(node.right.func, AnnCastName)
                        and node.right.func.name == "next"
                    ):
                        tuple_values = node.left.value
                        i = 2
                        pof_length = len(parent_gromet_fn.pof) - 1
                        for elem in tuple_values:
                            if isinstance(elem, AnnCastVar):
                                name = elem.val.name
                                parent_gromet_fn.pof[
                                    pof_length - i
                                ].name = name

                                self.add_var_to_env(
                                    name,
                                    elem,
                                    parent_gromet_fn.pof[pof_length - i],
                                    pof_length - i,
                                    parent_cast_node,
                                )
                                i -= 1
                            elif isinstance(elem, AnnCastLiteralValue):
                                name = elem.value[0].val.name
                                parent_gromet_fn.pof[
                                    pof_length - i
                                ].name = name

                                self.add_var_to_env(
                                    name,
                                    elem,
                                    parent_gromet_fn.pof[pof_length - i],
                                    pof_length - i,
                                    parent_cast_node,
                                )
                                i -= 1

                                # self.create_implicit_unpack(
                                #    node.left.value, parent_gromet_fn, parent_cast_node
                                # )

                        # self.create_implicit_unpack(
                        #    node.left.value, parent_gromet_fn, parent_cast_node
                        # )
                    else:
                        self.create_unpack(
                            node.left.value, parent_gromet_fn, parent_cast_node
                        )
                elif isinstance(node.right.func, AnnCastAttribute):
                    if (
                        parent_gromet_fn.pof == None
                    ):  # TODO: check this guard later
                        # print(node.source_refs[0])
                        if isinstance(node.left, AnnCastAttribute):
                            name = node.left.attr
                        else:
                            name = node.left.val.name
                        parent_gromet_fn.pof = insert_gromet_object(
                            parent_gromet_fn.pof,
                            GrometPort(name=name, box=-1),
                        )
                    else:
                        if isinstance(node.left, AnnCastAttribute):
                            parent_gromet_fn.pof = insert_gromet_object(
                                parent_gromet_fn.pof,
                                GrometPort(
                                    name=node.left.attr.name,
                                    box=len(parent_gromet_fn.pof),
                                ),
                            )
                            self.add_var_to_env(
                                node.left.attr.name,
                                node.left,
                                parent_gromet_fn.pof[-1],
                                len(parent_gromet_fn.pof),
                                parent_cast_node,
                            )
                        elif isinstance(node.left, AnnCastVar):
                            parent_gromet_fn.pof = insert_gromet_object(
                                parent_gromet_fn.pof,
                                GrometPort(
                                    name=node.left.val.name,
                                    box=len(parent_gromet_fn.bf),
                                ),
                            )
                            self.add_var_to_env(
                                node.left.val.name,
                                node.left,
                                parent_gromet_fn.pof[-1],
                                len(parent_gromet_fn.pof),
                                parent_cast_node,
                            )

                            if parent_gromet_fn.pif != None:
                                self.wire_from_var_env(
                                    node.left.val.name, parent_gromet_fn
                                )
                        else:
                            # NOTE: This case needs to eventually removed as this handler gets fleshed out more
                            parent_gromet_fn.pof[-1].name = node.left.val.name
                            self.add_var_to_env(
                                node.left.val.name,
                                node.left,
                                parent_gromet_fn.pof[-1],
                                len(parent_gromet_fn.pof),
                                parent_cast_node,
                            )

                            if parent_gromet_fn.pif != None:
                                self.wire_from_var_env(
                                    node.left.val.name, parent_gromet_fn
                                )
                else:
                    self.add_var_to_env(
                        node.left.val.name,
                        node.left,
                        parent_gromet_fn.pof[-1],
                        len(parent_gromet_fn.pof),
                        parent_cast_node,
                    )
                    parent_gromet_fn.pof[
                        len(parent_gromet_fn.pof) - 1
                    ].name = node.left.val.name

        elif isinstance(node.right, AnnCastName):
            # Assignment for
            # x = y
            # or some,set,of,values,... = y

            # Create a passthrough GroMEt
            new_gromet = GrometFN()
            new_gromet.b = insert_gromet_object(
                new_gromet.b,
                GrometBoxFunction(function_type=FunctionType.EXPRESSION),
            )
            new_gromet.opi = insert_gromet_object(
                new_gromet.opi, GrometPort(box=len(new_gromet.b))
            )
            new_gromet.opo = insert_gromet_object(
                new_gromet.opo, GrometPort(box=len(new_gromet.b))
            )
            new_gromet.wopio = insert_gromet_object(
                new_gromet.wopio,
                GrometWire(src=len(new_gromet.opo), tgt=len(new_gromet.opi)),
            )

            # Add it to the GroMEt collection
            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array, new_gromet
            )
            self.set_index()

            # Make it's 'call' expression in the parent gromet
            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf,
                GrometBoxFunction(
                    function_type=FunctionType.EXPRESSION,
                    body=len(self.gromet_module.fn_array),
                ),
            )

            parent_gromet_fn.pif = insert_gromet_object(
                parent_gromet_fn.pif, GrometPort(box=len(parent_gromet_fn.bf))
            )
            if isinstance(parent_gromet_fn.b[0], GrometBoxFunction) and (
                parent_gromet_fn.b[0].function_type == FunctionType.EXPRESSION
                or parent_gromet_fn.b[0].function_type
                == FunctionType.PREDICATE
            ):
                parent_gromet_fn.opi = insert_gromet_object(
                    parent_gromet_fn.opi,
                    GrometPort(
                        box=len(parent_gromet_fn.b), name=node.right.name
                    ),
                )

            self.wire_from_var_env(node.right.name, parent_gromet_fn)

            # if isinstance(node.left, AnnCastTuple): TODO: double check that this addition is correct
            if is_tuple(node.left):
                self.create_unpack(
                    node.left.value, parent_gromet_fn, parent_cast_node
                )
            else:
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=get_left_side_name(node.left),
                        box=len(parent_gromet_fn.bf),
                    ),
                )
                self.add_var_to_env(
                    get_left_side_name(node.left),
                    node.left,
                    parent_gromet_fn.pof[-1],
                    len(parent_gromet_fn.pof),
                    parent_cast_node,
                )
        elif isinstance(node.right, AnnCastLiteralValue):
            # Assignment for
            # LiteralValue (i.e. 3), tuples
            if is_tuple(node.right):
                # Case for when right hand side is a tuple
                # For instances like
                # x = a,b,c,...
                # x,y,z,... = w,a,b,...
                ref = node.source_refs[0]
                metadata = self.create_source_code_reference(ref)

                # Make Expression GrometFN
                # new_gromet = GrometFN()
                # new_gromet.b = insert_gromet_object(
                #    new_gromet.b,
                #    GrometBoxFunction(function_type=FunctionType.EXPRESSION),
                # )

                # Visit each individual value in the tuple
                # and collect the resulting
                # pofs from each value
                tuple_indices = []
                for val in node.right.value:
                    if isinstance(val, AnnCastLiteralValue):
                        new_gromet = GrometFN()
                        new_gromet.b = insert_gromet_object(
                            new_gromet.b,
                            GrometBoxFunction(
                                function_type=FunctionType.EXPRESSION
                            ),
                        )

                        self.visit(val, new_gromet, parent_cast_node)

                        # Create the opo for the Gromet Expression holding the literal and then wire its opo to the literal's pof
                        new_gromet.opo = insert_gromet_object(
                            new_gromet.opo, GrometPort(box=len(new_gromet.b))
                        )

                        new_gromet.wfopo = insert_gromet_object(
                            new_gromet.wfopo,
                            GrometWire(
                                src=len(new_gromet.opo),
                                tgt=len(new_gromet.pof),
                            ),
                        )

                        # Append this Gromet Expression holding the value to the overall gromet FN collection
                        self.gromet_module.fn_array = insert_gromet_object(
                            self.gromet_module.fn_array,
                            new_gromet,
                        )
                        self.set_index()

                        # Make the 'call' box function that connects the expression to the parent and creates its output port
                        # print(node.source_refs)
                        parent_gromet_fn.bf = insert_gromet_object(
                            parent_gromet_fn.bf,
                            GrometBoxFunction(
                                function_type=FunctionType.EXPRESSION,
                                body=len(self.gromet_module.fn_array),
                                metadata=self.insert_metadata(metadata),
                            ),
                        )

                        # Make the output port for this value in the tuple
                        # If the left hand side is also a tuple this port will get named
                        # further down
                        parent_gromet_fn.pof = insert_gromet_object(
                            parent_gromet_fn.pof,
                            GrometPort(
                                name=None,
                                box=len(parent_gromet_fn.bf),
                            ),
                        )

                        var_pof = len(parent_gromet_fn.pof)

                    elif isinstance(val, AnnCastOperator):
                        new_gromet = GrometFN()
                        new_gromet.b = insert_gromet_object(
                            new_gromet.b,
                            GrometBoxFunction(
                                function_type=FunctionType.EXPRESSION
                            ),
                        )
                        self.visit(val, new_gromet, parent_cast_node)

                        self.gromet_module.fn_array = insert_gromet_object(
                            self.gromet_module.fn_array,
                            new_gromet,
                        )
                        self.set_index()

                        # Make the 'call' box function that connects the expression to the parent and creates its output port
                        # print(node.source_refs)
                        parent_gromet_fn.bf = insert_gromet_object(
                            parent_gromet_fn.bf,
                            GrometBoxFunction(
                                function_type=FunctionType.EXPRESSION,
                                body=len(self.gromet_module.fn_array),
                                metadata=self.insert_metadata(metadata),
                            ),
                        )

                        parent_gromet_fn.pof = insert_gromet_object(
                            parent_gromet_fn.pof,
                            GrometPort(
                                name=None,
                                box=len(parent_gromet_fn.bf),
                            ),
                        )

                        var_pof = len(parent_gromet_fn.pof)
                    elif isinstance(val, AnnCastName):
                        var_pof = self.retrieve_var_port(val.name)
                    else:
                        var_pof = -1
                        # print(type(val))

                    tuple_indices.append(var_pof)

                # Determine if the left hand side is
                # - A tuple of variables
                #   - In this case we can directly wire each value from the
                #     right hand side to values on the left hand side
                # - One variable
                #   - We need to add a pack primitive if that's the case
                # NOTE: This is subject to change
                if is_tuple(node.left):
                    # tuple_indices stores 1-index pofs, so we have to offset by one
                    # to index with them
                    for i, val in enumerate(node.left.value, 0):
                        parent_gromet_fn.pof[
                            tuple_indices[i] - 1
                        ].name = get_left_side_name(node.left.value[i])

                        self.add_var_to_env(
                            get_left_side_name(node.left.value[i]),
                            node.left.value[i],
                            parent_gromet_fn.pof[tuple_indices[i] - 1],
                            tuple_indices[i],
                            parent_cast_node,
                        )
                elif isinstance(node.left, AnnCastVar):
                    self.create_pack(
                        node.left,
                        tuple_indices,
                        parent_gromet_fn,
                        parent_cast_node,
                    )
            else:
                if node.source_refs == None:
                    ref = []
                    metadata = None
                else:
                    ref = node.source_refs[0]
                    metadata = self.create_source_code_reference(ref)

                # Make Expression GrometFN
                new_gromet = GrometFN()
                new_gromet.b = insert_gromet_object(
                    new_gromet.b,
                    GrometBoxFunction(function_type=FunctionType.EXPRESSION),
                )

                # Visit the literal value, which makes a bf for a literal and puts a pof to it
                self.visit(node.right, new_gromet, node)

                # Create the opo for the Gromet Expression holding the literal and then wire its opo to the literal's pof
                new_gromet.opo = insert_gromet_object(
                    new_gromet.opo, GrometPort(box=len(new_gromet.b))
                )
                new_gromet.wfopo = insert_gromet_object(
                    new_gromet.wfopo,
                    GrometWire(
                        src=len(new_gromet.opo), tgt=len(new_gromet.pof)
                    ),
                )

                # Append this Gromet Expression holding the literal to the overall gromet FN collection
                self.gromet_module.fn_array = insert_gromet_object(
                    self.gromet_module.fn_array,
                    new_gromet,
                )
                self.set_index()

                # Make the 'call' box function that connects the expression to the parent and creates its output port
                # print(node.source_refs)
                parent_gromet_fn.bf = insert_gromet_object(
                    parent_gromet_fn.bf,
                    GrometBoxFunction(
                        function_type=FunctionType.EXPRESSION,
                        body=len(self.gromet_module.fn_array),
                        metadata=self.insert_metadata(metadata),
                    ),
                )
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=get_left_side_name(node.left),
                        box=len(parent_gromet_fn.bf),
                    ),
                )

                # TODO: expand on this later with loops
                if isinstance(parent_cast_node, AnnCastModelIf):
                    parent_gromet_fn.opi = insert_gromet_object(
                        parent_gromet_fn.opi,
                        GrometPort(box=len(parent_gromet_fn.b)),
                    )
                    parent_gromet_fn.opo = insert_gromet_object(
                        parent_gromet_fn.opo,
                        GrometPort(box=len(parent_gromet_fn.b)),
                    )
                    parent_gromet_fn.wfopo = insert_gromet_object(
                        parent_gromet_fn.wfopo,
                        GrometWire(
                            src=len(parent_gromet_fn.opo),
                            tgt=len(parent_gromet_fn.pof),
                        ),
                    )

                # Store the new variable we created into the variable environment
                self.add_var_to_env(
                    get_left_side_name(node.left),
                    node.left,
                    parent_gromet_fn.pof[-1],
                    len(parent_gromet_fn.pof),
                    parent_cast_node,
                )

        else:
            # General Case
            # Assignment for
            #   - Expression consisting of binary ops (x + y + ...), etc
            #   - Other cases we haven't thought about
            ref = node.source_refs[0]
            metadata = self.create_source_code_reference(ref)

            # Create an expression FN
            new_gromet = GrometFN()
            new_gromet.b = insert_gromet_object(
                new_gromet.b,
                GrometBoxFunction(function_type=FunctionType.EXPRESSION),
            )

            self.visit(node.right, new_gromet, node)
            # At this point we identified the variable being assigned (i.e. for exp0.py: x)
            # we need to do some bookkeeping to associate the source CAST/GrFN variable with
            # the output port of the GroMEt expression call
            # NOTE: This may need to change from just indexing to something more
            new_gromet.opo = insert_gromet_object(
                new_gromet.opo, GrometPort(box=len(new_gromet.b))
            )

            # GroMEt wiring creation
            # The creation of the wire between the output port (pof) of the top-level node
            # of the tree rooted in node.right needs to be wired to the output port out (OPO)
            # of the GExpression of this AnnCastAssignment
            if (
                new_gromet.opo == None and new_gromet.pof == None
            ):  # TODO: double check this guard to see if it's necessary
                # print(node.source_refs[0])
                new_gromet.wfopo = insert_gromet_object(
                    new_gromet.wfopo, GrometWire(src=-1, tgt=-1)
                )
            elif new_gromet.pof == None:
                new_gromet.wfopo = insert_gromet_object(
                    new_gromet.wfopo,
                    GrometWire(src=len(new_gromet.opo), tgt=-1),
                )
            elif new_gromet.opo == None:
                # print(node.source_refs[0])
                new_gromet.wfopo = insert_gromet_object(
                    new_gromet.wfopo,
                    GrometWire(src=-1, tgt=len(new_gromet.pof)),
                )
            else:
                new_gromet.wfopo = insert_gromet_object(
                    new_gromet.wfopo,
                    GrometWire(
                        src=len(new_gromet.opo), tgt=len(new_gromet.pof)
                    ),
                )
            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array, new_gromet
            )
            self.set_index()

            # An assignment in a conditional or loop's body doesn't add bf, pif, or pof to the parent gromet FN
            # So we check if this assignment is not in either of those and add accordingly
            # NOTE: The above is no longer true because now Ifs/Loops create an additional 'Function' GroMEt FN for
            #       their respective parts, so we do need to add this Expression GroMEt FN to the parent bf
            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf,
                GrometBoxFunction(
                    function_type=FunctionType.EXPRESSION,
                    body=len(self.gromet_module.fn_array),
                    metadata=self.insert_metadata(metadata),
                ),
            )

            # There's no guarantee that our expression GroMEt used any inputs
            # Therefore we check if we have any inputs before checking them
            # For each opi the Expression GroMEt may have, we add a corresponding pif
            # to it, and then we see if we need to wire the pif to anything
            if new_gromet.opi != None:
                for opi in new_gromet.opi:
                    parent_gromet_fn.pif = insert_gromet_object(
                        parent_gromet_fn.pif,
                        GrometPort(box=len(parent_gromet_fn.bf)),
                    )
                    self.wire_from_var_env(opi.name, parent_gromet_fn)

                    # This is kind of a hack, so the opis are labeled by the GroMEt expression creation, but then we have to unlabel them
                    opi.name = None

            # Put the final pof in the GroMEt expression call, and add its respective variable to the variable environment
            if isinstance(node.left, AnnCastAttribute):
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(
                        name=node.left.attr.name, box=len(parent_gromet_fn.bf)
                    ),
                )
            # elif isinstance(node.left, AnnCastTuple):  # TODO: double check that this addition is correct
            elif is_tuple(node.left):
                for i, elem in enumerate(node.left.value, 1):
                    if (
                        parent_gromet_fn.pof != None
                    ):  # TODO: come back and fix this guard later
                        pof_idx = len(parent_gromet_fn.pof) - 1
                    else:
                        pof_idx = -1
                    if (
                        parent_gromet_fn.pof != None
                    ):  # TODO: come back and fix this guard later
                        self.add_var_to_env(
                            elem.val.name,
                            elem,
                            parent_gromet_fn.pof[pof_idx],
                            pof_idx,
                            parent_cast_node,
                        )
                        parent_gromet_fn.pof[pof_idx].name = elem.val.name
            else:
                name = ""
                if isinstance(node.left, AnnCastCall):
                    name = node.left.func.name
                else:
                    name = node.left.val.name
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(name=name, box=len(parent_gromet_fn.bf)),
                )

            # TODO: expand on this later
            if isinstance(parent_cast_node, AnnCastModelIf):
                parent_gromet_fn.opi = insert_gromet_object(
                    parent_gromet_fn.opi,
                    GrometPort(box=len(parent_gromet_fn.b)),
                )
                parent_gromet_fn.opo = insert_gromet_object(
                    parent_gromet_fn.opo,
                    GrometPort(box=len(parent_gromet_fn.b)),
                )
                parent_gromet_fn.wfopo = insert_gromet_object(
                    parent_gromet_fn.wfopo,
                    GrometWire(
                        src=len(parent_gromet_fn.opo),
                        tgt=len(parent_gromet_fn.pof),
                    ),
                )

            if isinstance(node.left, AnnCastAttribute):
                self.add_var_to_env(
                    node.left.attr.name,
                    node.left,
                    parent_gromet_fn.pof[-1],
                    len(parent_gromet_fn.pof),
                    parent_cast_node,
                )
            # elif isinstance(node.left, AnnCastTuple):  # TODO: double check that this addition is correct
            elif is_tuple(node.left):
                for i, elem in enumerate(node.left.value, 1):
                    if (
                        parent_gromet_fn.pof != None
                    ):  # TODO: come back and fix this guard later
                        pof_idx = len(parent_gromet_fn.pof) - i
                        # pof_idx = len(parent_gromet_fn.pof) - 1
                    else:
                        pof_idx = -1
                    if (
                        parent_gromet_fn.pof != None
                    ):  # TODO: come back and fix this guard later
                        self.add_var_to_env(
                            elem.val.name,
                            elem,
                            parent_gromet_fn.pof[pof_idx],
                            pof_idx,
                            parent_cast_node,
                        )
                        parent_gromet_fn.pof[pof_idx].name = elem.val.name
            else:
                name = ""
                if isinstance(node.left, AnnCastCall):
                    name = node.left.func.name
                else:
                    name = node.left.val.name
                self.add_var_to_env(
                    name,
                    node.left,
                    parent_gromet_fn.pof[-1],
                    len(parent_gromet_fn.pof),
                    parent_cast_node,
                )

        # One way or another we have a hold of the GEXpression object here.
        # Whatever's returned by the RHS of the assignment,
        # i.e. LiteralValue or primitive operator or function call.
        # Now we can look at its output port(s)

    @_visit.register
    def visit_attribute(
        self, node: AnnCastAttribute, parent_gromet_fn, parent_cast_node
    ):
        # Use self.import_collection to look up the attribute name
        # to see if it exists in there.
        # If the attribute exists, then we can create an import reference
        # node.value: left-side (i.e. module name or a class variable)
        # node.attr: right-side (i.e. name of a function or an attribute of a class)
        ref = node.source_refs[0]
        if isinstance(node.value, AnnCastName):
            name = node.value.name
            if name in self.import_collection:
                func_info = self.determine_func_type(node)
                parent_gromet_fn.bf = insert_gromet_object(
                    parent_gromet_fn.bf,
                    GrometBoxFunction(
                        name=f"{name}.{node.attr.name}",
                        function_type=func_info[0],
                        import_type=func_info[1],
                        import_version=func_info[2],
                        import_source=func_info[3],
                        source_language=func_info[4],
                        source_language_version=func_info[5],
                        body=None,
                        metadata=self.insert_metadata(
                            self.create_source_code_reference(ref)
                        ),
                    ),
                )
            elif isinstance(node.attr, AnnCastName):
                if node.value.name == "self":
                    # Compose the case of "self.x" where x is an attribute
                    # Create string literal for "get" second argument
                    parent_gromet_fn.bf = insert_gromet_object(
                        parent_gromet_fn.bf,
                        GrometBoxFunction(
                            function_type=FunctionType.LITERAL,
                            value=GLiteralValue("string", node.attr.name),
                        ),
                    )
                    parent_gromet_fn.pof = insert_gromet_object(
                        parent_gromet_fn.pof,
                        GrometPort(box=len(parent_gromet_fn.bf)),
                    )

                    # Create "get" function and first argument, then wire it to 'self' argument
                    get_bf = GrometBoxFunction(
                        name="get", function_type=FunctionType.ABSTRACT
                    )
                    parent_gromet_fn.bf = insert_gromet_object(
                        parent_gromet_fn.bf, get_bf
                    )
                    parent_gromet_fn.pif = insert_gromet_object(
                        parent_gromet_fn.pif,
                        GrometPort(box=len(parent_gromet_fn.bf)),
                    )
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(src=len(parent_gromet_fn.pif), tgt=1),
                    )  # self is opi 1 everytime

                    # Create "get" second argument and wire it to the string literal from earlier
                    parent_gromet_fn.pif = insert_gromet_object(
                        parent_gromet_fn.pif,
                        GrometPort(box=len(parent_gromet_fn.bf)),
                    )
                    parent_gromet_fn.wff = insert_gromet_object(
                        parent_gromet_fn.wff,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=len(parent_gromet_fn.pof),
                        ),
                    )

                    # Create "get" pof
                    parent_gromet_fn.pof = insert_gromet_object(
                        parent_gromet_fn.pof,
                        GrometPort(box=len(parent_gromet_fn.bf)),
                    )
                elif isinstance(
                    parent_cast_node, AnnCastCall
                ):  # Case where a class is calling a method (i.e. mc is a class, and we do mc.get_c())
                    func_name = node.attr.name

                    if node.value.name in self.initialized_records:
                        obj_name = self.initialized_records[node.value.name]
                        if (
                            func_name in self.record[obj_name].keys()
                        ):  # TODO: remove this guard later
                            idx = self.record[obj_name][func_name]
                            parent_gromet_fn.bf = insert_gromet_object(
                                parent_gromet_fn.bf,
                                GrometBoxFunction(
                                    name=func_name,
                                    function_type=FunctionType.FUNCTION,
                                    body=idx,
                                ),
                            )
                            # parent_gromet_fn.bf = insert_gromet_object(parent_gromet_fn.bf, GrometBoxFunction(name=f"{obj_name}:{func_name}", function_type=FunctionType.FUNCTION, contents=idx, metadata=self.insert_metadata(metadata)))

                            parent_gromet_fn.pif = insert_gromet_object(
                                parent_gromet_fn.pif,
                                GrometPort(
                                    name=node.value.name,
                                    box=len(parent_gromet_fn.bf),
                                ),
                            )
                            parent_gromet_fn.pof = insert_gromet_object(
                                parent_gromet_fn.pof,
                                GrometPort(box=len(parent_gromet_fn.bf)),
                            )
                    else:  # Attribute of a class that we don't have access to
                        # NOTE: This will probably have to change later
                        func_info = self.determine_func_type(node)

                        parent_gromet_fn.bf = insert_gromet_object(
                            parent_gromet_fn.bf,
                            GrometBoxFunction(
                                name=f"{node.value.name}.{func_name}",
                                function_type=func_info[0],
                                import_type=func_info[1],
                                import_version=func_info[2],
                                import_source=func_info[3],
                                source_language=func_info[4],
                                source_language_version=func_info[5],
                                body=None,
                            ),
                        )
                        parent_gromet_fn.pof = insert_gromet_object(
                            parent_gromet_fn.pof,
                            GrometPort(box=len(parent_gromet_fn.bf)),
                        )

        elif isinstance(node.value, AnnCastCall):
            # NOTE: M7 placeholder
            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf,
                GrometBoxFunction(
                    function_type=FunctionType.FUNCTION,
                    body=None,
                    metadata=self.insert_metadata(
                        self.create_source_code_reference(ref)
                    ),
                ),
            )
        elif isinstance(node.value, AnnCastOperator):
            # Added to support scenario 2 of Jul'23 hackathon
            # Create an expression FN
            new_gromet = GrometFN()
            new_gromet.b = insert_gromet_object(
                new_gromet.b,
                GrometBoxFunction(function_type=FunctionType.EXPRESSION),
            )

            self.visit(node.value, new_gromet, node)

            new_gromet.opo = insert_gromet_object(
                new_gromet.opo, GrometPort(box=len(new_gromet.b))
            )

            new_gromet.wfopo = insert_gromet_object(
                new_gromet.wfopo,
                GrometWire(src=len(new_gromet.opo), tgt=len(new_gromet.pof)),
            )

            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array, new_gromet
            )
            self.set_index()

            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf,
                GrometBoxFunction(
                    function_type=FunctionType.FUNCTION,
                    body=len(self.gromet_module.fn_array),
                    metadata=self.insert_metadata(
                        self.create_source_code_reference(ref)
                    ),
                ),
            )

            operator_idx = len(parent_gromet_fn.bf)
            # The operation makes some opis, we attempt to
            # match the number of opis with pifs in the parent FN
            # and also wire these ports appropriately
            if new_gromet.opi != None:
                for opi in new_gromet.opi:
                    parent_gromet_fn.pif = insert_gromet_object(
                        parent_gromet_fn.pif, GrometPort(box=operator_idx)
                    )

                    # Attempt to find where the port is in the parent FN and wire it
                    # NOTE: this will need to be updated with more handling, i.e. for loops cond etc
                    var_loc = self.retrieve_var_port(opi.name)
                    parent_gromet_fn.wff = insert_gromet_object(
                        parent_gromet_fn.wff,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=var_loc,
                        ),
                    )

            parent_gromet_fn.pof = insert_gromet_object(
                parent_gromet_fn.pof, GrometPort(box=operator_idx)
            )
            operator_pof_idx = len(parent_gromet_fn.pof)

            if isinstance(parent_cast_node, AnnCastCall):
                func_name = node.attr.name
                func_info = (
                    FunctionType.IMPORTED,
                    ImportType.NATIVE,
                    None,
                    None,
                    "Python",
                    PYTHON_VERSION,
                )

                parent_gromet_fn.bf = insert_gromet_object(
                    parent_gromet_fn.bf,
                    GrometBoxFunction(
                        name=f"{func_name}",
                        function_type=func_info[0],
                        import_type=func_info[1],
                        import_version=func_info[2],
                        import_source=func_info[3],
                        source_language=func_info[4],
                        source_language_version=func_info[5],
                        body=None,
                    ),
                )
                # Add the input for this function, and then wire it
                # NOTE: This needs more development to support multiple arguments
                parent_gromet_fn.pif = insert_gromet_object(
                    parent_gromet_fn.pif,
                    GrometPort(box=len(parent_gromet_fn.bf)),
                )

                parent_gromet_fn.wff = insert_gromet_object(
                    parent_gromet_fn.wff,
                    GrometWire(
                        src=len(parent_gromet_fn.pif),
                        tgt=operator_pof_idx,
                    ),
                )

                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof,
                    GrometPort(box=len(parent_gromet_fn.bf)),
                )

        else:
            pass
            # if node.value.name not in self.record.keys():
            #  pass
            # if func_name in self.record.keys():
            #   idx = self.record[func_name][f"new:{func_name}"]

            # parent_gromet_fn.bf = insert_gromet_object(parent_gromet_fn.bf, GrometBoxFunction(name=func_name, function_type=FunctionType.FUNCTION, contents=idx, metadata=self.insert_metadata(metadata)))
            # func_call_idx = len(parent_gromet_fn.bf)

    def handle_unary_op(
        self, node: AnnCastOperator, parent_gromet_fn, parent_cast_node
    ):
        """
        Handles an AnnCastOperator node that consists of one operand
        """
        metadata = self.create_source_code_reference(node.source_refs[0])
        opd_ret_val = self.visit(node.operands[0], parent_gromet_fn, node)

        opd_pof = -1
        if parent_gromet_fn.pof != None:
            opd_pof = len(parent_gromet_fn.pof)
        if isinstance(node.operands[0], AnnCastName):
            opd_pof = -1
        elif isinstance(node.operands[0], AnnCastCall):
            opd_pof = len(parent_gromet_fn.pof)
            for arg in node.operands[0].arguments:
                if hasattr(arg, "name"):
                    found_opi, opi_idx = find_existing_opi(
                        parent_gromet_fn, arg.name
                    )

                    if found_opi:
                        parent_gromet_fn.wfopi = insert_gromet_object(
                            parent_gromet_fn.wfopi,
                            GrometWire(
                                src=len(parent_gromet_fn.pif),
                                tgt=opi_idx,
                            ),
                        )
                    else:
                        parent_gromet_fn.opi = insert_gromet_object(
                            parent_gromet_fn.opi,
                            GrometPort(
                                name=arg.name, box=len(parent_gromet_fn.b)
                            ),
                        )
                        parent_gromet_fn.wfopi = insert_gromet_object(
                            parent_gromet_fn.wfopi,
                            GrometWire(
                                src=len(parent_gromet_fn.pif),
                                tgt=len(parent_gromet_fn.opi),
                            ),
                        )

        parent_gromet_fn.bf = insert_gromet_object(
            parent_gromet_fn.bf,
            GrometBoxFunction(
                name=node.op,
                function_type=FunctionType.LANGUAGE_PRIMITIVE,
                metadata=self.insert_metadata(metadata),
            ),
        )
        unop_idx = len(parent_gromet_fn.bf)

        parent_gromet_fn.pif = insert_gromet_object(
            parent_gromet_fn.pif, GrometPort(box=unop_idx)
        )

        if (
            isinstance(node.operands[0], (AnnCastName, AnnCastVar))
            and opd_pof == -1
        ):
            if isinstance(node.operands[0], AnnCastName):
                name = node.operands[0].name
            elif isinstance(node.operands[0], AnnCastVar):
                name = node.operands[0].val.name

            if parent_gromet_fn.b[0].function_type != FunctionType.FUNCTION:
                found_opi, opi_idx = find_existing_opi(parent_gromet_fn, name)

                if not found_opi:
                    parent_gromet_fn.opi = insert_gromet_object(
                        parent_gromet_fn.opi,
                        GrometPort(name=name, box=len(parent_gromet_fn.b)),
                    )
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=len(parent_gromet_fn.opi),
                        ),
                    )
                else:
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=opi_idx,
                        ),
                    )
            else:
                # If we are in a function def then we retrieve where the variable is
                # Whether it's in the local or the args environment

                self.wire_from_var_env(name, parent_gromet_fn)
        else:
            parent_gromet_fn.wff = insert_gromet_object(
                parent_gromet_fn.wff,
                GrometWire(src=len(parent_gromet_fn.pif), tgt=opd_pof),
            )

        parent_gromet_fn.pof = insert_gromet_object(
            parent_gromet_fn.pof, GrometPort(box=unop_idx)
        )

    def handle_binary_op(
        self, node: AnnCastOperator, parent_gromet_fn, parent_cast_node
    ):
        # visit LHS first, storing the return value and used if necessary
        # cases where it's used
        # - Function call: function call returns its index which can be used for pof generation
        opd_one_ret_val = self.visit(node.operands[0], parent_gromet_fn, node)

        # Collect where the location of the left pof is
        # If the left node is an AnnCastName then it
        # automatically doesn't have a pof
        # (This creates an opi later)
        opd_one_pof = -1
        if parent_gromet_fn.pof != None:
            opd_one_pof = len(parent_gromet_fn.pof)
        if isinstance(node.operands[0], AnnCastName):
            opd_one_pof = -1
        elif isinstance(node.operands[0], AnnCastCall):
            opd_one_pof = len(parent_gromet_fn.pof)

        # visit RHS second, storing the return value and used if necessary
        # cases where it's used
        # - Function call: function call returns its index which can be used for pof generation
        opd_two_ret_val = self.visit(node.operands[1], parent_gromet_fn, node)

        # Collect where the location of the right pof is
        # If the right node is an AnnCastName then it
        # automatically doesn't have a pof
        # (This create an opi later)
        opd_two_pof = -1
        if parent_gromet_fn.pof != None:
            opd_two_pof = len(parent_gromet_fn.pof)
        if isinstance(node.operands[1], AnnCastName):
            opd_two_pof = -1
        elif isinstance(node.operands[1], AnnCastCall):
            opd_two_pof = len(parent_gromet_fn.pof)

        ref = node.source_refs[0]
        metadata = self.create_source_code_reference(ref)

        # NOTE/TODO Maintain a table of primitive operators that when queried give you back
        # their signatures that can be used for generating
        # A global mapping is maintained but it isnt being used for their signatures yet
        parent_gromet_fn.bf = insert_gromet_object(
            parent_gromet_fn.bf,
            GrometBoxFunction(
                name=node.op,
                function_type=FunctionType.LANGUAGE_PRIMITIVE,
                metadata=self.insert_metadata(metadata),
            ),
        )

        # After we visit the left and right they (in all scenarios but one) append a POF
        # The one case where it doesnt happen is when the left or right are variables in the expression
        # In this case then they need an opi and the appropriate wiring for it
        parent_gromet_fn.pif = insert_gromet_object(
            parent_gromet_fn.pif, GrometPort(box=len(parent_gromet_fn.bf))
        )
        if (
            isinstance(node.operands[0], (AnnCastName, AnnCastVar))
        ) and opd_one_pof == -1:
            if isinstance(node.operands[0], AnnCastName):
                name = node.operands[0].name
            elif isinstance(node.operands[0], AnnCastVar):
                name = node.operands[0].val.name

            if parent_gromet_fn.b[0].function_type != FunctionType.FUNCTION:
                # This check is used for when the binary operation is part of a Function and not an Expression
                # the FunctionDef handles creating opis, so we create any here as necessary
                found_opi, opi_idx = find_existing_opi(parent_gromet_fn, name)

                name_comp = comp_name_nodes(node.operands[0], node.operands[1])
                if not name_comp and not found_opi:
                    parent_gromet_fn.opi = insert_gromet_object(
                        parent_gromet_fn.opi,
                        GrometPort(name=name, box=len(parent_gromet_fn.b)),
                    )
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=len(parent_gromet_fn.opi),
                        ),
                    )
                elif name_comp and not found_opi:
                    # NOTE: Added for M7, handling operations like x * x
                    parent_gromet_fn.opi = insert_gromet_object(
                        parent_gromet_fn.opi,
                        GrometPort(name=name, box=len(parent_gromet_fn.b)),
                    )
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=len(parent_gromet_fn.opi),
                        ),
                    )
                else:
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=opi_idx if found_opi else -1,
                        ),
                    )
            else:
                # If we are in a function def then we retrieve where the variable is
                # Whether it's in the local or the args environment
                self.wire_from_var_env(name, parent_gromet_fn)
        else:
            # In this case, the left node gave us a pof, so we can wire it to the pif here
            # if left_pof == -1:
            parent_gromet_fn.wff = insert_gromet_object(
                parent_gromet_fn.wff,
                GrometWire(src=len(parent_gromet_fn.pif), tgt=opd_one_pof),
            )

        # Repeat the above but for the right node this time
        # NOTE: In the case that the left and the right node both refer to the same function argument we only
        # want one opi created and so we dont create one here
        parent_gromet_fn.pif = insert_gromet_object(
            parent_gromet_fn.pif,
            GrometPort(box=len(parent_gromet_fn.bf)),
        )
        if isinstance(node.operands[1], AnnCastName) and opd_two_pof == -1:
            # This check is used for when the binary operation is part of a Function and not an Expression
            # the FunctionDef handles creating opis, so we create any here as necessary
            if parent_gromet_fn.b[0].function_type != FunctionType.FUNCTION:
                found_opi, opi_idx = find_existing_opi(
                    parent_gromet_fn, node.operands[1].name
                )

                name_comp = comp_name_nodes(node.operands[0], node.operands[1])
                if not name_comp and not found_opi:
                    parent_gromet_fn.opi = insert_gromet_object(
                        parent_gromet_fn.opi,
                        GrometPort(
                            name=node.operands[1].name,
                            box=len(parent_gromet_fn.b),
                        ),
                    )
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=len(parent_gromet_fn.opi),
                        ),
                    )
                elif (
                    name_comp and not found_opi
                ):  # NOTE: Added for M7, handling operations like x * x
                    parent_gromet_fn.opi = insert_gromet_object(
                        parent_gromet_fn.opi,
                        GrometPort(name=name, box=len(parent_gromet_fn.b)),
                    )
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=len(parent_gromet_fn.opi),
                        ),
                    )
                else:
                    parent_gromet_fn.wfopi = insert_gromet_object(
                        parent_gromet_fn.wfopi,
                        GrometWire(
                            src=len(parent_gromet_fn.pif),
                            tgt=opi_idx if found_opi else -1,
                        ),
                    )
            else:
                # If we are in a function def then we retrieve where the variable is
                # Whether it's in the local or the args environment
                self.wire_from_var_env(node.operands[1].name, parent_gromet_fn)
        else:
            # In this case, the right node gave us a pof, so we can wire it to the pif here
            parent_gromet_fn.wff = insert_gromet_object(
                parent_gromet_fn.wff,
                GrometWire(src=len(parent_gromet_fn.pif), tgt=opd_two_pof),
            )

        # Add the pof that serves as the output of this operation
        parent_gromet_fn.pof = insert_gromet_object(
            parent_gromet_fn.pof, GrometPort(box=len(parent_gromet_fn.bf))
        )

    @_visit.register
    def visit_operator(
        self, node: AnnCastOperator, parent_gromet_fn, parent_cast_node
    ):
        # What constitutes the two pieces of a BinaryOp?
        # Each piece can either be
        # - A literal value (i.e. 2)
        # - A function call that returns a value (i.e. foo())
        # - A BinaryOp itself
        # - A variable reference (i.e. x), this is the only one that doesnt plug a pof
        #   - This generally causes us to create an opi and a wfopi to connect this to a pif
        # - Other
        #   - A list access (i.e. x[2]) translates to a function call (_list_set), same for other sequential types

        if len(node.operands) == 1:
            self.handle_unary_op(node, parent_gromet_fn, parent_cast_node)
        elif len(node.operands) == 2:
            self.handle_binary_op(node, parent_gromet_fn, parent_cast_node)

    def wire_binary_op_args(self, node, parent_gromet_fn):
        if isinstance(node, AnnCastName):
            parent_gromet_fn.pif = insert_gromet_object(
                parent_gromet_fn.pif, GrometPort(box=len(parent_gromet_fn.bf))
            )
            var_environment = self.symtab_variables()
            if node.name in var_environment["local"]:
                local_env = var_environment["local"]
                entry = local_env[node.name]
                if isinstance(entry[0], AnnCastLoop):
                    parent_gromet_fn.wlf = insert_gromet_object(
                        parent_gromet_fn.wlf,
                        GrometWire(
                            src=len(parent_gromet_fn.pif), tgt=entry[2]
                        ),
                    )
                else:
                    parent_gromet_fn.wff = insert_gromet_object(
                        parent_gromet_fn.wff,
                        GrometWire(
                            src=len(parent_gromet_fn.pif), tgt=entry[2]
                        ),
                    )
            elif node.name in var_environment["args"]:
                args_env = var_environment["args"]
                entry = args_env[node.name]
                parent_gromet_fn.wfopi = insert_gromet_object(
                    parent_gromet_fn.wfopi,
                    GrometWire(src=len(parent_gromet_fn.pif), tgt=entry[2]),
                )
            return
        if isinstance(node, AnnCastOperator):
            self.wire_binary_op_args(node.operands[0], parent_gromet_fn)
            if len(node.operands) > 1:
                self.wire_binary_op_args(node.operands[1], parent_gromet_fn)
            return

    def func_in_module(self, func_name):
        """See if func_name is actually a function from
        an imported module
        A tuple of (Boolean, String) where the boolean value tells us
        if we found it or not and the string denotes the module if we did find it

        """
        for mname in self.import_collection.keys():
            curr_module = self.import_collection[mname]
            if curr_module[2] and find_func_in_module(
                mname, func_name
            ):  # If curr module is of form 'from mname import *'
                return (True, mname)
            if (
                func_name in curr_module[1]
            ):  # If the function has been imported individually and is in the symbols list
                return (
                    True,
                    mname,
                )  # With the form 'from mname import func_name'

        return (False, "")

    @_visit.register
    def visit_call(
        self, node: AnnCastCall, parent_gromet_fn, parent_cast_node
    ):
        ref = node.source_refs[0]
        metadata = self.create_source_code_reference(ref)

        # Used in special scenarios, when we might need
        # to do something slightly different
        from_assignment = False
        from_call = False
        from_operator = False
        func_name, qual_func_name = get_func_name(node)

        if isinstance(parent_cast_node, AnnCastAssignment):
            from_assignment = True
        elif isinstance(parent_cast_node, AnnCastCall):
            from_call = True
        elif isinstance(parent_cast_node, AnnCastOperator):
            from_operator = True

        in_module = self.func_in_module(func_name)
        func_info = self.determine_func_type(node)

        if isinstance(node.func, AnnCastAttribute):
            self.visit(node.func, parent_gromet_fn, parent_cast_node)

        # Have to find the index of the function we're trying to call
        # What if it's a primitive?
        # What if it doesn't exist for some reason?
        # What if it's from a module?
        if is_primitive(func_name, "CAST") and not in_module[0]:
            call_bf_idx = self.handle_primitive_function(
                node, parent_gromet_fn, parent_cast_node, from_assignment
            )

            # Argument handling for primitives is a little different here, because we only want to find the variables that we need, and not create
            # any additional FNs. The additional FNs are created in the primitive handler
            for arg in node.arguments:
                if isinstance(arg, AnnCastName):
                    parent_gromet_fn.pif = insert_gromet_object(
                        parent_gromet_fn.pif, GrometPort(box=call_bf_idx)
                    )
                    pif_idx = len(parent_gromet_fn.pif)
                    # Have to wire from either
                    # - a local variable
                    # - an argument/global variable introduced as argument
                    # NAME:
                    # if it's local, attempt to find it in the local args
                    # if it's argument, attempt to find its opi first
                    var_env = self.symtab_variables()

                    if arg.name in var_env["local"]:
                        self.wire_from_var_env(arg.name, parent_gromet_fn)
                    elif arg.name in var_env["args"]:
                        # The reason we have to explicitly check if the call argument is in the variable environment as opposed
                        # to just attempting to wire with 'wire_from_var_env' is that the expression can be its own FN without
                        # Attempt to find the opi if it already exists and wire to it
                        # otherwise add it
                        found_opi, opi_idx = find_existing_opi(
                            parent_gromet_fn, arg.name
                        )
                        if found_opi:
                            parent_gromet_fn.wfopi = insert_gromet_object(
                                parent_gromet_fn.wfopi,
                                GrometWire(
                                    src=len(parent_gromet_fn.pif), tgt=opi_idx
                                ),
                            )
                        else:
                            parent_gromet_fn.opi = insert_gromet_object(
                                parent_gromet_fn.opi,
                                GrometPort(name=arg.name, box=call_bf_idx),
                            )
                            opi_idx = len(parent_gromet_fn.opi)
                            parent_gromet_fn.wfopi = insert_gromet_object(
                                parent_gromet_fn.wfopi,
                                GrometWire(
                                    src=len(parent_gromet_fn.pif), tgt=opi_idx
                                ),
                            )
                    elif arg.name in var_env["global"]:
                        self.wire_from_var_env(arg.name, parent_gromet_fn)
        else:
            if in_module[0]:
                if isinstance(node.func, AnnCastAttribute):
                    name = qual_func_name
                else:
                    name = f"{in_module[1]}.{func_name}_id{node.func.id}"
            else:
                name = qual_func_name

            if check_builtin(func_name):
                body = None
            else:
                identified_func_name = qual_func_name
                idx, found = self.find_gromet(identified_func_name)
                if not found and func_name not in self.record.keys():
                    temp_gromet_fn = GrometFN()
                    temp_gromet_fn.b = insert_gromet_object(
                        temp_gromet_fn.b,
                        GrometBoxFunction(
                            name=func_name, function_type=FunctionType.FUNCTION
                        ),
                    )
                    self.gromet_module.fn_array = insert_gromet_object(
                        self.gromet_module.fn_array, temp_gromet_fn
                    )
                    self.set_index()

                if func_name in self.record.keys():
                    idx = self.record[func_name][f"new:{func_name}"]

                body = idx

            call_box_func = GrometBoxFunction(
                name=name,
                function_type=func_info[0] if func_info != None else None,
                import_type=func_info[1] if func_info != None else None,
                import_version=func_info[2] if func_info != None else None,
                import_source=func_info[3] if func_info != None else None,
                source_language=func_info[4] if func_info != None else None,
                source_language_version=func_info[5]
                if func_info != None
                else None,
                body=body,
                metadata=metadata,
            )
            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf, call_box_func
            )
            call_bf_idx = len(parent_gromet_fn.bf)

            # Iterate through all the arguments first
            # In the case that we are looking at a primitive that
            # is not inlined or part of an assignment we don't visit the
            # arguments as that's already been handled by the primitive handler
            # if not is_primitive(func_name, "CAST") or (from_assignment or is_inline(func_name)):
            for arg in node.arguments:
                self.visit(arg, parent_gromet_fn, node)

                parent_gromet_fn.pif = insert_gromet_object(
                    parent_gromet_fn.pif, GrometPort(box=call_bf_idx)
                )
                pif_idx = len(parent_gromet_fn.pif)
                if is_tuple(arg):
                    for v in arg.value:
                        if hasattr(v, "name"):
                            self.wire_from_var_env(v.name, parent_gromet_fn)
                elif isinstance(
                    arg, (AnnCastLiteralValue, AnnCastCall, AnnCastOperator)
                ):
                    # Can wff here due to all these ^^ giving us local pofs

                    pof_idx = len(parent_gromet_fn.pof)
                    parent_gromet_fn.wff = insert_gromet_object(
                        parent_gromet_fn.wff,
                        GrometWire(src=pif_idx, tgt=pof_idx),
                    )
                elif isinstance(arg, AnnCastName):
                    # Have to wire from either
                    # - a local variable
                    # - an argument/global variable introduced as argument
                    # NAME:
                    # if it's local, attempt to find it in the local args
                    # if it's argument, attempt to find its opi first
                    var_env = self.symtab_variables()

                    if arg.name in var_env["local"]:
                        self.wire_from_var_env(arg.name, parent_gromet_fn)
                    elif arg.name in var_env["args"]:
                        # The reason we have to explicitly check if the call argument is in the variable environment as opposed
                        # to just attempting to wire with 'wire_from_var_env' is that the expression can be its own FN without
                        # Attempt to find the opi if it already exists and wire to it
                        # otherwise add it
                        found_opi, opi_idx = find_existing_opi(
                            parent_gromet_fn, arg.name
                        )
                        if found_opi:
                            parent_gromet_fn.wfopi = insert_gromet_object(
                                parent_gromet_fn.wfopi,
                                GrometWire(
                                    src=len(parent_gromet_fn.pif), tgt=opi_idx
                                ),
                            )
                        else:
                            parent_gromet_fn.opi = insert_gromet_object(
                                parent_gromet_fn.opi,
                                GrometPort(name=arg.name, box=call_bf_idx),
                            )
                            opi_idx = len(parent_gromet_fn.opi)
                            parent_gromet_fn.wfopi = insert_gromet_object(
                                parent_gromet_fn.wfopi,
                                GrometWire(
                                    src=len(parent_gromet_fn.pif), tgt=opi_idx
                                ),
                            )

        if from_call or from_operator or from_assignment:
            # Operator and calls need a pof appended here because they dont
            # do it themselves
            # At some point we would like the call handler to always append a POF
            if from_assignment and is_tuple(parent_cast_node.left):
                # If an assignment is to a tuple, we create multiple pofs
                for _ in parent_cast_node.left.value:
                    parent_gromet_fn.pof = insert_gromet_object(
                        parent_gromet_fn.pof, GrometPort(box=call_bf_idx)
                    )
            else:
                parent_gromet_fn.pof = insert_gromet_object(
                    parent_gromet_fn.pof, GrometPort(box=call_bf_idx)
                )

        # If we're doing a call to a Record's "__init__" which is
        # determined by the function name matching the
        # record name, then we need to add one additional argument
        # to represent the parent class that this current record 'might'
        # inherit. Currently we support either no parent class or one parent class
        if func_name in self.record.keys():
            # Generate a "None" for no parent class
            val = GLiteralValue("None", "None")

            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf,
                GrometBoxFunction(
                    function_type=FunctionType.LITERAL,
                    value=val,
                    metadata=None,  # TODO: Insert metadata for generated none value
                ),
            )

            # The None LiteralValue needs a pof to wire to
            parent_gromet_fn.pof = insert_gromet_object(
                parent_gromet_fn.pof, GrometPort(box=len(parent_gromet_fn.bf))
            )
            none_pof = len(parent_gromet_fn.pof)

            parent_gromet_fn.pif = insert_gromet_object(
                parent_gromet_fn.pif, GrometPort(box=call_bf_idx)
            )
            none_pif = len(parent_gromet_fn.pif)

            parent_gromet_fn.wff = insert_gromet_object(
                parent_gromet_fn.wff, GrometWire(src=none_pif, tgt=none_pof)
            )

        return call_bf_idx

    def wire_return_name(self, name, gromet_fn, index=1):
        var_environment = self.symtab_variables()
        if name in var_environment["local"]:
            # If it's in the local env, then
            # either it comes from a loop (wlopo), a conditional (wcopo), or just another
            # function (wfopo), then we check where it comes from and wire appropriately
            local_env = var_environment["local"]
            entry = local_env[name]
            if isinstance(entry[0], AnnCastLoop):
                gromet_fn.wlopo = insert_gromet_object(
                    gromet_fn.wlopo, GrometWire(src=index, tgt=entry[2])
                )
            elif isinstance(entry[0], AnnCastModelIf):
                gromet_fn.wcopo = insert_gromet_object(
                    gromet_fn.wcopo, GrometWire(src=index, tgt=entry[2])
                )
            else:
                gromet_fn.wfopo = insert_gromet_object(
                    gromet_fn.wfopo, GrometWire(src=index, tgt=entry[2])
                )
        elif name in var_environment["args"]:
            # If it comes from arguments, then that means the variable
            # Didn't get changed in the function at all and thus it's just
            # A pass through (wopio)
            args_env = var_environment["args"]
            entry = args_env[name]
            gromet_fn.wopio = insert_gromet_object(
                gromet_fn.wopio, GrometWire(src=index, tgt=entry[2])
            )

    def pack_return_tuple(self, node, gromet_fn):
        """Given a tuple node in a return statement
        This function creates the appropriate packing
        construct to pack the values of the tuple into one value
        that gets returned
        """
        metadata = self.create_source_code_reference(node.source_refs[0])

        if isinstance(node, AnnCastLiteralValue):
            ret_vals = list(node.value)
        else:
            ret_vals = list(node.values)

        # Create the pack primitive
        gromet_fn.bf = insert_gromet_object(
            gromet_fn.bf,
            GrometBoxFunction(
                function_type=FunctionType.ABSTRACT,
                name="pack",
                metadata=self.insert_metadata(metadata),
            ),
        )
        pack_bf_idx = len(gromet_fn.bf)

        for i, val in enumerate(ret_vals, 1):
            if isinstance(val, AnnCastName):
                # Need: The port number where it is from, and whether it's a local/function param/global
                name = val.name
                var_environment = self.symtab_variables()
                if name in var_environment["local"]:
                    local_env = var_environment["local"]
                    entry = local_env[name]
                    gromet_fn.pif = insert_gromet_object(
                        gromet_fn.pif, GrometPort(box=pack_bf_idx)
                    )
                    gromet_fn.wff = insert_gromet_object(
                        gromet_fn.wff,
                        GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
                    )
                elif name in var_environment["args"]:
                    args_env = var_environment["args"]
                    entry = args_env[name]
                    gromet_fn.pif = insert_gromet_object(
                        gromet_fn.pif, GrometPort(box=pack_bf_idx)
                    )
                    gromet_fn.wfopi = insert_gromet_object(
                        gromet_fn.wfopi,
                        GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
                    )
                elif name in var_environment["global"]:
                    # TODO
                    global_env = var_environment["global"]
                    entry = global_env[name]
                    gromet_fn.wff = insert_gromet_object(
                        gromet_fn.wff,
                        GrometWire(src=len(gromet_fn.pif), tgt=entry[2]),
                    )

            # elif isinstance(val, AnnCastTuple): # or isinstance(val, AnnCastList):
            elif (
                isinstance(val, AnnCastLiteralValue)
                and val.value_type == StructureType.TUPLE
            ):
                # TODO: this wire an extra wfopo that we don't need, must fix
                self.pack_return_tuple(val, gromet_fn)
            # elif isinstance(val, AnnCastCall):
            #  pass
            else:  # isinstance(val, AnnCastBinaryOp) or isinstance(val, AnnCastCall):
                # A Binary Op will create an expression FN
                # Which leaves a pof
                self.visit(val, gromet_fn, node)
                last_pof = len(gromet_fn.pof)
                gromet_fn.pif = insert_gromet_object(
                    gromet_fn.pif, GrometPort(box=pack_bf_idx)
                )
                gromet_fn.wff = insert_gromet_object(
                    gromet_fn.wff,
                    GrometWire(src=len(gromet_fn.pif), tgt=last_pof),
                )

        gromet_fn.pof = insert_gromet_object(
            gromet_fn.pof, GrometPort(box=pack_bf_idx)
        )

        # Add the opo for this gromet FN for the one return value that we're returning with the
        # pack
        gromet_fn.wfopo = insert_gromet_object(
            gromet_fn.wfopo,
            GrometWire(src=len(gromet_fn.opo), tgt=len(gromet_fn.pof)),
        )

    def wire_return_node(self, node, gromet_fn):
        """Return statements have many ways in which they can be wired, and thus
        we use this recursive function to handle all the possible cases
        """
        # NOTE: Thinking of adding an index parameter that is set to 1 when originally called, and then
        # if we have a tuple of returns then we can change the index then
        if isinstance(node, AnnCastLiteralValue):
            if is_tuple(node):
                self.pack_return_tuple(node, gromet_fn)
            return
        elif isinstance(node, AnnCastVar):
            var_name = node.val.name
            self.wire_return_name(var_name, gromet_fn)
        elif isinstance(node, AnnCastName):
            name = node.name
            self.wire_return_name(name, gromet_fn)
        elif (
            isinstance(node, AnnCastLiteralValue)
            and node.val.value_type == StructureType.LIST
        ):
            ret_vals = list(node.value)
            for i, val in enumerate(ret_vals, 1):
                if isinstance(val, AnnCastOperator):
                    self.wire_return_node(val.operands[0], gromet_fn)
                    if len(val.operands) > 1:
                        self.wire_return_node(val.operands[1], gromet_fn)
                # elif isinstance(val, AnnCastTuple) or (
                elif is_tuple(val) or (
                    isinstance(val, AnnCastLiteralValue)
                    and val.value_type == StructureType.LIST
                ):
                    self.wire_return_node(val, gromet_fn)
                else:
                    self.wire_return_name(val.name, gromet_fn, i)
        elif isinstance(node, AnnCastOperator):
            # A BinaryOp currently implies that we have one single OPO to put return values into
            gromet_fn.wfopo = insert_gromet_object(
                gromet_fn.wfopo, GrometWire(src=1, tgt=len(gromet_fn.pof))
            )
            # self.wire_return_node(node.left, gromet_fn)
            # self.wire_return_node(node.right, gromet_fn)
        return

    def handle_function_def(
        self,
        node: AnnCastFunctionDef,
        new_gromet_fn,
        func_body,
        parent_cast_node=None,
    ):
        """Handles the logic of making a function, whether the function itself is a real
        function definition (that is, it comes from an AnnCastFunctionDef) or it's
        'artifically generated' (that is, a set of statements coming from a loop or an if statement)
        """

        # If this function definition is within another function definition
        # Then we need to do some merging of function argument environments
        # so that this inner function definition can see and use the arguments from the outer
        # function definition
        var_environment = self.symtab_variables()

        prev_local_env = {}
        if isinstance(parent_cast_node, AnnCastFunctionDef):
            prev_local_env = deepcopy(var_environment["local"])

        var_environment["local"] = {}

        for n in func_body:
            self.visit(n, new_gromet_fn, node)

        # Create wfopo/wlopo/wcopo to wire the final computations to the output port
        # TODO: What about the case where there's multiple return values
        # also TODO: We need some kind of logic check to determine when we make a wopio for the case that an argument just passes through without
        # being used

        # If the last node in  the FunctionDef is a return node we must do some final wiring
        if isinstance(n, AnnCastModelReturn):
            self.wire_return_node(n.value, new_gromet_fn)

        elif (
            new_gromet_fn.opo != None
        ):  # This is in the case of a loop or conditional adding opos
            for i, opo in enumerate(new_gromet_fn.opo, 1):
                if opo.name in var_environment["local"]:
                    local_env = var_environment["local"]
                    entry = local_env[opo.name]
                    if isinstance(entry[0], AnnCastLoop):
                        new_gromet_fn.wlopo = insert_gromet_object(
                            new_gromet_fn.wlopo,
                            GrometWire(src=i, tgt=entry[2]),
                        )
                    # elif isinstance(entry[0], AnnCastModelIf):
                    #    new_gromet_fn.wcopo = insert_gromet_object(new_gromet_fn.wcopo, GrometWire(src=i,tgt=entry[2]+1))
                    else:
                        new_gromet_fn.wfopo = insert_gromet_object(
                            new_gromet_fn.wfopo,
                            GrometWire(src=i, tgt=entry[2]),
                        )
                elif opo.name in var_environment["args"]:
                    args_env = var_environment["args"]
                    entry = args_env[opo.name]
                    new_gromet_fn.wopio = insert_gromet_object(
                        new_gromet_fn.wopio,
                        GrometWire(src=i, tgt=entry[2]),
                    )

        # We're out of the function definition here, so we
        # can clear the local  variable environment
        var_environment["local"] = deepcopy(prev_local_env)

    @_visit.register
    def visit_function_def(
        self, node: AnnCastFunctionDef, parent_gromet_fn, parent_cast_node
    ):
        func_name = f"{node.name.name}_id{node.name.id}"
        identified_func_name = ".".join(node.con_scope)
        idx, found = self.find_gromet(func_name)

        ref = node.source_refs[0]

        if not found:
            new_gromet = GrometFN()
            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array, new_gromet
            )
            self.set_index()
            new_gromet.b = insert_gromet_object(
                new_gromet.b,
                GrometBoxFunction(
                    name=func_name,
                    function_type=FunctionType.FUNCTION
                    # name=func_name, function_type=FunctionType.FUNCTION
                ),
            )
        else:
            new_gromet = self.gromet_module.fn_array[idx - 1]

        metadata = self.create_source_code_reference(ref)

        new_gromet.b[0].metadata = self.insert_metadata(metadata)
        var_environment = self.symtab_variables()

        # metadata type for capturing the original identifier name (i.e. just foo) as it appeared in the code
        # as opposed to the PA derived name (i.e. module.foo_id0, etc..)
        # source_code_identifier_name

        # If this function definition is within another function definition
        # Then we need to do some merging of function argument environments
        # so that this inner function definition can see and use the arguments from the outer
        # function definition
        if isinstance(parent_cast_node, AnnCastFunctionDef):
            prev_arg_env = deepcopy(var_environment["args"])
        else:
            # Initialize the function argument variable environment and populate it as we
            # visit the function arguments
            prev_arg_env = {}
            var_environment["args"] = {}
        # arg_env = var_environment["args"]

        # Copy the previous local and argument environments
        # If we're a function within a function this effectively lets us
        # see all the local and arguments from the outer scope and use them
        # within here
        # If we have an argument or a local variable that share a name
        # With a variable or argument in the outer scope, then they get
        # overwritten (to simulate scope shadowing)
        # The use of {**var_env_args, **var_env_local} here creates new dictionaries,
        # so the old environments are left unchanged
        arg_env = {**var_environment["args"], **var_environment["local"]}
        var_environment["args"] = arg_env

        for arg in node.func_args:
            # Visit the arguments
            self.visit(arg, new_gromet, node)

            # for each argument we want to have a corresponding port (OPI) here
            arg_ref = arg.source_refs[0]
            arg_name = arg.val.name

            if arg.default_value != None:
                # if isinstance(arg.default_value, AnnCastTuple):
                if is_tuple(arg.default_value):
                    new_gromet.opi = insert_gromet_object(
                        new_gromet.opi,
                        GrometPort(
                            box=len(new_gromet.b),
                            name=arg_name,
                            default_value=arg.default_value.value,
                            metadata=self.insert_metadata(
                                self.create_source_code_reference(arg_ref)
                            ),
                        ),
                    )
                elif isinstance(arg.default_value, AnnCastCall):
                    new_gromet.opi = insert_gromet_object(
                        new_gromet.opi,
                        GrometPort(
                            box=len(new_gromet.b),
                            name=arg_name,
                            default_value=None,  # TODO: What's the actual default value?
                            metadata=self.insert_metadata(
                                self.create_source_code_reference(arg_ref)
                            ),
                        ),
                    )
                elif isinstance(arg.default_value, AnnCastOperator):
                    new_gromet.opi = insert_gromet_object(
                        new_gromet.opi,
                        GrometPort(
                            box=len(new_gromet.b),
                            name=arg_name,
                            default_value=None,  # TODO: M7 placeholder
                            metadata=self.insert_metadata(
                                self.create_source_code_reference(arg_ref)
                            ),
                        ),
                    )
                else:
                    new_gromet.opi = insert_gromet_object(
                        new_gromet.opi,
                        GrometPort(
                            box=len(new_gromet.b),
                            name=arg_name,
                            default_value=arg.default_value.value,
                            metadata=self.insert_metadata(
                                self.create_source_code_reference(arg_ref)
                            ),
                        ),
                    )
            else:
                new_gromet.opi = insert_gromet_object(
                    new_gromet.opi,
                    GrometPort(
                        box=len(new_gromet.b),
                        name=arg_name,
                        metadata=self.insert_metadata(
                            self.create_source_code_reference(arg_ref)
                        ),
                    ),
                )

            # Store each argument, its opi, and where it is in the opi table
            # For use when creating wfopi wires
            # Have to add 1 to the third value if we want to use it as an index reference
            arg_env[arg_name] = (
                arg,
                new_gromet.opi[-1],
                len(new_gromet.opi),
            )

        for var in var_environment["args"]:
            if new_gromet.opi != None and not var in [
                opi.name for opi in new_gromet.opi
            ]:
                new_gromet.opi = insert_gromet_object(
                    new_gromet.opi,
                    GrometPort(
                        box=len(new_gromet.b),
                        name=var,
                        metadata=self.insert_metadata(
                            self.create_source_code_reference(arg_ref)
                        ),
                    ),
                )
                arg_env[var] = (
                    var_environment["args"][var][0],
                    new_gromet.opi[-1],
                    len(new_gromet.opi),
                )

        # handle_function_def() will visit the body of the function and take care of
        # wiring any GroMEt FNs in its body
        self.handle_function_def(
            node, new_gromet, node.body, parent_cast_node=parent_cast_node
        )

        var_environment["args"] = deepcopy(prev_arg_env)

    @_visit.register
    def visit_literal_value(
        self, node: AnnCastLiteralValue, parent_gromet_fn, parent_cast_node
    ):
        if node.value_type == StructureType.TUPLE:
            self.visit_node_list(
                node.value, parent_gromet_fn, parent_cast_node
            )
        else:
            # Create the GroMEt literal value (A type of Function box)
            # This will have a single outport (the little blank box)
            # What we dont determine here is the wiring to whatever variable this
            # literal value goes to (that's up to the parent context)
            ref = node.source_code_data_type
            source_code_metadata = self.create_source_code_reference(
                node.source_refs[0]
            )

            code_data_metadata = SourceCodeDataType(
                metadata_type="source_code_data_type",
                provenance=generate_provenance(),
                source_language=ref[0],
                source_language_version=ref[1],
                data_type=str(ref[2]),
            )
            val = GLiteralValue(
                node.value_type if node.value_type is not None else "None",
                node.value if node.value is not None else "None",
            )

            parent_gromet_fn.bf = insert_gromet_object(
                parent_gromet_fn.bf,
                GrometBoxFunction(
                    function_type=FunctionType.LITERAL,
                    value=val,
                    metadata=self.insert_metadata(
                        code_data_metadata, source_code_metadata
                    ),
                ),
            )
            parent_gromet_fn.pof = insert_gromet_object(
                parent_gromet_fn.pof, GrometPort(box=len(parent_gromet_fn.bf))
            )

        # Perhaps we may need to return something in the future
        # an idea: the index of where this exists

    # node type: Loop or Condition
    def loop_create_condition(self, node, parent_gromet_fn, parent_cast_node):
        """
        Creates the condition field in a loop
        Steps:
        1. Create the predicate box
        2. Given all the vars, make opis and opos for them,
           and then wire them all together using wopio's
        3. Visit the node's conditional box and create everything as usual
            - (Add an extra check to the conditional visitor to make sure we don't double add)
        4. Add the extra exit condition port
        """
        # Step 1
        gromet_predicate_fn = GrometFN()
        self.gromet_module.fn_array = insert_gromet_object(
            self.gromet_module.fn_array,
            gromet_predicate_fn,
        )
        self.set_index()
        condition_array_idx = len(self.gromet_module.fn_array)

        # Step 2
        gromet_predicate_fn.b = insert_gromet_object(
            gromet_predicate_fn.b,
            GrometBoxFunction(function_type=FunctionType.PREDICATE),
        )
        for _, var_name in node.used_vars.items():
            gromet_predicate_fn.opi = insert_gromet_object(
                gromet_predicate_fn.opi,
                GrometPort(name=var_name, box=len(gromet_predicate_fn.b)),
            )

            gromet_predicate_fn.opo = insert_gromet_object(
                gromet_predicate_fn.opo,
                GrometPort(name=var_name, box=len(gromet_predicate_fn.b)),
            )

            gromet_predicate_fn.wopio = insert_gromet_object(
                gromet_predicate_fn.wopio,
                GrometWire(
                    src=len(gromet_predicate_fn.opi),
                    tgt=len(gromet_predicate_fn.opo),
                ),
            )

        # Step 3
        self.visit(node.expr, gromet_predicate_fn, node)  # visit condition

        # Step 4
        # Create the predicate's opo and wire it appropriately
        gromet_predicate_fn.opo = insert_gromet_object(
            gromet_predicate_fn.opo, GrometPort(box=len(gromet_predicate_fn.b))
        )
        gromet_predicate_fn.wfopo = insert_gromet_object(
            gromet_predicate_fn.wfopo,
            GrometWire(
                src=len(gromet_predicate_fn.opo),
                tgt=len(gromet_predicate_fn.pof),
            ),
        )

        return condition_array_idx

    def loop_create_body(self, node, parent_gromet_fn, parent_cast_node):
        """
        Creates a body FN for a loop

        """
        # The body section of the loop is itself a Gromet FN, so we create one and add it to our global list of FNs for this overall module
        gromet_body_fn = GrometFN()

        ref = node.body[0].source_refs[0]
        metadata = self.insert_metadata(self.create_source_code_reference(ref))

        gromet_body_fn.b = insert_gromet_object(
            gromet_body_fn.b,
            GrometBoxFunction(
                function_type=FunctionType.FUNCTION, metadata=metadata
            ),
        )
        self.gromet_module.fn_array = insert_gromet_object(
            self.gromet_module.fn_array, gromet_body_fn
        )
        self.set_index()

        body_array_idx = len(self.gromet_module.fn_array)
        var_environment = self.symtab_variables()

        # The 'call' bf for the body FN needs to have its pifs and pofs generated here as well
        # for (_, val) in node.used_vars.items():

        # Because the code in a loop body is technically a function on its own, we have to create a new
        # Variable environment for the local variables and function arguments
        # While preserving the old one
        # After we're done with the body of the loop, we restore the old environment
        previous_func_def_args = deepcopy(var_environment["args"])
        previous_local_args = deepcopy(var_environment["local"])

        var_environment["args"] = {}

        # The Gromet FN for the loop body needs to have its opis and opos generated here, since it isn't an actual FunctionDef here to make it with
        # Any opis we create for this Gromet FN are also added to the variable environment
        for _, val in node.used_vars.items():
            gromet_body_fn.opi = insert_gromet_object(
                gromet_body_fn.opi,
                GrometPort(name=val, box=len(gromet_body_fn.b)),
            )
            arg_env = var_environment["args"]
            arg_env[val] = (
                AnnCastFunctionDef(None, None, None, None),
                gromet_body_fn.opi[-1],
                len(gromet_body_fn.opi),
            )
            gromet_body_fn.opo = insert_gromet_object(
                gromet_body_fn.opo,
                GrometPort(name=val, box=len(gromet_body_fn.b)),
            )

        self.handle_function_def(
            AnnCastFunctionDef(None, None, None, None),
            gromet_body_fn,
            node.body,
        )

        # If the opo's name doesn't appear as a pof
        # then it hasn't been changed, create a wopio for it
        # Restore the old variable environment
        var_environment["args"] = previous_func_def_args
        var_environment["local"] = previous_local_args

        return body_array_idx

    def loop_create_post(self, node, parent_gromet_fn, parent_cast_node):
        # TODO
        pass

    @_visit.register
    def visit_loop(
        self, node: AnnCastLoop, parent_gromet_fn, parent_cast_node
    ):
        var_environment = self.symtab_variables()

        # Create empty gromet box loop that gets filled out before
        # being added to the parent gromet_fn
        gromet_bl = GrometBoxLoop()

        # Insert the gromet box loop into the parent gromet
        parent_gromet_fn.bl = insert_gromet_object(
            parent_gromet_fn.bl, gromet_bl
        )

        # Create the pil ports that the gromet box loop uses
        # Also, create any necessary wires that the pil uses
        for pil_idx, (_, val) in enumerate(node.used_vars.items(), 1):
            pil_port = GrometPort(name=val, box=len(parent_gromet_fn.bl))

            parent_gromet_fn.pil = insert_gromet_object(
                parent_gromet_fn.pil,
                pil_port,
            )

            port = self.retrieve_var_port(pil_port.name)
            if port != -1:
                if self.check_var_location(pil_port.name, "local"):
                    # Local variables manifest themselves through pofs
                    parent_gromet_fn.wfl = insert_gromet_object(
                        parent_gromet_fn.wfl, GrometWire(src=pil_idx, tgt=port)
                    )
                elif self.check_var_location(pil_port.name, "args"):
                    # Function arguments manifest themselves through opis
                    parent_gromet_fn.wlopi = insert_gromet_object(
                        parent_gromet_fn.wlopi,
                        GrometWire(src=pil_idx, tgt=port),
                    )
                elif self.check_var_location(pil_port.name, "global"):
                    # globals manifest themselves through opis or pofs depending
                    # on whether we're at the global scope or function def scope
                    # through an opi
                    if isinstance(parent_cast_node, AnnCastModule):
                        parent_gromet_fn.wfl = insert_gromet_object(
                            parent_gromet_fn.wfl,
                            GrometWire(src=pil_idx, tgt=port),
                        )
                    else:
                        parent_gromet_fn.wlopi = insert_gromet_object(
                            parent_gromet_fn.wlopi,
                            GrometWire(src=pil_idx, tgt=port),
                        )

        ######### Loop Pre (if one exists)
        if node.pre != None and len(node.pre) > 0:
            gromet_pre_fn = GrometFN()
            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array, gromet_pre_fn
            )
            self.set_index()

            pre_array_idx = len(self.gromet_module.fn_array)

            gromet_pre_fn.b = insert_gromet_object(
                gromet_pre_fn.b,
                GrometBoxFunction(function_type=FunctionType.FUNCTION),
            )

            # Copy the var environment, as we're in a 'function' of sorts
            # so we need a new var environment
            var_args_copy = deepcopy(var_environment["args"])
            var_local_copy = deepcopy(var_environment["local"])
            var_environment["args"] = {}
            var_environment["local"] = {}

            for _, val in node.used_vars.items():
                gromet_pre_fn.opi = insert_gromet_object(
                    gromet_pre_fn.opi, GrometPort(name=val, box=pre_array_idx)
                )

                var_environment["args"][val] = (
                    val,
                    gromet_pre_fn.opi[-1],
                    len(gromet_pre_fn.opi),
                )

                gromet_pre_fn.opo = insert_gromet_object(
                    gromet_pre_fn.opo, GrometPort(name=val, box=pre_array_idx)
                )

            for line in node.pre:
                # self.visit(line, gromet_pre_fn, parent_cast_node)
                self.visit(
                    line,
                    gromet_pre_fn,
                    AnnCastFunctionDef(None, None, None, None),
                )

            def find_opo_idx(gromet_fn, name):
                i = 1
                for opo in gromet_fn.opo:
                    if opo.name == name:
                        return i
                    i += 1
                return -1  # Not found

            # The pre GroMEt FN always has three OPOs to match up with the return values of the '_next' call
            # Create and wire the pofs to the OPOs
            gromet_port_name = gromet_pre_fn.pof[
                len(gromet_pre_fn.pof) - 3
            ].name
            gromet_pre_fn.wfopo = insert_gromet_object(
                gromet_pre_fn.wfopo,
                GrometWire(
                    src=find_opo_idx(gromet_pre_fn, gromet_port_name),
                    tgt=len(gromet_pre_fn.pof) - 2,
                ),
            )

            gromet_port_name = gromet_pre_fn.pof[
                len(gromet_pre_fn.pof) - 2
            ].name
            gromet_pre_fn.wfopo = insert_gromet_object(
                gromet_pre_fn.wfopo,
                GrometWire(
                    src=find_opo_idx(gromet_pre_fn, gromet_port_name),
                    tgt=len(gromet_pre_fn.pof) - 1,
                ),
            )

            gromet_port_name = gromet_pre_fn.pof[
                len(gromet_pre_fn.pof) - 1
            ].name
            gromet_pre_fn.wfopo = insert_gromet_object(
                gromet_pre_fn.wfopo,
                GrometWire(
                    src=find_opo_idx(gromet_pre_fn, gromet_port_name),
                    tgt=len(gromet_pre_fn.pof),
                ),
            )

            # Create wopios
            local_env = var_environment["local"]
            i = 1
            for opi in gromet_pre_fn.opi:
                if not opi.name in local_env.keys():
                    gromet_pre_fn.wopio = insert_gromet_object(
                        gromet_pre_fn.wopio, GrometWire(src=i, tgt=i)
                    )
                i += 1

            var_environment["args"] = var_args_copy
            var_environment["local"] = var_local_copy

            gromet_bl.pre = pre_array_idx

        ######### Loop Condition

        # This creates a predicate Gromet FN
        condition_array_idx = self.loop_create_condition(
            node, parent_gromet_fn, parent_cast_node
        )
        ref = node.expr.source_refs[0]
        # metadata = self.insert_metadata(self.create_source_code_reference(ref))

        # NOTE: gromet_bl and gromet_bc store indicies into the fn_array directly now
        gromet_bl.condition = condition_array_idx

        ######### Loop Body

        # The body section of the loop is itself a Gromet FN, so we create one and add it to our global list of FNs for this overall module
        gromet_bl.body = self.loop_create_body(
            node, parent_gromet_fn, parent_cast_node
        )
        # pols become 'locals' from this point on
        # That is, any code that is after the while loop should be looking at the pol ports to fetch data for
        # any variables that were used in the loop even if they weren't directly modified by it

        # post section of the loop, currently used in Fortran for loops
        if node.post != None and len(node.post) > 0:
            gromet_post_fn = GrometFN()
            self.gromet_module.fn_array = insert_gromet_object(
                self.gromet_module.fn_array, gromet_post_fn
            )
            self.set_index()

            post_array_idx = len(self.gromet_module.fn_array)

            gromet_post_fn.b = insert_gromet_object(
                gromet_post_fn.b,
                GrometBoxFunction(function_type=FunctionType.FUNCTION),
            )

            # Copy the var environment, as we're in a 'function' of sorts
            # so we need a new var environment
            var_args_copy = deepcopy(var_environment["args"])
            var_local_copy = deepcopy(var_environment["local"])
            var_environment["args"] = {}
            var_environment["local"] = {}

            for _, val in node.used_vars.items():
                gromet_post_fn.opi = insert_gromet_object(
                    gromet_post_fn.opi,
                    GrometPort(name=val, box=post_array_idx),
                )

                var_environment["args"][val] = (
                    val,
                    gromet_post_fn.opi[-1],
                    len(gromet_post_fn.opi),
                )

                gromet_post_fn.opo = insert_gromet_object(
                    gromet_post_fn.opo,
                    GrometPort(name=val, box=post_array_idx),
                )

            for line in node.post:
                self.visit(
                    line,
                    gromet_post_fn,
                    AnnCastFunctionDef(None, None, None, None),
                )

            # The pre GroMEt FN always has three OPOs to match up with the return values of the '_next' call
            # Create and wire the pofs to the OPOs

            # Create wopios
            local_env = var_environment["local"]
            i = 1
            for opi in gromet_post_fn.opi:
                if not opi.name in local_env.keys():
                    gromet_post_fn.wopio = insert_gromet_object(
                        gromet_post_fn.wopio, GrometWire(src=i, tgt=i)
                    )
                i += 1

            var_environment["args"] = var_args_copy
            var_environment["local"] = var_local_copy

            gromet_bl.post = post_array_idx

        for _, val in node.used_vars.items():
            parent_gromet_fn.pol = insert_gromet_object(
                parent_gromet_fn.pol,
                GrometPort(name=val, box=len(parent_gromet_fn.bl)),
            )
            self.add_var_to_env(
                val,
                AnnCastLoop(None, None, None, None, None),
                parent_gromet_fn.pol[-1],
                len(parent_gromet_fn.pol),
                node,
            )

    @_visit.register
    def visit_model_break(
        self, node: AnnCastModelBreak, parent_gromet_fn, parent_cast_node
    ):
        pass

    @_visit.register
    def visit_model_continue(
        self, node: AnnCastModelContinue, parent_gromet_fn, parent_cast_node
    ):
        pass

    def if_create_condition(
        self, node: AnnCastModelIf, parent_gromet_fn, parent_cast_node
    ):
        # This creates a predicate Gromet FN
        gromet_predicate_fn = GrometFN()
        self.gromet_module.fn_array = insert_gromet_object(
            self.gromet_module.fn_array, gromet_predicate_fn
        )
        self.set_index()

        condition_array_index = len(self.gromet_module.fn_array)

        gromet_predicate_fn.b = insert_gromet_object(
            gromet_predicate_fn.b,
            GrometBoxFunction(function_type=FunctionType.PREDICATE),
        )

        # Create all opis and opos for conditionals
        for _, val in node.used_vars.items():
            gromet_predicate_fn.opi = insert_gromet_object(
                gromet_predicate_fn.opi,
                GrometPort(name=val, box=len(gromet_predicate_fn.b)),
            )

            gromet_predicate_fn.opo = insert_gromet_object(
                gromet_predicate_fn.opo,
                GrometPort(name=val, box=len(gromet_predicate_fn.b)),
            )

        # Create wopios
        if gromet_predicate_fn.opi != None and gromet_predicate_fn.opo != None:
            i = 1
            while i - 1 < len(gromet_predicate_fn.opi) and i - 1 < len(
                gromet_predicate_fn.opo
            ):
                gromet_predicate_fn.wopio = insert_gromet_object(
                    gromet_predicate_fn.wopio, GrometWire(src=i, tgt=i)
                )
                i += 1

        self.visit(node.expr, gromet_predicate_fn, node)

        # Create the predicate's opo and wire it appropriately
        gromet_predicate_fn.opo = insert_gromet_object(
            gromet_predicate_fn.opo, GrometPort(box=len(gromet_predicate_fn.b))
        )

        # TODO: double check this guard to see if it's necessary
        if isinstance(node.expr, AnnCastModelIf):
            for i, _ in enumerate(gromet_predicate_fn.opi, 1):
                gromet_predicate_fn.wcopi = insert_gromet_object(
                    gromet_predicate_fn.wcopi, GrometWire(src=i, tgt=i)
                )

            gromet_predicate_fn.poc = insert_gromet_object(
                gromet_predicate_fn.poc,
                GrometPort(box=len(gromet_predicate_fn.bc)),
            )

            for i, _ in enumerate(gromet_predicate_fn.opo, 1):
                gromet_predicate_fn.wcopo = insert_gromet_object(
                    gromet_predicate_fn.wcopo, GrometWire(src=i, tgt=i)
                )
        else:
            if (
                gromet_predicate_fn.opo == None
                and gromet_predicate_fn.pof == None
            ):
                gromet_predicate_fn.wfopo = insert_gromet_object(
                    gromet_predicate_fn.wfopo, GrometWire(src=-1, tgt=-11112)
                )
            elif gromet_predicate_fn.pof == None:
                gromet_predicate_fn.wfopo = insert_gromet_object(
                    gromet_predicate_fn.wfopo,
                    GrometWire(src=len(gromet_predicate_fn.opo), tgt=-1111),
                )
            elif gromet_predicate_fn.opo == None:
                gromet_predicate_fn.wfopo = insert_gromet_object(
                    gromet_predicate_fn.wfopo,
                    GrometWire(src=-11113, tgt=len(gromet_predicate_fn.pof)),
                )
            else:
                gromet_predicate_fn.wfopo = insert_gromet_object(
                    gromet_predicate_fn.wfopo,
                    GrometWire(
                        src=len(gromet_predicate_fn.opo),
                        tgt=len(gromet_predicate_fn.pof),
                    ),
                )

        ref = node.expr.source_refs[0]
        metadata = self.insert_metadata(self.create_source_code_reference(ref))
        gromet_predicate_fn.metadata = metadata

        return condition_array_index

    def if_create_body(
        self, node: AnnCastModelIf, parent_gromet_fn, parent_cast_node
    ):
        body_if_fn = GrometFN()
        body_if_fn.b = insert_gromet_object(
            body_if_fn.b,
            GrometBoxFunction(function_type=FunctionType.FUNCTION),
        )
        self.gromet_module.fn_array = insert_gromet_object(
            self.gromet_module.fn_array, body_if_fn
        )
        self.set_index()

        body_if_idx = len(self.gromet_module.fn_array)

        ref = node.body[0].source_refs[0]
        var_environment = self.symtab_variables()

        # NOTE: might change this to an if/elif if it's proven that
        # both an "And" and an "Or" can't exist at the same time in both parts
        # of the if statement
        # Having a boolean literal value in the node body implies a True value which means we have an Or
        # Having a boolean literal value in the node orelse implies a False value which means we have an And
        and_or_metadata = None
        if (
            len(node.body) > 0
            and isinstance(node.body[0], AnnCastLiteralValue)
            and node.body[0].value_type == ScalarType.BOOLEAN
        ):
            and_or_metadata = SourceCodeBoolOr()

        if and_or_metadata != None:
            metadata = self.insert_metadata(
                self.create_source_code_reference(ref), and_or_metadata
            )
        else:
            metadata = self.insert_metadata(
                self.create_source_code_reference(ref)
            )

        body_if_fn.metadata = metadata
        # copy the old var environments over since we're going into a function
        previous_func_def_args = deepcopy(var_environment["args"])
        previous_local_args = deepcopy(var_environment["local"])

        var_environment["args"] = {}

        # TODO: determine a better for loop that only grabs
        # what appears in the body of the if_true
        # for (_, val) in node.expr_used_vars.items():
        for _, val in node.used_vars.items():
            body_if_fn.opi = insert_gromet_object(
                body_if_fn.opi, GrometPort(box=len(body_if_fn.b))
            )
            arg_env = var_environment["args"]
            arg_env[val] = (
                AnnCastFunctionDef(None, None, None, None),
                body_if_fn.opi[-1],
                len(body_if_fn.opi),
            )

            body_if_fn.opo = insert_gromet_object(
                body_if_fn.opo, GrometPort(name=val, box=len(body_if_fn.b))
            )

        self.handle_function_def(
            AnnCastFunctionDef(None, None, None, None), body_if_fn, node.body
        )

        if (
            len(node.body) > 0
            and isinstance(node.body[0], AnnCastLiteralValue)
            and node.body[0].value_type == ScalarType.BOOLEAN
        ):
            body_if_fn.opo = insert_gromet_object(
                body_if_fn.opo, GrometPort(box=len(body_if_fn.b))
            )
            body_if_fn.wfopo = insert_gromet_object(
                body_if_fn.wfopo,
                GrometWire(src=len(body_if_fn.opo), tgt=len(body_if_fn.pof)),
            )

        if (
            len(node.body) > 0
            and isinstance(node.body[0], AnnCastOperator)
            and node.body[0].op
            in (
                "ast.Eq",
                "ast.NotEq",
                "ast.Lt",
                "ast.LtE",
                "ast.Gt",
                "ast.GtE",
            )
        ):
            body_if_fn.opo = insert_gromet_object(
                body_if_fn.opo, GrometPort(box=len(body_if_fn.b))
            )
            body_if_fn.wfopo = insert_gromet_object(
                body_if_fn.wfopo,
                GrometWire(src=len(body_if_fn.opo), tgt=len(body_if_fn.pof)),
            )

        # restore previous var environments
        var_environment["args"] = previous_func_def_args
        var_environment["local"] = previous_local_args

        return body_if_idx

    def if_create_orelse(
        self, node: AnnCastModelIf, parent_gromet_fn, parent_cast_node
    ):
        orelse_if_fn = GrometFN()
        orelse_if_fn.b = insert_gromet_object(
            orelse_if_fn.b,
            GrometBoxFunction(function_type=FunctionType.FUNCTION),
        )
        self.gromet_module.fn_array = insert_gromet_object(
            self.gromet_module.fn_array, orelse_if_fn
        )
        self.set_index()

        orelse_if_idx = len(self.gromet_module.fn_array)
        ref = node.orelse[0].source_refs[0]
        var_environment = self.symtab_variables()

        # NOTE: might change this to an if/elif if it's proven that
        # both an "And" and an "Or" can't exist at the same time in both parts
        # of the if statement
        # Having a boolean literal value in the node orelse implies a False value which means we have an And
        and_or_metadata = None
        if (
            len(node.orelse) > 0
            and isinstance(node.orelse[0], AnnCastLiteralValue)
            and node.orelse[0].value_type == ScalarType.BOOLEAN
        ):
            and_or_metadata = SourceCodeBoolAnd()

        if and_or_metadata != None:
            metadata = self.insert_metadata(
                self.create_source_code_reference(ref), and_or_metadata
            )
        else:
            metadata = self.insert_metadata(
                self.create_source_code_reference(ref)
            )

        orelse_if_fn.metadata = metadata
        # copy the old var environments over since we're going into a function
        previous_func_def_args = deepcopy(var_environment["args"])
        previous_local_args = deepcopy(var_environment["local"])

        var_environment["args"] = {}

        # TODO: determine a better for loop that only grabs
        # what appears in the orelse of the if_true
        # for (_, val) in node.expr_used_vars.items():
        for _, val in node.used_vars.items():
            orelse_if_fn.opi = insert_gromet_object(
                orelse_if_fn.opi, GrometPort(box=len(orelse_if_fn.b))
            )
            arg_env = var_environment["args"]
            arg_env[val] = (
                AnnCastFunctionDef(None, None, None, None),
                orelse_if_fn.opi[-1],
                len(orelse_if_fn.opi),
            )

            orelse_if_fn.opo = insert_gromet_object(
                orelse_if_fn.opo, GrometPort(name=val, box=len(orelse_if_fn.b))
            )

        self.handle_function_def(
            AnnCastFunctionDef(None, None, None, None),
            orelse_if_fn,
            node.orelse,
        )

        if (
            len(node.orelse) > 0
            and isinstance(node.orelse[0], AnnCastLiteralValue)
            and node.orelse[0].value_type == ScalarType.BOOLEAN
        ):
            orelse_if_fn.opo = insert_gromet_object(
                orelse_if_fn.opo, GrometPort(box=len(orelse_if_fn.b))
            )
            orelse_if_fn.wfopo = insert_gromet_object(
                orelse_if_fn.wfopo,
                GrometWire(
                    src=len(orelse_if_fn.opo), tgt=len(orelse_if_fn.pof)
                ),
            )

        if (
            len(node.orelse) > 0
            and isinstance(node.orelse[0], AnnCastOperator)
            and node.orelse[0].op
            in (
                "ast.Eq",
                "ast.NotEq",
                "ast.Lt",
                "ast.LtE",
                "ast.Gt",
                "ast.GtE",
            )
        ):
            orelse_if_fn.opo = insert_gromet_object(
                orelse_if_fn.opo, GrometPort(box=len(orelse_if_fn.b))
            )
            orelse_if_fn.wfopo = insert_gromet_object(
                orelse_if_fn.wfopo,
                GrometWire(
                    src=len(orelse_if_fn.opo), tgt=len(orelse_if_fn.pof)
                ),
            )

        # restore previous var environments
        var_environment["args"] = previous_func_def_args
        var_environment["local"] = previous_local_args

        return orelse_if_idx

    @_visit.register
    def visit_model_if(
        self, node: AnnCastModelIf, parent_gromet_fn, parent_cast_node
    ):
        ref = node.source_refs[0]
        metadata = self.insert_metadata(self.create_source_code_reference(ref))
        gromet_bc = GrometBoxConditional(metadata=metadata)

        parent_gromet_fn.bc = insert_gromet_object(
            parent_gromet_fn.bc, gromet_bc
        )

        bc_index = len(parent_gromet_fn.bc)

        for _, val in node.used_vars.items():
            parent_gromet_fn.pic = insert_gromet_object(
                parent_gromet_fn.pic,
                GrometPort(name=val, box=len(parent_gromet_fn.bc)),
            )

            parent_gromet_fn.poc = insert_gromet_object(
                parent_gromet_fn.poc,
                GrometPort(name=val, box=len(parent_gromet_fn.bc)),
            )

        # TODO: We also need to put this around a loop
        # And in particular we only want to make wires to variables that are used in the conditional
        # Check type of parent_cast_node to determine which wire to create
        # TODO: Previously, we were always generating a wfc wire for variables coming into a conditional
        # However, we can also have variables coming in from other sources such as an opi.
        # This is a temporary fix for the specific case in the CHIME model, but will need to be revisited
        if isinstance(parent_cast_node, AnnCastFunctionDef):
            if (
                parent_gromet_fn.pic == None and parent_gromet_fn.opi == None
            ):  # TODO: double check this guard to see if it's necessary
                # print(node.source_refs[0])
                parent_gromet_fn.wcopi = insert_gromet_object(
                    parent_gromet_fn.wcopi, GrometWire(src=-1, tgt=-1)
                )
            elif parent_gromet_fn.opi == None:
                # print(node.source_refs[0])
                parent_gromet_fn.wcopi = insert_gromet_object(
                    parent_gromet_fn.wcopi,
                    GrometWire(src=len(parent_gromet_fn.pic), tgt=-1),
                )
            elif parent_gromet_fn.pic == None:
                # print(node.source_refs[0])
                parent_gromet_fn.wcopi = insert_gromet_object(
                    parent_gromet_fn.wcopi,
                    GrometWire(src=-1, tgt=len(parent_gromet_fn.opi)),
                )
            else:
                for pic_idx, pic in enumerate(parent_gromet_fn.pic, 1):
                    if pic.box == bc_index:
                        port = self.retrieve_var_port(pic.name)
                        if port != -1:
                            if self.check_var_location(pic.name, "local"):
                                parent_gromet_fn.wfc = insert_gromet_object(
                                    parent_gromet_fn.wfc,
                                    GrometWire(src=pic_idx, tgt=port),
                                )
                            elif self.check_var_location(pic.name, "args"):
                                parent_gromet_fn.wcopi = insert_gromet_object(
                                    parent_gromet_fn.wcopi,
                                    GrometWire(src=pic_idx, tgt=port),
                                )
                            elif self.check_var_location(pic.name, "global"):
                                parent_gromet_fn.wfc = insert_gromet_object(
                                    parent_gromet_fn.wfc,
                                    GrometWire(src=pic_idx, tgt=port),
                                )

        if isinstance(parent_cast_node, AnnCastModule):
            for pic_idx, pic in enumerate(parent_gromet_fn.pic, 1):
                port = self.retrieve_var_port(pic.name)
                if port != -1:
                    if self.check_var_location(pic.name, "local"):
                        parent_gromet_fn.wfc = insert_gromet_object(
                            parent_gromet_fn.wfc,
                            GrometWire(src=pic_idx, tgt=port),
                        )
                    elif self.check_var_location(pic.name, "args"):
                        parent_gromet_fn.wcopi = insert_gromet_object(
                            parent_gromet_fn.wcopi,
                            GrometWire(src=pic_idx, tgt=port),
                        )
                    elif self.check_var_location(pic.name, "global"):
                        parent_gromet_fn.wfc = insert_gromet_object(
                            parent_gromet_fn.wfc,
                            GrometWire(src=pic_idx, tgt=port),
                        )

        gromet_bc.condition = self.if_create_condition(
            node, parent_gromet_fn, parent_cast_node
        )

        ########### If true generation
        gromet_bc.body_if = self.if_create_body(
            node, parent_gromet_fn, parent_cast_node
        )

        ########### If false generation
        if (
            len(node.orelse) > 0
        ):  # NOTE: guards against when there's no else to the if statement
            gromet_bc.body_else = self.if_create_orelse(
                node, parent_gromet_fn, parent_cast_node
            )

    def add_import_symbol_to_env(
        self, symbol, parent_gromet_fn, parent_cast_node
    ):
        """
        Adds symbol to the GroMEt FN as a 'variable'
        When we import something from another file with a symbol,
        we don't know the symbol is a function call or variable
        so we add in a 'dummy' variable of sorts so that it can
        be used in this file
        """

        parent_gromet_fn.bf = insert_gromet_object(
            parent_gromet_fn.bf,
            GrometBoxFunction(
                function_type=FunctionType.EXPRESSION, name=symbol, body=None
            ),
        )

        bf_idx = len(parent_gromet_fn.bf)

        parent_gromet_fn.pof = insert_gromet_object(
            parent_gromet_fn.pof, GrometPort(name=symbol, box=bf_idx)
        )

        pof_idx = len(parent_gromet_fn.pof)

        self.add_var_to_env(
            symbol,
            None,
            parent_gromet_fn.pof[pof_idx - 1],
            pof_idx,
            parent_cast_node,
        )

    @_visit.register
    def visit_model_import(
        self, node: AnnCastModelImport, parent_gromet_fn, parent_cast_node
    ):
        name = node.name
        alias = node.alias
        symbol = node.symbol
        all = node.all

        # self.import collection maintains a dictionary of
        # name:(alias, [symbols], all boolean flag)
        # pairs that we can use to look up later
        if (
            name in self.import_collection
        ):  # If this import already exists, then perhaps we add a new symbol to its list of symbols
            if symbol != None:
                if self.import_collection[name][1] == None:
                    self.import_collection[name] = (
                        self.import_collection[name][0],
                        [],
                        self.import_collection[name][2],
                    )
                self.import_collection[name][1].append(symbol)
                # We also maintain the symbol as a 'variable' of sorts in the global environment
                self.add_import_symbol_to_env(
                    symbol, parent_gromet_fn, parent_cast_node
                )

            self.import_collection[name] = (
                self.import_collection[name][0],
                self.import_collection[name][1],
                all,
            )
            # self.import_collection[name][2] = all # Update the all field if necessary
        else:  # Otherwise we haven't seen this import yet and we add its fields and potential symbol accordingly
            if symbol == None:
                self.import_collection[name] = (alias, [], all)
            else:
                self.import_collection[name] = (alias, [symbol], all)
                # We also maintain the symbol as a 'variable' of sorts in the global environment
                self.add_import_symbol_to_env(
                    symbol, parent_gromet_fn, parent_cast_node
                )

    @_visit.register
    def visit_model_return(
        self, node: AnnCastModelReturn, parent_gromet_fn, parent_cast_node
    ):
        # if not isinstance(node.value, AnnCastTuple):
        if not is_tuple(node.value):
            self.visit(node.value, parent_gromet_fn, node)
        ref = node.source_refs[0]

        # A binary op sticks a single return value in the opo
        # Where as a tuple can stick multiple opos, one for each thing being returned
        # NOTE: The above comment about tuples is outdated, as we now pack the tuple's values into a pack, and return one
        # value with that
        if isinstance(node.value, AnnCastOperator):
            parent_gromet_fn.opo = insert_gromet_object(
                parent_gromet_fn.opo,
                GrometPort(
                    box=len(parent_gromet_fn.b),
                    metadata=self.insert_metadata(
                        self.create_source_code_reference(ref)
                    ),
                ),
            )
        # elif isinstance(node.value, AnnCastTuple):
        elif is_tuple(node.value):
            parent_gromet_fn.opo = insert_gromet_object(
                parent_gromet_fn.opo,
                GrometPort(
                    box=len(parent_gromet_fn.b),
                    metadata=self.insert_metadata(
                        self.create_source_code_reference(ref)
                    ),
                ),
            )
            # for elem in node.value.values:
            #   parent_gromet_fn.opo = insert_gromet_object(parent_gromet_fn.opo, GrometPort(box=len(parent_gromet_fn.b),metadata=self.insert_metadata(self.create_source_code_reference(ref))))

    @_visit.register
    def visit_module(
        self, node: AnnCastModule, parent_gromet_fn, parent_cast_node
    ):
        # We create a new GroMEt FN and add it to the GroMEt FN collection

        # Creating a new Function Network (FN) where the outer box is a module
        # i.e. a gray colored box in the drawings
        # It's like any FN but it doesn't have any outer ports, or inner/outer port boxes
        # on it (i.e. little squares on the gray box in a drawing)

        file_name = node.source_refs[0].source_file_name
        var_environment = self.symtab_variables()
        var_environment["global"] = {}

        # Have a FN constructor to build the GroMEt FN
        # and pass this FN to maintain a 'nesting' approach (boxes within boxes)
        # instead of passing a GrFNSubgraph through the visitors
        new_gromet = GrometFN()

        # Initialie the Gromet module's Record Bookkeeping metadata
        # Which lives in the very first element of the metadata array
        self.gromet_module.metadata_collection = [[]]
        self.gromet_module.metadata = 0

        # Initialize the Gromet module's SourceCodeCollection of CodeFileReferences
        code_file_references = [
            CodeFileReference(uid=str(uuid.uuid4()), name=file_name, path="")
        ]
        self.gromet_module.metadata = self.insert_metadata(
            SourceCodeCollection(
                provenance=generate_provenance(),
                name="",
                global_reference_id="",
                files=code_file_references,
            ),
            GrometCreation(provenance=generate_provenance()),
        )

        # Outer module box only has name 'module' and its type 'Module'
        new_gromet.b = insert_gromet_object(
            new_gromet.b,
            GrometBoxFunction(
                name="module",
                function_type=FunctionType.MODULE,
                metadata=self.insert_metadata(
                    self.create_source_code_reference(node.source_refs[0])
                ),
            ),
        )

        # Module level GroMEt FN sits in its own special field dicating the module node
        self.gromet_module.fn = new_gromet

        # Set the name of the outer Gromet module to be the source file name
        self.gromet_module.name = os.path.basename(file_name).replace(
            ".py", ""
        )

        self.build_function_arguments_table(node.body)

        self.visit_node_list(node.body, new_gromet, node)

        var_environment["global"] = {}

    @_visit.register
    def visit_name(
        self, node: AnnCastName, parent_gromet_fn, parent_cast_node
    ):
        # NOTE: Maybe make wfopi between the function input and where it's being used

        # If this name access comes from a return node then we make the opo for the GroMEt FN that this
        # return is in
        if isinstance(parent_cast_node, AnnCastModelReturn):
            parent_gromet_fn.opo = insert_gromet_object(
                parent_gromet_fn.opo, GrometPort(box=len(parent_gromet_fn.b))
            )

    @_visit.register
    def visit_record_def(
        self, node: AnnCastRecordDef, parent_gromet_fn, parent_cast_node
    ):
        record_name = node.name
        record_methods = []  # strings (method names)
        record_fields = {}  # field:method_name

        self.symbol_table["records"][record_name] = record_name
        var_environment = self.symtab_variables()

        # Find 'init' and create a special new:Object function for it
        # Repeat with the getters I think?
        f = None
        for f in node.funcs:
            if isinstance(f, AnnCastFunctionDef) and f.name.name == "__init__":
                record_methods.append("__init__")
                break

        new_gromet = GrometFN()
        self.gromet_module.fn_array = insert_gromet_object(
            self.gromet_module.fn_array, new_gromet
        )
        self.set_index()

        # Because "new:Record" is a function definition itself we
        # need to maintain an argument environment for it
        # store copies of previous ones and create new ones
        arg_env_copy = deepcopy(var_environment["args"])
        local_env_copy = deepcopy(var_environment["local"])

        var_environment["args"] = {}

        # Generate the init new:ClassName FN
        new_gromet.b = insert_gromet_object(
            new_gromet.b,
            GrometBoxFunction(
                name=f"new:{node.name}", function_type=FunctionType.FUNCTION
            ),
        )
        if f != None:
            for arg in f.func_args:
                if arg.val.name != "self":
                    new_gromet.opi = insert_gromet_object(
                        new_gromet.opi,
                        GrometPort(name=arg.val.name, box=len(new_gromet.b)),
                    )
                    var_environment["args"][arg.val.name] = (
                        arg,
                        new_gromet.opi[-1],
                        len(new_gromet.opi),
                    )

        # We maintain an additional 'obj' field that is used in the case that we inherit a parent class
        new_gromet.opi = insert_gromet_object(
            new_gromet.opi, GrometPort(name="obj", box=len(new_gromet.b))
        )
        var_environment["args"]["obj"] = (
            None,
            new_gromet.opi[-1],
            len(new_gromet.opi),
        )
        new_gromet.opo = insert_gromet_object(
            new_gromet.opo, GrometPort(box=len(new_gromet.b))
        )

        # The first value that goes into the "new_Record" primitive is the name of the class
        new_gromet.bf = insert_gromet_object(
            new_gromet.bf,
            GrometBoxFunction(
                function_type=FunctionType.LITERAL,
                value=GLiteralValue("string", node.name),
            ),
        )
        new_gromet.pof = insert_gromet_object(
            new_gromet.pof, GrometPort(box=len(new_gromet.bf))
        )

        # Create the initial constructor function and wire it accordingly
        inline_new_record = GrometBoxFunction(
            name="new_Record", function_type=FunctionType.ABSTRACT
        )
        new_gromet.bf = insert_gromet_object(new_gromet.bf, inline_new_record)
        new_record_idx = len(new_gromet.bf)

        # Create the first port for "new_Record" and wire the first value created earlier
        new_gromet.pif = insert_gromet_object(
            new_gromet.pif, GrometPort(box=new_record_idx)
        )
        new_gromet.wff = insert_gromet_object(
            new_gromet.wff,
            GrometWire(src=len(new_gromet.pif), tgt=len(new_gromet.pof)),
        )

        # The second value that goes into the "new_Record" primitive is either the name of the superclass or None
        # Checking if we have a superclass (parent class) or not
        if len(node.bases) == 0:
            new_gromet.bf = insert_gromet_object(
                new_gromet.bf,
                GrometBoxFunction(
                    function_type=FunctionType.LITERAL,
                    value=GLiteralValue("None", "None"),
                ),
            )
            new_gromet.pof = insert_gromet_object(
                new_gromet.pof, GrometPort(box=len(new_gromet.bf))
            )
            new_gromet.pif = insert_gromet_object(
                new_gromet.pif, GrometPort(box=new_record_idx)
            )
            new_gromet.wff = insert_gromet_object(
                new_gromet.wff,
                GrometWire(src=len(new_gromet.pif), tgt=len(new_gromet.pof)),
            )
        else:
            base = node.bases[0]
            name = ""
            if isinstance(base, AnnCastAttribute):
                name = base.attr
            else:
                name = base.name
            new_gromet.bf = insert_gromet_object(
                new_gromet.bf,
                GrometBoxFunction(
                    function_type=FunctionType.LITERAL,
                    value=GLiteralValue("string", name),
                ),
            )
            new_gromet.pof = insert_gromet_object(
                new_gromet.pof, GrometPort(box=len(new_gromet.bf))
            )
            new_gromet.pif = insert_gromet_object(
                new_gromet.pif, GrometPort(box=new_record_idx)
            )
            new_gromet.wff = insert_gromet_object(
                new_gromet.wff,
                GrometWire(src=len(new_gromet.pif), tgt=len(new_gromet.pof)),
            )

        # Add the third argument to new_Record, which is the obj argument
        new_gromet.pif = insert_gromet_object(
            new_gromet.pif, GrometPort(box=new_record_idx)
        )
        new_gromet.wfopi = insert_gromet_object(
            new_gromet.wfopi,
            GrometWire(
                src=len(new_gromet.pif),
                tgt=var_environment["args"]["obj"][2],
            ),
        )

        # pof for "new_Record"
        new_gromet.pof = insert_gromet_object(
            new_gromet.pof, GrometPort(box=new_record_idx)
        )

        if f != None:
            for s in f.body:
                if (
                    isinstance(s, AnnCastAssignment)
                    and isinstance(s.left, AnnCastAttribute)
                    and s.left.value.name == "self"
                ):
                    record_fields[s.left.attr.name] = record_name

                    inline_new_record = GrometBoxFunction(
                        name="new_Field", function_type=FunctionType.ABSTRACT
                    )
                    new_gromet.bf = insert_gromet_object(
                        new_gromet.bf, inline_new_record
                    )
                    new_field_idx = len(new_gromet.bf)

                    # Wire first pif of "new_field" which relies on the previous pof of "new_record" or a previous "set" call
                    new_gromet.pif = insert_gromet_object(
                        new_gromet.pif, GrometPort(box=new_field_idx)
                    )
                    new_gromet.wff = insert_gromet_object(
                        new_gromet.wff,
                        GrometWire(
                            src=len(new_gromet.pif), tgt=len(new_gromet.pof)
                        ),
                    )

                    # Second pif of "new_field"/"set" involves this variable and its pof
                    new_gromet.bf = insert_gromet_object(
                        new_gromet.bf,
                        GrometBoxFunction(
                            function_type=FunctionType.LITERAL,
                            value=GLiteralValue("string", s.left.attr.name),
                        ),
                    )
                    new_gromet.pof = insert_gromet_object(
                        new_gromet.pof, GrometPort(box=len(new_gromet.bf))
                    )

                    var_pof = len(new_gromet.pof)

                    # Second argument to "new_Field"
                    new_gromet.pif = insert_gromet_object(
                        new_gromet.pif, GrometPort(box=new_field_idx)
                    )
                    new_gromet.wff = insert_gromet_object(
                        new_gromet.wff,
                        GrometWire(src=len(new_gromet.pif), tgt=var_pof),
                    )
                    new_gromet.pof = insert_gromet_object(
                        new_gromet.pof, GrometPort(box=new_field_idx)
                    )

                    # Create set
                    record_set = GrometBoxFunction(
                        name="set", function_type=FunctionType.ABSTRACT
                    )
                    # Wires first arg for "set"
                    new_gromet.bf = insert_gromet_object(
                        new_gromet.bf, record_set
                    )
                    record_set_idx = len(new_gromet.bf)
                    new_gromet.pif = insert_gromet_object(
                        new_gromet.pif, GrometPort(box=record_set_idx)
                    )
                    new_gromet.wff = insert_gromet_object(
                        new_gromet.wff,
                        GrometWire(
                            src=len(new_gromet.pif), tgt=len(new_gromet.pof)
                        ),
                    )

                    # Wires second arg for "set"
                    new_gromet.pif = insert_gromet_object(
                        new_gromet.pif, GrometPort(box=record_set_idx)
                    )
                    new_gromet.wff = insert_gromet_object(
                        new_gromet.wff,
                        GrometWire(src=len(new_gromet.pif), tgt=var_pof),
                    )

                    # Create third argument for "set"
                    new_gromet.pif = insert_gromet_object(
                        new_gromet.pif, GrometPort(box=record_set_idx)
                    )
                    set_third_arg = len(new_gromet.pif)

                    # Wire the last argument for "set" depending on what it is
                    if isinstance(s.right, AnnCastName):
                        # Find argument opi for "set" third argument
                        if (
                            new_gromet.opi != None
                        ):  # TODO: Fix it so opis aren't ever None
                            for opi_i, opi in enumerate(new_gromet.opi, 1):
                                if (
                                    isinstance(s.right, AnnCastName)
                                    and opi.name == s.right.name
                                ):
                                    break

                            new_gromet.wfopi = insert_gromet_object(
                                new_gromet.wfopi,
                                GrometWire(src=set_third_arg, tgt=opi_i),
                            )

                    else:
                        # The visitor sets a pof that we have to wire
                        self.visit(s.right, new_gromet, parent_cast_node)

                        new_gromet.wff = insert_gromet_object(
                            new_gromet.wff,
                            GrometWire(
                                src=set_third_arg, tgt=len(new_gromet.pof)
                            ),
                        )

                    # Output port for "set"
                    new_gromet.pof = insert_gromet_object(
                        new_gromet.pof, GrometPort(box=record_set_idx)
                    )

        # Wire output wire for "new:Record"
        new_gromet.wfopo = insert_gromet_object(
            new_gromet.wfopo,
            GrometWire(src=len(new_gromet.opo), tgt=len(new_gromet.pof)),
        )

        # Need to store the index of where "new:Record" is in the GroMEt table
        # in the record table
        self.record[node.name] = {}
        self.record[node.name][f"new:{node.name}"] = len(
            self.gromet_module.fn_array
        )

        var_environment["args"] = deepcopy(arg_env_copy)
        var_environment["local"] = deepcopy(local_env_copy)

        # Generate and store the rest of the functions associated with this record
        for f in node.funcs:
            if isinstance(f, AnnCastFunctionDef) and f.name.name != "__init__":
                arg_env_copy = deepcopy(var_environment["args"])
                local_env_copy = deepcopy(var_environment["local"])
                var_environment["args"] = {}

                # This is a new function, so  create a GroMEt FN
                new_gromet = GrometFN()
                self.gromet_module.fn_array = insert_gromet_object(
                    self.gromet_module.fn_array, new_gromet
                )
                self.set_index()

                # Create its name and its arguments
                new_gromet.b = insert_gromet_object(
                    new_gromet.b,
                    GrometBoxFunction(
                        name=f"{node.name}:{f.name.name}",
                        function_type=FunctionType.FUNCTION,
                    ),
                )

                record_methods.append(f.name.name)
                for arg in f.func_args:
                    new_gromet.opi = insert_gromet_object(
                        new_gromet.opi,
                        GrometPort(name=arg.val.name, box=len(new_gromet.b)),
                    )
                    var_environment["args"][arg.val.name] = (
                        arg,
                        new_gromet.opi[-1],
                        len(new_gromet.opi),
                    )
                new_gromet.opo = insert_gromet_object(
                    new_gromet.opo, GrometPort(box=len(new_gromet.b))
                )

                for s in f.body:
                    self.visit(
                        s,
                        new_gromet,
                        AnnCastFunctionDef(None, None, None, None),
                    )

                if new_gromet.pof != None:
                    new_gromet.wfopo = insert_gromet_object(
                        new_gromet.wfopo,
                        GrometWire(
                            src=len(new_gromet.opo), tgt=len(new_gromet.pof)
                        ),
                    )
                else:
                    new_gromet.wfopo = insert_gromet_object(
                        new_gromet.wfopo,
                        GrometWire(src=len(new_gromet.opo), tgt=-1),
                    )

                var_environment["args"] = deepcopy(arg_env_copy)
                var_environment["local"] = deepcopy(local_env_copy)

                self.record[node.name][f.name.name] = len(
                    self.gromet_module.fn_array
                )

        record_metadata = ProgramAnalysisRecordBookkeeping(
            provenance=generate_provenance(),
            type_name=record_name,
            field_declarations=record_fields,
            method_declarations=record_methods,
        )

        self.insert_record_info(record_metadata)

    @_visit.register
    def visit_tuple(
        self, node: AnnCastTuple, parent_gromet_fn, parent_cast_node
    ):
        self.visit_node_list(node.values, parent_gromet_fn, parent_cast_node)

    @_visit.register
    def visit_var(self, node: AnnCastVar, parent_gromet_fn, parent_cast_node):
        self.visit(node.val, parent_gromet_fn, parent_cast_node)

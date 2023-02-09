import pprint
import json
from dataclasses import dataclass, field
from xml.etree import cElementTree
from typing import Tuple, List, Set, Dict

# If you install https://pylatexenc.readthedocs.io/en/latest/latexencode/
#    then you can uncomment lines 9 and 10 and comment-out line.
# UNICODE_TO_LATEX = True
# from pylatexenc.latexencode import unicode_to_latex
def unicode_to_latex(s): return s

# Latex to MathML
# https://temml.org/


# -----------------------------------------------------------------------------

"""
TODO

() Extend to allow for seeding the species and rate sets
() Allow for taking existing equations and then recalculating with new species/rates

# JSON PN export
{
  "species": [str...],
  "rates": [str...]
}

-----

Revisit Bucky dynamics


"""


# -----------------------------------------------------------------------------

V1 = 'mml/simple_sir_v1/mml_list.txt'
V2 = 'mml/simple_sir_v2/mml_list.txt'
V3 = 'mml/simple_sir_v3/mml_list.txt'
V4 = 'mml/simple_sir_v4/mml_list.txt'


# -----------------------------------------------------------------------------
# Older code
# The following represents initial flailing, but being preserved just walking
# and displaying the mml (show_xml)
# -----------------------------------------------------------------------------

def read_xml_et(filepath):
    with open(filepath, 'r') as fin:
        xml = fin.read()
        return cElementTree.XML(xml)


def walk_mml_et(et, indent=0):
    idt = ' '*indent
    for elm in et:
        print(f'{idt}{elm} {elm.tag} {elm.attrib} text=\'{elm.text.strip()}\'')
        walk_mml_et(elm, indent=indent + 2)


def walk_mml_et_2(et, indent=0):
    idt = ' '*indent
    if et.tag == 'mfrac':
        parse_mfrac(et)
    elif et.tag == 'mi':
        print(f'elm: {et.text}')
    elif et.tag == 'mo':
        print(f'operator: {et.text}')
    else:
        for elm in et:
            print(f'{idt}{elm} {elm.tag} {elm.attrib} text=\'{elm.text.strip()}\'')
            walk_mml_et_2(elm, indent=indent + 2)


def parse_mfrac(et):
    print('numerator')
    walk_mml_et(et[0])
    print('denominator')
    walk_mml_et(et[1])


def process_1(et):
    if et.tag == 'math':
        if et[0].tag == 'mrow':
            return [elm for elm in et[0]]
        else:
            raise(Exception(f'MathML root does not have mrow content: {et[0]}'))
    else:
        raise(Exception('Element is not MathML root'))


def show_xml(filepath):
    element_tree = read_xml_et(filepath)
    et_list = process_1(element_tree)
    print(et_list)
    for element in et_list:
        print(f'tag=\'{element.tag}\' text=\'{element.text.strip()}\'')
        walk_mml_et(element)


# -----------------------------------------------------------------------------
# Collect MML
# (this is used in pre-processing of xml to easier-to-manipulate form)
# -----------------------------------------------------------------------------


def collect_mml(elm):
    if elm.tag == 'mrow':
        return [':mrow'] + [collect_mml(e1) for e1 in elm]
    elif elm.tag == 'mfrac':
        return [':mfrac', collect_mml(elm[0]), collect_mml(elm[1])]
    elif elm.tag == 'mi':
        return [':mi', f'{elm.text}']
    elif elm.tag == 'mn':
        return [':mn', f'{elm.text}']
    elif elm.tag == 'mo':
        return [':mo', f'{elm.text}']
    elif elm.tag == 'mover':
        return [':mover', collect_mml(elm[0]), collect_mml(elm[1])]
    elif elm.tag == 'msub':
        return [':msub', collect_mml(elm[0]), collect_mml(elm[1])]
    else:
        return [':unhandled', f'tag:{elm.tag}', f'text:{elm.text}', f'attrib:{elm.attrib}']


def xml_to_mml(xml_str):
    et = cElementTree.XML(xml_str)
    if et.tag == 'math':
        math_contents = [elm for elm in et]
        if len(math_contents) == 1:
            if math_contents[0].tag == 'mrow':
                return collect_mml(math_contents[0])
            else:
                raise (Exception(f'MathML root is not tag \'mrow\': {math_contents[0]}'))
        else:
            raise (Exception(f'Math content is not a single value: {math_contents[0]}'))
    else:
        raise(Exception(f'Element is not MathML root: {et}'))


def xml_to_mml_from_file(filepath):
    with open(filepath, 'r') as file:
        xml_string = file.read()
        mml = xml_to_mml(xml_string)
        # pprint.pprint(mml)
        return mml


# -----------------------------------------------------------------------------
# MML to Eqn -- the main code for the utility
# -----------------------------------------------------------------------------

@dataclass(eq=False)
class Var:
    """
    Representation of a "named" variable
    Here, 'variable' is intended to mean a symbolic name for a value.
    Variables could be names of species (states) or rate (parameters).
    This class is hashable, and implements equality comparison.
    """
    var: str
    sub: Tuple[str, ...] = None

    def name_latex(self):
        if self.sub is not None:
            latex_str = f'{self.var}_{{{"".join(self.sub)}}}'
        else:
            latex_str = self.var
        return unicode_to_latex(latex_str)

    def name_mml(self):
        if self.sub is not None:
            subscript = list()
            for s in self.sub:
                if s.isnumeric():
                    subscript.append(f'<mn>{s}</mn>')
                else:
                    subscript.append(f'<mi>{s}</mi>')
            return f'<math><msub><mi>{self.var}</mi><mrow>{"".join(subscript)}</mrow></msub></math>'
        else:
            return f'<math><mi>{self.var}</mi></math>'

    def __hash__(self):
        return hash((self.var, self.sub))

    # TODO: Still needed now that hashable?
    def __eq__(self, other):
        if not isinstance(other, Var):
            return NotImplementedError
        if self.var != other.var:
            return False
        if self.sub is not None and other.sub is None:
            return False
        if self.sub is None and other.sub is not None:
            return False
        if self.sub is not None and other.sub is not None:
            if len(self.sub) != len(other.sub):
                return False
            for (e1, e2) in zip(self.sub, other.sub):
                if e1 != e2:
                    return False
        return True  # if you get this far, they appear equal


def unit_test_var_eq():
    """
    quick-n-dirty test of Var.__eq__
    :return:
    """
    var1 = Var('a')
    var2 = Var('a')

    var3 = Var('a', sub=tuple('2'))
    var4 = Var('a', sub=tuple('2'))

    var5 = Var('a', sub=tuple(['2', '3']))
    var6 = Var('a', sub=tuple(['2', '3']))

    var7 = Var('a', sub=tuple(['2', 'a']))
    var8 = Var('b', sub=tuple(['2', '3']))

    assert var1 == var2
    assert var3 == var4
    assert var5 == var6

    assert var1 != var3
    assert var3 != var5
    assert var5 != var7
    assert var5 != var8

    print('unit_test_var_eq() DONE')


@dataclass
class Tangent:
    """
    Represents the Tangent var of an ODE.
    TODO: This is perhaps not really needed, although it at least introduces a type.
    """
    var: Var


@dataclass
class Term:
    """
    A produce of rate and species that is added or subtracted.
    THere should just be one rate, but since we're parsing and there could
    be noise, this accommodates possibly reading several names of things
    that should be combined into a single rate.

    Possible sources of multiple rates (both of which are errors):
      (1) MathML interpreted multiple symbols as individual vars when they're really a single rate name
      (2) Error in identification of rates vs. species
      ...?
    """
    rate: Tuple[Var, ...]
    species: Tuple[Var, ...]
    polarity: str  # one of 'add', 'sub'

    def __eq__(self, other):
        return self.rate == other.rate and self.species == other.species

    def __hash__(self):
        return hash((self.rate, self.species))


def unit_test_term_eq():
    """
    quick-n-dirty test of Term.__eq__
    :return:
    """
    t1_v1 = Var(var='beta', sub=('1',))
    t1_v2 = Var(var='S')
    t1_v3 = Var(var='I')
    t1 = Term(rate=(t1_v1,), species=(t1_v2, t1_v3), polarity='sub')

    t2_v1 = Var(var='beta', sub=('1',))
    t2_v2 = Var(var='S')
    t2_v3 = Var(var='I')
    t2 = Term(rate=(t2_v1,), species=(t2_v2, t2_v3), polarity='add')

    t3_v1 = Var(var='beta', sub=('1',))
    t3_v2 = Var(var='S')
    t3_v3 = Var(var='I')
    t3 = Term(rate=(t3_v1,), species=(t3_v3, t3_v2), polarity='add')  # swapped order of species

    assert t1 == t2
    assert t1 != t3
    assert t2 != t3

    print('unit_test_term_eq() DONE')



@dataclass
class Eqn:
    """
    A single ODE equation.
    lhs: the left-hand-side is the tangent.
    rhs: the right-hand-side of the equation, a sequence of Terms.
    rhs_vars: collects all of the Vars that appear in the rhs.
    """
    lhs: Tangent
    rhs: List[Term]
    rhs_vars: Set[Var] = field(default_factory=set)


@dataclass
class EqnDict:
    """
    A collection of Eqns, indexed by the Var of the lhs Tangent.
    species: the set of all Vars across the eqns that are interpreted as species.
    rates: the set of all Vars across the eqns that are interpreted as species.
    """
    eqns: Dict[Var, Eqn]
    term_to_eqns: Dict[Term, Dict[str, List[Eqn]]]
    species: Set[Var] = field(default_factory=set)
    rates: Set[Var] = field(default_factory=set)
    term_to_edges: Dict[Term, Dict[str, List[Var]]] = field(default_factory=dict)  # dict: "in", "out"


def var_candidate_to_var(vc) -> Var:
    """
    Translate a variable candidate as a Var
    Different MathML can be interpreted as a name of Var:
      :mi : MathML identifier
      :msub : MathML tag attaching a subscript (as single value or mrow
              (sequence of values)) to an expression
    TODO: generalize the :msub to test that the vc[1] is an :mi ( or also :mn ? )
    :param vc:
    :return:
    """
    # print(vc)
    if vc[0] == ':mi':
        return Var(var=vc[1])
    elif vc[0] == ':msub':
        if vc[2][0] == ':mrow':   # Not yet tested!: list of subs
            # case: subscript as multiple chars
            # TODO: handle commas, which mml represents as mo
            sub_elms = vc[2][1:]
            subs = tuple([elm[1] for elm in sub_elms])
            return Var(var=vc[1][1], sub=subs)
        else:
            # case: subscript as single char
            return Var(var=vc[1][1], sub=tuple([vc[2][1]]))
    else:
        raise Exception(f'var_candidate_to_var(): Unexpected var candidate: {vc}')


def mfrac_leibniz_to_var(mfrac) -> Var:
    """
    Translate a MathML mfrac (fraction) as an expression of a Leibniz differential operator.
    In this case, the values after the 'd' or '∂' in the numerator are interpreted as
    the Var tangent.
    TODO: possibly generalize to accommodate superscripts for higher order derivatives;
        although likely that would be an nsup, so still the "first" elm of the numerator,
        with the second (and beyond) elm(s) being the Var.
    :param mfrac:
    :return:
    """
    numerator = mfrac[1]
    if len(numerator) > 2:
        var_candidate = numerator[2:]
        if len(var_candidate) == 1:
            return var_candidate_to_var(var_candidate[0])
        else:
            raise Exception(f'mfrac_to_var(): Unexpected mfrac var_candidate {var_candidate}')
    else:
        raise Exception(f'mfrac_to_var(): Unexpected mfrac form {mfrac}')


def is_leibniz_diff_op(mfrac):
    """
    Predicate to test whether and mfrac looks like a Liebniz differential operator
        d Var      ∂ Var
        -----  or  -----
        d t        ∂ t
    :param mfrac:
    :return:
    """
    numer = mfrac[1]
    denom = mfrac[2]
    # print(f'is_leibniz_diff_op(): {numer[1][1]} / {denom[1][1]}')
    numer_diff_op_p = numer[1][1] == 'd' or numer[1][1] == '∂'
    denom_diff_op_p = denom[1][1] == 'd' or denom[1][1] == '∂'
    return numer_diff_op_p and denom_diff_op_p


def mover_to_var(mover):
    """
    Translate MathML :mover as Newton dot notation: [:mover <Var> [:mo '˙']]
    :param mover:
    :return:
    """
    return var_candidate_to_var(mover[1])


def process_lhs(lhs) -> Tangent:
    """
    Interpret the lhs of the expression as a differential operator expression
    :param lhs:
    :return:
    """
    var = None
    if lhs[0] == ':mfrac':
        if is_leibniz_diff_op(lhs):
            var = mfrac_leibniz_to_var(lhs)
        else:
            raise Exception(f'process_lhs(): lhs is mfrac but does not look like Leibniz diff op: {lhs}')
    elif lhs[0] == ':mover':
        var = mover_to_var(lhs)
    return Tangent(var=var)


def is_sum_or_sub(elm):
    """
    Predicate testing whether a MathML operator (:mo) is a subtraction or addition.
    :param elm:
    :return:
    """
    return elm[0] == ':mo' and (elm[1] == '−' or elm[1] == '+')


def is_var_candidate(elm):
    """
    Predicate testing whether a MathML elm could be interpreted as a Var.
    TODO: This currently permits :mn -> MathML numerical literals.
        Perhaps useful to represent constant coefficients?
        But should those be Vars?
    :param elm:
    :return:
    """
    return elm[0] == ':mi' or elm[0] == ':mn' or elm[0] == ':msub'


def group_rhs(rhs):
    """
    Walk rhs sequence of MathML, identifying subsequences of elms that represent Terms
    :param rhs:
    :return:
    """
    # print(f'group_rhs(): {rhs}')

    # group terms
    terms = list()
    current_term = list()
    for elm in rhs:
        if is_sum_or_sub(elm):
            if len(current_term) > 0:
                terms.append(current_term)
                current_term = [elm]
            else:
                current_term.append(elm)
        elif is_var_candidate(elm):
            current_term.append(elm)
        else:
            raise Exception(f'group_rhs(): unhandled rhs elm: {elm}, rhs={rhs}')
    if len(current_term) > 0:
        terms.append(current_term)

    return terms


def rhs_groups_to_vars(rhs_groups):
    """
    Identify elms in grouped rhs elms that are Var candidates and translate them to Vars
    :param rhs_groups:
    :return:
    """
    var_set = set()
    new_groups = list()
    for group in rhs_groups:
        new_group = list()
        for elm in group:
            if is_var_candidate(elm):
                var = var_candidate_to_var(elm)
                new_group.append(var)
                var_set.add(var)
            else:
                new_group.append(elm)
        new_groups.append(new_group)
    return new_groups, var_set


def mml_to_eqn(mml: cElementTree.XML) -> Eqn:
    """
    Take an Element Tree that represent MathML and interpret it as a
    mass action kinetics ODE equation.
    :param mml:
    :return:
    """
    print(f'mml_to_eqn {mml}')
    try:
        idx = mml.index([':mo', '='])  # requires a try, raises ValueError is not found
        lhs = mml[1:idx][0]
        rhs = mml[idx+1:]

        # print('lhs:')
        # pprint.pprint(lhs)
        # print('rhs:')
        # print('before group_rhs')
        # pprint.pprint(rhs)
        # print()

        tangent = process_lhs(lhs)  # creates Tangent
        rhs_groups = group_rhs(rhs)  # groups the sequence of MathML elms into term-groups

        # print('before rhs_groups_to_vars')
        # pprint.pprint(rhs_groups)

        new_groups, var_set = rhs_groups_to_vars(rhs_groups)  # for each group, turns var-candidates into vars

        # print('after rhs_groups_to_vars')
        # pprint.pprint(new_groups)

        # NOTE: Here the rhs represents an intermediate step of having
        #       grouped the rhs into components of terms and translated
        #       var candidates into Vars, but not yet fully translated
        #       into a Tuple of Terms. This intermediate step must be
        #       finished before all equations have been collection, after
        #       which we can identify the tangent Vars that in turn
        #       are the basis for inferring which Vars are species vs rates
        #       Once species and rates are distinguished, then Terms can
        #       be formed.
        eqn = Eqn(lhs=tangent, rhs=new_groups, rhs_vars=var_set)
        # pprint.pprint(eqn)

        return eqn
    except ValueError:
        print(f'Does not look like Eqn: {mml}')
        raise ValueError


def mml_str_list_to_eqn_dict(mml_str_list) -> EqnDict:
    """
    Translate a list of MathML XML-containing strings into an EqnDict
      containing a collection of MAK ODE equations.
    :param mml_str_list:
    :return:
    """
    eqn_list = list()
    var_set = set()
    species = set()
    for mml_str in mml_str_list:
        mml = xml_to_mml(mml_str)
        eqn = mml_to_eqn(mml)
        var_set |= eqn.rhs_vars
        species.add(eqn.lhs.var)
        eqn_list.append(eqn)

    rates = var_set - species

    for eqn in eqn_list:
        terms = list()
        for rhs_group in eqn.rhs:
            term_rate = list()
            term_species = list()
            term_polarity = 'add'
            for elm in rhs_group:
                # print(f'elm: {elm}')
                if isinstance(elm, Var):
                    if elm in species:
                        term_species.append(elm)
                    else:
                        term_rate.append(elm)
                elif elm[0] == ':mo' and elm[1] == '−':
                    term_polarity = 'sub'
                    # print(f'Updating polarity!: {term_polarity}')
                elif elm[0] == ':mo' and elm[1] == '+':
                    term_polarity = 'add'  # redundant, but want to explicitly check this case
                else:
                    raise Exception(f'mml_str_list_to_eqn_set(): Unexpected rhs elm: {elm}')
            term = Term(rate=tuple(term_rate), species=tuple(term_species), polarity=term_polarity)
            terms.append(term)
        eqn.rhs = tuple(terms)

    eqn_dict = dict()
    term_to_eqn_dict = dict()
    for eqn in eqn_list:
        eqn_dict[eqn.lhs.var] = eqn

        # link Terms to eqns
        #   term_to_eqns: Dict[Term, Dict[str, Eqn]]
        for rhs_term in eqn.rhs:
            if rhs_term in term_to_eqn_dict:
                term_to_eqn_dict[rhs_term][rhs_term.polarity].append(eqn.lhs.var)
                # if rhs_term.polarity == 'sub':
                #     term_to_eqn_dict[rhs_term]['sub'].append(eqn.lhs.var)
                # elif rhs_term.polarity == 'add':
                #     term_to_eqn_dict[rhs_term]['add'].append(eqn.lhs.var)
            else:
                polarity_dict = {'sub': list(), 'add': list()}
                polarity_dict[rhs_term.polarity].append(eqn.lhs.var)
                term_to_eqn_dict[rhs_term] = polarity_dict

    eqn_dict = EqnDict(eqns=eqn_dict, term_to_eqns=term_to_eqn_dict, species=species, rates=rates)

    wire_pn(eqn_dict)

    return eqn_dict


def wire_pn(eqn_dict: EqnDict):
    """

    :param eqn_dict:
    :return:
    """
    for term, in_out_flow_dict in eqn_dict.term_to_eqns.items():
        in_list = list()
        out_list = list()
        for state in term.species:
            in_list.append(state)
            if state not in in_out_flow_dict['sub']:
                out_list.append(state)
        for state in in_out_flow_dict['add']:
            out_list.append(state)
        eqn_dict.term_to_edges[term] = {'in': in_list, 'out': out_list}


def read_mml_str_list(filepath) -> List[str]:
    """
    Read a list of MathML XML strings from file.
    File format: Each line is either
        a comment (starting with '#') -- comment lines are ignored
        a string MathML XML representation for an equation, on a single line
    :param filepath:
    :return:
    """
    mml_str_list = list()
    with open(filepath, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if not line.startswith('#'):
                mml_str_list.append(line)
    return mml_str_list


def export_eqn_dict_json(eqn_dict: EqnDict, filepath: str = None, verbose=False):

    species_sequence = list(eqn_dict.species)
    # rates_sequence = list(eqn_dict.rates)
    term_sequence = list(eqn_dict.term_to_edges.keys())

    species_index = dict()
    counter = 0
    for species in species_sequence:
        species_index[species] = counter
        counter += 1

    term_index = dict()
    counter = 0
    for term in term_sequence:
        term_index[term] = counter
        counter += 1

    json_dict = dict()
    json_dict['rates_mml'] = [term.rate[0].name_mml() for term in term_sequence]
    json_dict['rates_latex'] = [term.rate[0].name_latex() for term in term_sequence]
    json_dict['species_mml'] = [species.name_mml() for species in species_sequence]
    json_dict['species_latex'] = [species.name_latex() for species in species_sequence]

    edges = list()
    for term in term_sequence:
        edges.append(tuple([[species_index[species] for species in eqn_dict.term_to_edges[term]['in']],
                            [species_index[species] for species in eqn_dict.term_to_edges[term]['out']]]))

    json_dict['edges'] = edges

    if verbose:
        print(json.dumps(json_dict))

    if filepath is not None:
        with open(filepath, 'w') as file:
            json.dump(json_dict, file, indent=2)


def main(mml_str_list_filepath, json_export_filepath: str = None, verbose=False) -> EqnDict:

    mml_list = read_mml_str_list(mml_str_list_filepath)

    if verbose:
        print('>>> mml strings:')
        for mathml_str in mml_list:
            print(mathml_str)

    equation_dict = mml_str_list_to_eqn_dict(mml_list)

    if verbose:
        print('\n==========\n')
        pprint.pprint(equation_dict)
        print('\n==========\n')

    export_eqn_dict_json(equation_dict, json_export_filepath, verbose=verbose)

    return equation_dict


# -----------------------------------------------------------------------------
# SCRIPT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Use the above V1..V4 paths to run existing mml_list.txt files
    main(mml_str_list_filepath=V3, json_export_filepath='model_pn.json', verbose=True)

    # unit_test_var_eq()
    unit_test_term_eq()

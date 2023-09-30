# -*- coding: utf-8 -*-

import re
import json
import sys


def get_config(config_path):
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    return config


def simplification(mml_org):
    """
    simplify the mathml by removing unnecessary information.
    """

    # Removing multiple backslashes
    i = mml_org.find("\\\\")
    mml_org = mml_org.encode().decode("unicode_escape")

    while i > 0:
        mml_org = mml_org.replace("\\\\", "\\")
        i = mml_org.find("\\\\")

    # Removing initial information about URL, display, and equation itself
    # keeping display="block"
    begin = mml_org.find("<math") + len("<math")
    end = mml_org.find(">")
    mml_org = mml_org.replace(mml_org[begin:end], ' display="block"')

    # ATTRIBUTES

    ## Attributes commonly used in MathML codes to represent equations
    elements = [
        "mrow",
        "mi",
        "mn",
        "mo",
        "ms",
        "mtext",
        "math",
        "mtable",
        "mspace",
        "maction",
        "menclose",
        "merror",
        "mfenced",
        "mfrac",
        "mglyph",
        "mlabeledtr",
        "mmultiscripts",
        "mover",
        "mroot",
        "mpadded",
        "mphantom",
        "msqrt",
        "mstyle",
        "msub",
        "msubsup",
        "msup",
        "mtd",
        "mtr",
        "munder",
        "munderover",
        "semantics",
    ]

    ## Attributes that can be removed
    attr_tobe_removed = [
        "class",
        "id",
        "style",
        "href",
        "mathbackground",
        "mathcolor",
    ]

    ## Attributes that need to be checked before removing,
    #  if mentioned in code with their default value,
    ## will be removed else will keep it. This dictionary 
    # contains all the attributes with thier default values.
    attr_tobe_checked = {
        "displaystyle": "false",
        "mathsize": "normal",
        "mathvariant": "normal",
        "fence": "false",
        "accent": "false",
        "movablelimits": "false",
        "largeop": "false",
        "stretchy": "false",
        "lquote": "&quot;",
        "rquote": "&quot;",
        "overflow": "linebreak",
        # "display": "block",
        "denomalign": "center",
        "numalign": "center",
        "align": "axis",
        "rowalign": "baseline",
        "columnalign": "center",
        "alignmentscope": "true",
        "equalrows": "true",
        "equalcolumns": "true",
        "groupalign": "{left}",
        "linebreak": "auto",
        "accentunder": "false",
    }

    mml_mod = attribute_definition(
        mml_org, elements, attr_tobe_removed, attr_tobe_checked
    )

    mml_mod = cleaning_mml(mml_mod)
    mml_mod = tokenize(mml_mod)

    return mml_mod


def attribute_definition(
    mml_code, elements, attr_tobe_removed, attr_tobe_checked
):
    """
    Removing unnecessary information or attributes having default values.
    """

    # Defining array to keep Attribute definition
    definition_array = []

    for ele in elements:
        # Getting indices of the position of the element in the MML code
        position = [
            i for i in re.finditer(r"\b%s\b" % re.escape(ele), mml_code)
        ]

        for p in position:
            # Attribute begining and ending indices
            (attr_begin, attr_end) = p.span()

            # length of the definition of the attribute
            length = mml_code[attr_end:].find(">")

            if length > 0:
                # Grabbing definition
                definition = mml_code[attr_end : attr_end + length].split()

                # Append unique definition
                for deftn in definition:
                    if deftn not in definition_array:
                        definition_array.append(deftn)

    # remove all the attributes that need to be removed
    for darr in definition_array:
        if "=" in darr:
            # Attribute and its value -- of the element
            attribute_parameter = darr.replace(" ", "").split("=")[0]
            attribute_value = darr.replace(" ", "").split("=")[1]

            # If Attribute has a defualt value, we can remove it
            # Checking which attributes can be removed
            if attribute_parameter not in attr_tobe_removed:
                if attribute_parameter in attr_tobe_checked.keys():
                    if (
                        attribute_value.replace("\\", "").replace('"', "")
                        == attr_tobe_checked[attribute_parameter]
                    ):
                        mml_code = mml_code.replace(" " + darr, "")
            else:
                mml_code = mml_code.replace(" " + darr, "")

    return mml_code


def count(eqn, e):
    """
    counts number of times a token appears in an eqn.
    """
    c = 0
    for word in eqn.split():
        if e in word:
            c += 1
    return c


def isfloat(num):
    """
    checking if the token is float.
    """
    try:
        float(num)
        return True
    except:
        return False


def isint(num):
    """
    checking if the token is int.
    """
    try:
        int(num)
        return True
    except:
        return False


def isfrac(num):
    """
    checking if the token is fraction.
    """
    return re.match("[|-|+]?\d+\/\d+$", num)


def remove_unecc_tokens(eqn):
    """
    This function further cleans the MathML equation.
    One can add or remove the token that needs to be removed.
    """
    eliminate = [
        "mspace",
        # "mtable",
        "mathvariant",
        "class",
        "mpadded",
        "symmetric",
        "fence",
        "rspace",
        "lspace",
        "displaystyle",
        "scriptlevel",
        "stretchy",
        "form",
        "movablelimits",
        "maxsize",
        "minsize",
        "linethickness",
        "mstyle",
    ]

    keep = ["mo", "mi", "mfrac", "mn", "mrow", "mtr", "mtd"]

    for e in eliminate:
        if e in eqn:
            c = count(eqn, e)
            for _ in range(c):
                idx = eqn.find(e)

                # find the '<' just before the e
                temp1 = eqn[: idx + 1]
                temp2 = eqn[idx + 1 :]
                open_angle = [
                    idx_open
                    for idx_open, angle in enumerate(temp1)
                    if angle == "<"
                ]
                close_angle = [
                    idx_close
                    for idx_close, angle in enumerate(temp2)
                    if angle == ">"
                ]
                filtered = (
                    temp1[open_angle[-1] :] + temp2[: close_angle[0] + 1]
                )
                flag = False
                for k in keep:
                    if ("<" + k) in filtered:
                        flag = True
                        if e in ["movablelimits", "minsize"] and k in [
                            "mo",
                            "mi",
                        ]:
                            true_k = [
                                k
                                for f in filtered.split()
                                if k in f and e not in f
                            ]
                            if len(true_k) > 0:
                                keep_token = true_k[0]
                        else:
                            keep_token = k
                if flag is True:
                    eqn = (
                        temp1[: open_angle[-1]]
                        + f" <{keep_token}>"
                        + temp2[close_angle[0] + 1 :]
                    )
                else:
                    eqn = temp1[: open_angle[-1]] + temp2[close_angle[0] + 1 :]

    return eqn


def remove_single_mrow_pairs(lst):
    stack = []
    for i, elem in enumerate(lst):
        if "<mrow" in elem:
            stack.append(i)
        elif elem == "</mrow>":
            start = stack.pop()
            if i - start <= 2:
                lst.pop(i)
                lst.pop(start)
                return remove_single_mrow_pairs(lst)
    return lst


def remove_additional_tokens(eqn):
    """
    remove redundant tokens like mtext and mrow.
    <mrow> will not consoidered if it has only one
    row. In case of more than one row, <mrow> will be
    considered and not be removed.
    """
    if "mtext" in eqn:
        try:
            c = count(eqn, "mtext")
            for _ in range(c):
                e1, e2 = eqn.find("<mtext>"), eqn.find("</mtext>")
                eqn = eqn[:e1] + eqn[e2 + len("</mtext>") :]
        except:
            pass

    if "mrow" in eqn:
        try:
            eqn = eqn.replace("><", "> <")
            eqn_arr = eqn.split()
            eqn_arr = remove_single_mrow_pairs(eqn_arr)
            f = ""
            for F in eqn_arr:
                f = f + F + " "
            return f
        except:
            f = ""
            for F in eqn.split():
                f = f + F + " "
            return f

    else:
        f = ""
        for F in eqn.split():
            f = f + F + " "

        return f


def remove_hexComments(eqn):
    """
    removing comments associated with the unicodes i.e. /$#x.../<!-- symbol -->
    <!-- ... --> any thing within these < > are comments representing
    the symbol associated with the unicode.
    """
    temp_arr = []
    eqn_split = eqn.split()

    skip_idx = None
    for _idx, _o in enumerate(eqn_split):
        if _idx != skip_idx:
            if "&#x" in _o:
                temp_arr.append(_o.split(";")[0].strip())
                if _idx + 1 != len(eqn_split) - 1:
                    skip_idx = _idx + 1

            elif "-->" in _o:
                temp_arr.append(_o.split("-->")[-1].strip())

            else:
                temp_arr.append(_o)

    final = " ".join(temp_arr)

    return final


def cleaning_mml(eqn):
    """
    clean the equation.
    """
    eqn = remove_unecc_tokens(eqn)
    eqn = remove_additional_tokens(eqn)
    if "&#x" in eqn:
        eqn = remove_hexComments(eqn)
    return eqn


def remove_attributes(mathml_str):
    """
    Remove all attributes from MathML string tokens.

    Args:
        mathml_str (str): A string containing MathML tokens with attributes.

    Returns:
        str: The MathML string with all attributes removed.
    """
    # Define a regular expression pattern that matches attributes
    attribute_pattern = r'\s+\w+="[^"]*"'

    # Use regular expressions to replace all attribute matches with an empty string
    result = re.sub(attribute_pattern, "", mathml_str)

    return result


def extract_inbetween_tokens(text):
    clean_mml_eqn = remove_attributes(text)
    # Use regular expression to extract all tokens from the MathML string
    tokens = re.findall(r"<[^>]+>|[^<]+", clean_mml_eqn)
    # Use a list to save the contents of all token pairs
    contents = []
    for token in tokens:
        # If the token is a MathML tag, skip it
        if token.startswith("<"):
            continue
        # Add the content of the token to the contents list
        if len(token) > 0 and not token.isspace():
            contents.append(token)
    return contents


def tokenize(mml_eqn):
    """
    tokenize the final cleaned equation based on < >
    i.e. <math><mo> ... </mo></math> will be tokenized like
    <math> <mo> ... </mo> </math>.
    """
    mml_split = re.split(">|<", mml_eqn)
    tokenized_mml = ""

    inbetween_tokens = extract_inbetween_tokens(mml_eqn)

    for token in mml_split:
        token = token.strip()

        if len(token) > 0:
            if "&#x" in token or len(token) == 1:
                tokenized_mml += token

            elif (
                token.isdigit()
            ):  # entire number is made up integers e.g. 12345
                for intgr in list(map(int, token)):
                    tokenized_mml += f" {intgr} "

            elif isfloat(token):  # eg. 120.456
                try:
                    token_arr = token.split(".")
                    for tok_idx, tok in enumerate(token_arr):
                        if tok_idx == 1:
                            tokenized_mml += "."

                        for intgr in list(map(int, token_arr[tok_idx])):
                            tokenized_mml += f" {intgr} "
                except:
                    pass

            elif isfrac(token):
                token_arr = token.split("/")

                for tok_idx, tok in enumerate(token_arr):
                    if tok_idx == 1:
                        tokenized_mml += "/"
                    for intgr in list(map(int, token_arr[tok_idx])):
                        tokenized_mml += f" {intgr} "

            elif token in inbetween_tokens:
                if len(token.replace(" ", "")) < len(token):
                    tokenized_mml += token.replace(" ", "")
                else:
                    tokenized_mml += token

            else:
                tokenized_mml += " <" + token + "> "

    return tokenized_mml.strip()


if __name__ == "__main__":
    # get config
    config = get_config(sys.argv[-1])

    # get the rejected images
    data_path = f"{config['data_path']}/{config['dataset_type']}"
    org_mml = open(
        f"{data_path}/original_{config['markup']}.lst", "r"
    ).readlines()
    modified_mml_file = open(f"{data_path}/{config['markup']}.lst", "w")

    blank_images = open("logs/blank_images.lst").readlines()
    idx_to_be_ignored = [int(i.split(".")[0]) for i in blank_images]

    for eqn_idx in range(len(org_mml)):
        if eqn_idx not in idx_to_be_ignored:
            eqn = org_mml[eqn_idx]
            if len(eqn) > 2:
                mml = simplification(eqn)
                # writing
                if "\n" not in mml:
                    modified_mml_file.write(mml + "\n")
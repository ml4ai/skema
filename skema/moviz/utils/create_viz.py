import json
import graphviz
from skema.moviz.utils.helper_functions import drawB, setExpandValue

#remove this before commit
# from skema.utils.fold import del_nulls, dictionary_to_gromet_json

from utils.helper_functions import (
    drawBC,
    drawBL,
    drawWFC,
    drawWFF,
    drawBF,
    drawWFL,
    drawWFOPO,
    drawWFOPI,
    drawWOPIO,
)


from utils.init import init

from pathlib import Path


def draw_graph(gromet, program_name: str):
    
    cwd = Path(__file__).parents[0]
    data = gromet.to_dict()
    init(data)

    ###remove this before commit
    # with open(f"{program_name}--Gromet-FN-auto.json", "w") as f:
    #     f.write(dictionary_to_gromet_json(del_nulls(data)))
    ###

    cwd = Path(__file__).parents[0]
    filepath = cwd / "../../../data/gromet/config/config.json"
    with open(filepath, 'r') as cf:
        config = json.load(cf)

    g = graphviz.Graph(
        "G",
        filename=program_name,
        engine="dot",
        format="png",
        directory="static",
    )
    g.attr(compound='true')
    g.attr(rankdir="BT")

    # print(type(g))
    if config['start_collapsed'] == 'false':
        lhs= drawLHS(data, g)
        rhs = drawRHS(data, g)
        output = drawArrows(data, g, False)
        return output
    elif config['start_collapsed'] == 'true':
        if not config['uncollapse_list']:
            lhs = drawLHS(data, g)
            return lhs
        else:
            lhs = drawLHS(data, g)
            for item in config['uncollapse_list']:
                setExpandValue(data, item)
                rhs = drawRHS(data, g)
                # print(rhs)
            # print(rhs)
            output = drawArrows(data, rhs, True)
            return output

#LHS
def drawLHS(data, g):
    i=0
    # print(data.get('fn'))
    for b in data.get("fn").get("b"):
        # print(b)
        if b["function_type"] == "MODULE":
            with g.subgraph(name="clusterA") as a:
                a.attr(color='gray', style='rounded', penwidth='3', label=f"id: {b.get('box')}")
                a.attr("node",shape = 'point')
                a.node(name=f"clusterA_{i}", style = 'invis')
                i+=1
                if data.get("fn").get("bc") != None:
                    drawBC(data.get("fn"), a)
                if data.get("fn").get("bl") != None:
                    # print("here")
                    drawBL(data.get("fn"), a)
                else:
                    if data.get("fn").get("bf") != None:
                        # print(data.get("fn").get("bf"))
                        for bf in data.get("fn").get("bf"):
                            drawBF(data.get("fn"), a, bf)
                            # print("bf: ", bf)
    # print(data.get('fn').get('pof'))
    drawWFC(data["fn"], g)
    drawWFL(data["fn"], g)
    drawWFF(data["fn"], g)
    return g

def drawRHS(data, g):
    print("RHS")
    i=0
    for attribute in data.get("fn_array"):
        # if attribute.get("type") == "FN":
        if attribute.get("b") != None:
            for b in attribute.get("b"):    
                if attribute.get('expand') == True:
                    if b.get("function_type") == "EXPRESSION":
                        i = drawB(g, attribute, b, "expr", "purple", i)
                    if b.get("function_type") == "FUNCTION":
                        i = drawB(g, b, attribute, "func", "green", i)
                    if b.get("function_type") == "PREDICATE":
                        i = drawB(g, b, attribute, "pred", "pink", i)
                    drawWFF(attribute, g)
                    drawWFOPO(attribute, g)
                    drawWFOPI(attribute, g)
                    drawWOPIO(attribute, g)
                elif attribute.get('expand') == None:
                    if b.get("function_type") == "EXPRESSION":
                        i = drawB(g, attribute, b, "expr", "purple", i)
                    if b.get("function_type") == "FUNCTION":
                        i = drawB(g, b, attribute, "func", "green", i)
                    if b.get("function_type") == "PREDICATE":
                        i = drawB(g, b, attribute, "pred", "pink", i)
                    drawWFF(attribute, g)
                    drawWFOPO(attribute, g)
                    drawWFOPI(attribute, g)
                    drawWOPIO(attribute, g)
        # if attribute.get("type") == "IMPORT":
            # with g.subgraph(name=f"cluster_import_{attribute.index()}") as b:
            #     b.attr(label=str(attribute))
    return g

# connecting LHS and RHS
def drawArrows(data, g, expand):
    # g.attr(rankdir="LR")
    
    if data.get("fn").get("bf") != None:
        for bf in data.get("fn").get("bf"):
            # for attribute in data.get('attributes'):
            if bf.get("body") != None:
                attribute = data.get("fn_array")[bf.get("body") - 1]
                # print(attribute)
                if attribute.get("b") != None:
                    b = attribute.get("b")
                    for b in attribute.get("b"):
                        # for b in attribute.get("b"):
                        print(b, expand, attribute.get("expand"))
                        if expand == True and attribute.get("expand") == True:
                            print("hi",bf.get("node"), b.get("node"))
                            print(bf.get("invisNode"), b.get("invisNode"))
                            g.edge(bf.get("invisNode"), b.get("invisNode"), ltail=bf.get('node'), lhead=b.get('node'), dir='forward', arrowhead='normal', color="brown", style="dashed", minlen="3")
                        elif expand == False:
                            print("in draw arrows")
                            print(bf.get("invisNode"), b.get("invisNode"))
                            g.edge(bf.get("invisNode"), b.get("invisNode"), ltail=bf.get('node'), lhead=b.get('node'), dir='forward', arrowhead='normal', color="brown", style="dashed", minlen="3")

    # edges between the different attributes in RHS 
    for attribute in data.get("fn_array"):
        if attribute.get("bf") != None:
            for bf in attribute.get("bf"):
                if bf.get("contents") != None:
                    attr = data.get("attributes")[bf.get("contents") - 1]
                    if attr.get("b") != None:
                        for b in attr.get("b"):
                            g.edge(bf.get("invisNode"), b.get("invisNode"), ltail=bf.get('node'), lhead=b.get('node'), dir='forward', arrowhead='normal', color="brown", style="dashed", minlen="5")

    return g
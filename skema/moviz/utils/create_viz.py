import json
import graphviz
from utils.helper_functions import (
    drawBC,
    drawBL,
    drawOPO,
    drawOPI,
    drawWFC,
    drawWFF,
    drawBF,
    drawWFL,
    drawWFOPO,
    drawWFOPI,
    drawWOPIO,
)

from utils.init import init

def draw_graph(gromet, program_name: str):
    data = gromet.to_dict()
    init(data)

    g = graphviz.Graph(
        "G",
        filename=program_name,
        engine="fdp",
        format="png",
        directory="static",
    )

    # LHS
    for b in data.get("fn").get("b"):
        if b["function_type"] == "MODULE":
            with g.subgraph(name="clusterA") as a:
                if data.get("fn").get("bc") != None:
                    drawBC(data.get("fn"), a)
                if data.get("fn").get("bl") != None:
                    print("here")
                    drawBL(data.get("fn"), a)
                else:
                    if data.get("fn").get("bf") != None:
                        for bf in data.get("fn").get("bf"):
                            drawBF(data.get("fn"), a, bf)
    drawWFC(data["fn"], g)
    drawWFL(data["fn"], g)
    drawWFF(data["fn"], g)

    # RHS
    for attribute in data.get("attributes"):
        if attribute.get("type") == "FN":
            if attribute.get("value").get("b") != None:
                for b in attribute.get("value").get("b"):
                    if b.get("function_type") == "EXPRESSION":
                        with g.subgraph(
                            name=f"cluster_expr_{b.get('box')}"
                        ) as a:
                            b["node"] = f"cluster_expr_{b.get('box')}"
                            drawOPO(b, a, attribute.get("value"))
                            drawOPI(b, a, attribute.get("value"))
                            if attribute.get("value").get("bf") != None:
                                # print("attribute")
                                for bf in attribute.get("value").get("bf"):
                                    drawBF(attribute.get("value"), a, bf)
                    if b.get("function_type") == "FUNCTION":
                        with g.subgraph(
                            name=f"cluster_func_{b.get('box')}"
                        ) as a:
                            if b.get("name") != None:
                                a.attr(label=str(b.get("name")))
                            b["node"] = f"cluster_func_{b.get('box')}"
                            drawOPO(b, a, attribute.get("value"))
                            drawOPI(b, a, attribute.get("value"))
                            if attribute.get("value").get("bf") != None:
                                # print("attribute")
                                for bf in attribute.get("value").get("bf"):
                                    drawBF(attribute.get("value"), a, bf)
                    if b.get("function_type") == "PREDICATE":
                        with g.subgraph(
                            name=f"cluster_pred_{b.get('box')}"
                        ) as a:
                            a.attr(label=str(b.get("name")))
                            b["node"] = f"cluster_pred_{b.get('box')}"
                            drawOPO(b, a, attribute.get("value"))
                            drawOPI(b, a, attribute.get("value"))
                            if attribute.get("value").get("bf") != None:
                                for bf in attribute.get("value").get("bf"):
                                    drawBF(attribute.get("value"), a, bf)
                drawWFF(attribute.get("value"), g)
                drawWFOPO(attribute.get("value"), g)
                drawWFOPI(attribute.get("value"), g)
                drawWOPIO(attribute.get("value"), g)
        if attribute.get("value").get("type") == "IMPORT":
            with g.subgraph(name=f"cluster_import_{attribute.index()}") as b:
                b.attr(label=str(attribute))

    print("hi")
    print(data.get("fn").get("pof"))
    # connecting LHS and RHS
    for bf in data.get("fn").get("bf"):
        # for attribute in data.get('attributes'):
        if bf.get("contents") != None:
            attribute = data.get("attributes")[bf.get("contents") - 1]
            if attribute.get("value").get("b") != None:
                for b in attribute.get("value").get("b"):
                    for b in attribute.get("value").get("b"):
                        print(bf.get("node"), b.get("node"))
                        g.edge(bf.get("node"), b.get("node"))

    # edges between the different attributes in RHS
    for attribute in data.get("attributes"):
        if attribute.get("value").get("bf") != None:
            for bf in attribute.get("value").get("bf"):
                if bf.get("contents") != None:
                    attr = data.get("attributes")[bf.get("contents") - 1]
                    if attr.get("value").get("b") != None:
                        for b in attr.get("value").get("b"):
                            g.edge(bf["node"], b.get("node"))
    return g

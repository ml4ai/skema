def drawPOF(bf, c, data, label):
    if data.get("pof") != None:
        for pof in data["pof"]:
            index = bf["box"].split("-")
            if str(pof["box"]) == str(index[-1]):
                c.attr("node", shape="box")
                if pof.get("name") != None:
                    pof["node"] = f"pof-{bf['box']}"
                    c.node(name=f"pof-{bf['box']}", label=str(pof.get("name")), width='0.5', penwidth='2')
                    # c.attr(label = str(pof.get('name')))
                elif label != None:
                    pof["node"] = f"pof-{bf['box']}"
                    c.node(name=f"pof-{bf['box']}", label=label, width='0.5', penwidth='2')
                else:
                    pof["node"] = f"pof-{bf['box']}"
                    c.node(name=f"pof-{bf['box']}", fontcolor="white", label="", width='0.5', penwidth='2')


def drawPIF(bf, c, data):
    if data.get("pif") != None:
        i = 1
        index = bf.get("box").split("-")
        for pif in data["pif"]:
            if str(pif["box"]) == str(index[-1]):
                c.attr("node", shape="box")
                pif["node"] = f"pif-{bf.get('box')}-{i}"
                c.node(name=f"pif-{bf.get('box')}-{i}", fontcolor="white", label="", width='0.5', penwidth='2')
                i += 1


def drawOPO(b, c, data):
    if data.get("opo") != None:
        for opo in data["opo"]:
            index = b.get("box").split("-")
            if str(opo.get("box")) == index[-1]:
                opo["node"] = f"opo-{b.get('box')}"
                c.attr("node", shape="box")
                c.node(name=f"opo-{b.get('box')}", fontcolor="white", label="", width='0.5', penwidth='2')


def drawOPI(b, c, data):
    if data.get("opi") != None:
        for opi in data["opi"]:
            index = b.get("box").split("-")
            if str(opi.get("box")) == index[-1]:
                if opi.get("name") != None:
                    opi["node"] = f"opi-{b.get('box')}-{str(opi.get('name'))}"
                    c.attr("node", shape="box")
                    c.node(
                        name=f"opi-{b.get('box')}-{str(opi.get('name'))}",
                        label=str(opi.get("name")),
                        width='0.5', 
                        penwidth='2'
                    )
                else:
                    opi["node"] = f"opi-{b.get('box')}"
                    c.attr("node", shape="box")
                    c.node(name=f"opi-{b.get('box')}", fontcolor="white", label="", width='0.5', penwidth='2')


def drawPIC(data, bc, c):
    if data.get("pic") != None:
        for pic in data.get("pic"):
            index = bc["box"].split("-")
            # print(index)
            c.attr("node", shape="box")
            if str(pic["box"]) == str(index[-1]):
                if pic.get("name") != None:
                    pic["node"] = f"pic-{bc['box']}"
                    c.node(name=str(pic.get("name")), width='0.5')
                else:
                    pic["node"] = f"pic-{bc['box']}"
                    c.node(name=f"pic-{bc['box']}", fontcolor="white", width='0.5')


def drawPOC(data, bc, c):
    if data.get("poc") != None:
        for poc in data.get("poc"):
            index = bc["box"].split("-")
            c.attr("node", shape="box")
            if str(poc["box"]) == str(index[-1]):
                if poc.get("name") != None:
                    c.node(name=str(poc.get("name")), width='0.5')
                else:
                    c.node(name=f"poc-{bc['box']}", fontcolor="white", width='0.5')


def drawPIL(data, bl, c):
    if data.get("pil") != None:
        for pil in data.get("pil"):
            index = bl["box"].split("-")
            c.attr("node", shape="box")
            if str(pil["box"]) == str(index[-1]):
                if pil.get("name") != None:
                    pil["node"] = f"pil-{bl['box']}"
                    c.node(name=f"pil-{bl['box']}", label=str(pil.get("name")),  width='0.5')
                else:
                    pil["node"] = f"pil-{bl['box']}"
                    c.node(name=f"pil-{bl['box']}", fontcolor="white", width='0.5')


def drawPOL(data, bl, c):
    if data.get("pol") != None:
        for pol in data.get("pol"):
            index = bl["box"].split("-")
            c.attr("node", shape="box")
            if str(pol["box"]) == str(index[-1]):
                if pol.get("name") != None:
                    pol["node"] = f"pol-{bl['box']}"
                    c.node(name=f"pol-{bl['box']}", label=str(pol.get("name")), width='0.5')
                else:
                    pol["node"] = f"pol-{bl['box']}"
                    c.node(name=f"pol-{bl['box']}", fontcolor="white", width='0.5')


def drawWFF(data, g):
    if data.get("wff") != None:
        for wff in data["wff"]:
            print(wff)
            if data.get("pif") != None:
                for pif in data["pif"]:
                    if data.get("pof") != None:
                        for pof in data["pof"]:
                            # print(wff["src"], pif["id"], wff["tgt"], pof["id"])
                            if wff["src"] == pif["id"] and wff["tgt"] == pof["id"]:
                                # print(pif.get("node"), pof.get("node"), "\n",pif, '\n', pof)
                                g.edge(pif.get("node"), pof.get("node"), dir='forward', arrowhead='normal', color="brown")


def drawWFC(data, g):
    if data.get("wfc") != None:
        for wfc in data.get("wfc"):
            for pof in data.get("pof"):
                for pic in data.get("pic"):
                    if wfc["src"] == pic["id"] and wfc["tgt"] == pof["id"]:
                        # print("here: ",pic.get('node'), pof.get('node'))
                        g.edge(pic.get("node"), pof.get("node"), dir='forward', arrowhead='normal', color="brown")


def drawWFL(data, g):
    if data.get("wfl") != None:
        for wfc in data.get("wfl"):
            for pof in data.get("pof"):
                for pil in data.get("pil"):
                    if wfc["src"] == pil["id"] and wfc["tgt"] == pof["id"]:
                        # print("here: ",pil.get('node'), pof.get('node'))
                        g.edge(pil.get("node"), pof.get("node"), dir='forward', arrowhead='normal', color="brown")


def drawWFOPO(data, g):
    if data.get("wfopo") != None:
        for wfopo in data.get("wfopo"):
            if data.get("opo") != None:
                for opo in data.get("opo"):
                    for pof in data.get("pof"):
                        if (
                            wfopo["src"] == opo["id"]
                            and wfopo["tgt"] == pof["id"]
                        ):
                            # print(opo.get('node'), pof.get('node'))
                            g.edge(opo.get("node"), pof.get("node"), dir='forward', arrowhead='normal', color="brown")


def drawWFOPI(data, g):
    if data.get("wfopi") != None:
        for wfopi in data.get("wfopi"):
            if data.get("opi") != None:
                for opi in data.get("opi"):
                    for pif in data.get("pif"):
                        if (
                            wfopi["src"] == opi["id"]
                            and wfopi["tgt"] == pif["id"]
                        ):
                            g.edge( pif["node"], opi["node"], dir='forward', arrowhead='normal', color="brown")


def drawWOPIO(data, g):
    if data.get("wopio") != None:
        for wopio in data.get("wopio"):
            if data.get("opo") != None:
                for opo in data.get("opo"):
                    if data.get("opi") != None:
                        for opi in data.get("opi"):
                            if (
                                wopio["src"] == opo["id"]
                                and wopio["tgt"] == opi["id"]
                            ):
                                g.edge(opo["node"], opi["node"],dir='forward', arrowhead='normal', color="brown")


# def drawWFCARGS(data, g):

#     for bf in data.get('fn').get(bf):

#     if data.get('wl_cargs') != None:
#         for wl_cargs in data.get('wl_cargs'):
#             for poc in data.get('poc'):

#                 if wl_cargs['src'] == ['id'] and wl_cargs['tgt'] == poc['id']:
#                     print("here: ",poc.get('node'), poc.get('node'))
#                     g.edge(pic.get('node'), pof.get('node'))


def drawBF(data, a, bf):
    i=0
    # print(bf)
    if bf.get('node') == None:
        if bf.get("function_type") == "EXPRESSION":
            with a.subgraph(name=f"cluster_expr_{bf['box']}") as b:
                b.attr(color='purple', style='rounded', penwidth='3', label=f"id: {bf.get('box')}")
                b.attr("node", shape = 'point')
                b.node(name=f"cluster_expr_{bf['box']}_{i}", style = 'invis')
                
                bf['invisNode'] = f"cluster_expr_{bf['box']}_{i}"
                i+=1
                bf["node"] = f"cluster_expr_{bf['box']}"
                # b.attr("node", shape="box")
                drawPIF(bf, b, data)
                drawPOF(bf, b, data, None)
        if bf.get("function_type") == "LITERAL":
            if bf.get("value").get("value_type") == "Integer" or bf.get("value").get("value_type") == "Boolean":
                literal = str(bf.get("value").get("value"))
                with a.subgraph(name=f"cluster_lit_{literal}_{bf['box']}") as c:
                    print(bf.get('box'))
                    label = f"{literal}"+"\n id: "+bf.get('box')
                    c.attr(color='red', shape='box', style='rounded', penwidth='3', label=label)
                    c.attr("node", shape = 'point')
                    c.node(name=f"cluster_lit_{literal}_{bf['box']}_{i}", style = 'invis')
                    bf['invisNode'] = f"cluster_lit_{literal}_{bf['box']}_{i}"
                    i+=1
                    bf["node"] = f"cluster_lit_{literal}_{bf['box']}"
                    drawPIF(bf, c, data)
                    drawPOF(bf, c, data, None)                 
            if bf.get('value').get('value_type') == 'List':
                literal = ""
                for value in bf.get('value').get('value'):
                    print(value)
                    literal = literal+", "+str(value)
                    print('vals: ',literal)
                with a.subgraph(name=f"cluster_lit_{literal}_{bf['box']}") as c:
                    print(bf.get('box'))
                    label = f"{literal}"+"\n id: "+bf.get('box')
                    c.attr(color='red', shape='box', style='rounded', penwidth='3', label=label)
                    c.attr("node", shape = 'point')
                    c.node(name=f"cluster_lit_{literal}_{bf['box']}_{i}", style = 'invis')
                    bf['invisNode'] = f"cluster_lit_{literal}_{bf['box']}_{i}"
                    i+=1
                    bf["node"] = f"cluster_lit_{literal}_{bf['box']}"
                    drawPIF(bf, c, data)
                    drawPOF(bf, c, data, None) 
        if bf["function_type"] == "PRIMITIVE":
            primitive = str(bf["name"])
            with a.subgraph(name=f"cluster_prim_{primitive}_{bf['box']}") as d:
                d.attr("node",shape = 'point')
                d.node(name=f"cluster_prim_{primitive}_{bf['box']}_{i}", style = 'invis')
                bf['invisNode'] = f"cluster_prim_{primitive}_{bf['box']}_{i}"
                i+=1
                if primitive != None:
                    bf["node"] = f"cluster_prim_{primitive}_{bf['box']}"
                label = primitive+"\n id: "+bf.get('box')
                d.attr(label=label)
                d.attr(color='black', shape='box', penwidth='3')
                d.attr("node", shape="box")
                drawPIF(bf, d, data)
                drawPOF(bf, d, data, None)
        if bf.get("function_type") == "FUNCTION":
            with a.subgraph(name=f"cluster_func_{bf['box']}") as e:  # function
                e.attr(color='green', style='rounded', penwidth='3', label=f"bf-{bf.get('box')[-1]}")
                e.attr("node",shape = 'point')
                e.node(name=f"cluster_func_{bf['box']}_{i}", style = 'invis')
                bf['invisNode'] = f"cluster_func_{bf['box']}_{i}"
                i+=1
                bf["node"] = f"cluster_func_{bf['box']}"
                drawPOF(bf, e, data, None)
                drawPIF(bf, e, data)
        if bf.get("function_type") == "PREDICATE":
            with a.subgraph(name=f"cluster_pred_{bf['box']}") as f:
                f.attr(color='pink', style='rounded', penwidth='3', label=f"bf-{bf.get('box')[-1]}")
                f.attr("node", shape = 'point')
                f.node(name=f"cluster_pred_{bf['box']}_{i}", style = 'invis')
                bf['invisNode'] = f"cluster_pred_{bf['box']}_{i}"
                i+=1
                bf["node"] = f"cluster_pred_{bf['box']}"
                drawPIF(bf, f, data)
                drawPOF(bf, f, data, "c")
    # print("in bf: ",bf)


def drawBC(data, a):
    bf_dict = {}
    for bf in data.get("bf"):
        bf_dict[int(bf.get("box").split("-")[-1])] = 0

    for bc in data.get("bc"):
        for key, value in bc.items():
            if key != "metadata" and key != "box":
                if value in bf_dict:
                    with a.subgraph(name=f"cluster_{bc['box']}") as c:
                        drawBF(data, c, data.get("bf")[value - 1])
                        bf_dict[value] = 1
                        drawPIC(data, bc, c)
                        drawPOC(data, bc, c)

    for k in bf_dict.keys():
        if bf_dict.get(k) == 0:
            drawBF(data, a, data.get("bf")[k - 1])


def drawBL(data, a):
    bf_dict = {}
    for bf in data.get("bf"):
        bf_dict[int(bf.get("box").split("-")[-1])] = 0

    for bl in data.get("bl"):
        for key, value in bl.items():
            if key != "metadata" and key != "box":
                if value in bf_dict:
                    with a.subgraph(name=f"cluster_{bl['box']}") as c:
                        # print("if: ", data.get('bf')[value-1])
                        drawBF(data, c, data.get("bf")[value - 1])
                        bf_dict[value] = 1
                        drawPIL(data, bl, c)
                        drawPOL(data, bl, c)

    for k in bf_dict.keys():
        if bf_dict.get(k) == 0:
            drawBF(data, a, data.get("bf")[k - 1])

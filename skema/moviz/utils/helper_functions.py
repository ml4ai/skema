def drawPOF(bf, c, data, label):
    if data.get('pof') != None:
        for pof in data['pof']:           
            index = bf['box'].split('-')
            if str(pof['box']) == str(index[-1]):
                c.attr('node', shape = 'box')
                if pof.get('name') != None:
                    pof['node'] = f"pof-{bf['box']}"
                    c.node(name = f"pof-{bf['box']}", label = str(pof.get('name')) )
                    # c.attr(label = str(pof.get('name')))
                elif label != None:
                    pof['node'] = f"pof-{bf['box']}"
                    c.node(name = f"pof-{bf['box']}", label = label)                   
                else: 
                    pof['node'] = f"pof-{bf['box']}"
                    c.node(name = f"pof-{bf['box']}", fontcolor = "white")

def drawPIF(bf, c, data):
    if data.get('pif') != None:
        i = 1
        index = bf.get('box').split('-')
        for pif in data['pif']:
            if str(pif['box']) == str(index[-1]):
                c.attr('node', shape = 'box')
                pif['node'] = f"pif-{bf.get('box')}-{i}"
                c.node(name = f"pif-{bf.get('box')}-{i}", fontcolor = "white")
                i += 1

def drawOPO(b, c, data):
    if data.get('opo') != None:
        for opo in data['opo']:
            index = b.get('box').split('-')
            if str(opo.get('box')) ==  index[-1]:
                opo['node'] = f"opo-{b.get('box')}"                  
                c.attr('node', shape='box')
                c.node(name = f"opo-{b.get('box')}", fontcolor = "white")

def drawOPI(b, c, data):
    if data.get('opi') != None:
        for opi in data['opi']:
            index = b.get('box').split('-')
            if str(opi.get('box')) ==  index[-1]:
                if opi.get('name') != None:
                    opi['node'] = f"opi-{b.get('box')}-{str(opi.get('name'))}"
                    c.attr('node', shape='box')
                    c.node(name = f"opi-{b.get('box')}-{str(opi.get('name'))}", label=str(opi.get('name')))
                else:
                    opi['node'] = f"opi-{b.get('box')}"                  
                    c.attr('node', shape='box')
                    c.node(name = f"opi-{b.get('box')}", fontcolor = "white")

def drawPIC(data, bc, c):
    if data.get('pic') != None:
        for pic in data.get('pic'):
            index = bc['box'].split('-')
            # print(index)
            c.attr('node', shape = 'box')
            if str(pic['box']) == str(index[-1]):
                if pic.get('name') != None:
                    pic['node'] = f"pic-{bc['box']}"
                    c.node(name = str(pic.get('name')))
                else:
                    pic['node'] = f"pic-{bc['box']}"
                    c.node(name = f"pic-{bc['box']}", fontcolor = "white")        

def drawPOC(data, bc, c):
    if data.get('poc') != None:
        for poc in data.get('poc'):
            index = bc['box'].split('-')
            c.attr('node', shape = 'box')
            if str(poc['box']) == str(index[-1]):
                if poc.get('name') != None:
                    c.node(name = str(poc.get('name')))
                else:
                    c.node(name = f"poc-{bc['box']}", fontcolor = "white")    

def drawPIL(data, bl, c):
    if data.get('pil') != None:
        for pil in data.get('pil'):
            index = bl['box'].split('-')
            c.attr('node', shape = 'box')
            if str(pil['box']) == str(index[-1]):
                if pil.get('name') != None:
                    pil['node'] = f"pil-{bl['box']}"
                    c.node(name = f"pil-{bl['box']}", label = str(pil.get('name')))
                else:
                    pil['node'] = f"pil-{bl['box']}"
                    c.node(name = f"pil-{bl['box']}", fontcolor = "white")   

def drawPOL(data, bl, c):
    if data.get('pol') != None:
        for pol in data.get('pol'):
            index = bl['box'].split('-')
            c.attr('node', shape = 'box')
            if str(pol['box']) == str(index[-1]):
                if pol.get('name') != None:
                    pol['node'] = f"pol-{bl['box']}"
                    c.node(name = f"pol-{bl['box']}", label = str(pol.get('name')))
                else:
                    pol['node'] = f"pol-{bl['box']}"
                    c.node(name = f"pol-{bl['box']}", fontcolor = "white")   

def drawWFF(data, g):
    if data.get('wff') != None:
        for wff in data['wff']:
            for pif in data['pif']:
                for pof in data['pof']:
                    if wff['src'] == pif['id'] and wff['tgt'] == pof['id']:
                        print(pif.get('node'), pof.get('node'))
                        g.edge(pif.get('node'), pof.get('node'))

def drawWFC(data, g):
    if data.get('wfc') != None:
        for wfc in data.get('wfc'):
            for pof in data.get('pof'):
                for pic in data.get('pic'):
                    if wfc['src'] == pic['id'] and wfc['tgt'] == pof['id']:
                        # print("here: ",pic.get('node'), pof.get('node'))
                        g.edge(pic.get('node'), pof.get('node'))

def drawWFL(data, g):
    if data.get('wfl') != None:
        for wfc in data.get('wfl'):
            for pof in data.get('pof'):
                for pil in data.get('pil'):
                    if wfc['src'] == pil['id'] and wfc['tgt'] == pof['id']:
                        # print("here: ",pil.get('node'), pof.get('node'))
                        g.edge(pil.get('node'), pof.get('node'))

def drawWFOPO(data, g):
    if data.get('wfopo') != None:
        for wfopo in data.get('wfopo'):
            if data.get('opo') != None:
                for opo in data.get('opo'):
                    for pof in data.get('pof'):
                        if wfopo['src'] == opo['id'] and wfopo['tgt'] == pof['id']:
                            # print(opo.get('node'), pof.get('node'))
                            g.edge(opo.get('node'), pof.get('node'))

def drawWFOPI(data, g):
    if data.get('wfopi') != None:
        for wfopi in data.get('wfopi'):
            if data.get('opi') != None:
                for opi in data.get('opi'):
                    for pif in data.get('pif'):
                        if wfopi['src'] == opi['id'] and wfopi['tgt'] == pif['id']:
                            g.edge(opi['node'], pif['node'])

def drawWOPIO(data, g):
    if data.get('wopio') != None:
        for wopio in data.get('wopio'):
            if data.get('opo') != None:
                for opo in data.get('opo'):
                    if data.get('opi') != None:
                        for opi in data.get('opi'):
                            if wopio['src'] == opo['id'] and wopio['tgt'] == opi['id']:
                                g.edge(opo['node'], opi['node'])

# def drawWFCARGS(data, g):

#     for bf in data.get('fn').get(bf):

#     if data.get('wl_cargs') != None:
#         for wl_cargs in data.get('wl_cargs'):
#             for poc in data.get('poc'):
                
#                 if wl_cargs['src'] == ['id'] and wl_cargs['tgt'] == poc['id']:
#                     print("here: ",poc.get('node'), poc.get('node'))
#                     g.edge(pic.get('node'), pof.get('node'))

def drawBF(data, a, bf):
    if bf.get('function_type') == 'EXPRESSION':
        with a.subgraph(name=f"cluster_expr_{bf['box']}") as b: 
            bf['node'] = f"cluster_expr_{bf['box']}"
            b.attr('node', shape='box')
            drawPIF(bf, b, data)
            drawPOF(bf, b, data, None)
    if bf.get('function_type') == 'LITERAL':
        if bf.get('value').get('value_type') == "Integer":
            literal = str(bf.get('value').get('value'))
            with a.subgraph(name=f"cluster_lit_{literal}_{bf['box']}") as c: 
                bf['node'] = f"cluster_lit_{literal}_{bf['box']}"
                c.attr(label=literal)
                c.attr('node', shape='box')
                drawPIF(bf, c, data)
                drawPOF(bf, c, data, None)
        # if bf.get('value').get('value_type') == 'List':
    if bf['function_type'] == 'PRIMITIVE':
        primitive = str(bf['name'])
        with a.subgraph(name=f"cluster_prim_{primitive}_{bf['box']}") as d: 
            if primitive != None:
                bf['node'] = f"cluster_prim_{primitive}_{bf['box']}"
            d.attr(label=primitive)
            d.attr('node', shape='box')
            drawPIF(bf, d, data)
            drawPOF(bf, d, data, None)
    if bf.get('function_type') == 'FUNCTION':
        with a.subgraph(name=f"cluster_func_{bf['box']}") as e: #function
            bf['node'] = f"cluster_func_{bf['box']}"                        
            drawPOF(bf, e, data, None)
            drawPIF(bf, e, data)   
    if bf.get('function_type') == 'PREDICATE':
        with a.subgraph(name=f"cluster_pred_{bf['box']}") as f:
            bf['node'] = f"cluster_pred_{bf['box']}"
            drawPIF(bf, f, data)
            drawPOF(bf, f, data, 'c')

def drawBC(data, a):
    bf_dict = {}
    for bf in data.get('bf'):
        bf_dict[int(bf.get('box').split('-')[-1])] = 0

    for bc in data.get('bc'):
        for key, value in bc.items():
            if key != 'metadata' and key!= 'box':
                if value in bf_dict:
                    with a.subgraph(name = f"cluster_{bc['box']}") as c:
                        drawBF(data, c, data.get('bf')[value-1])
                        bf_dict[value] = 1
                        drawPIC(data, bc, c)
                        drawPOC(data, bc, c)                       

    for k in bf_dict.keys():
        if bf_dict.get(k) == 0:
            drawBF(data, a, data.get('bf')[k-1])

def drawBL(data, a):
    bf_dict = {}
    for bf in data.get('bf'):
        bf_dict[int(bf.get('box').split('-')[-1])] = 0

    for bl in data.get('bl'):
        for key, value in bl.items():
            if key != 'metadata' and key!= 'box':
                if value in bf_dict:
                    with a.subgraph(name = f"cluster_{bl['box']}") as c:
                        # print("if: ", data.get('bf')[value-1])
                        drawBF(data, c, data.get('bf')[value-1])
                        bf_dict[value] = 1
                        drawPIL(data, bl, c)
                        drawPOL(data, bl, c)                       

    for k in bf_dict.keys():
        if bf_dict.get(k) == 0:
            drawBF(data, a, data.get('bf')[k-1])
  

def drawPOF(bf, c, data):
    if data.get('pof') != None:
        for pof in data['pof']:
            index = bf['box'].split('-')
            if str(pof['box']) == str(index[-1]):
                c.attr('node', shape = 'box')
                if pof.get('name') != None:
                    pof['node'] = f"pof-{bf['box']}"
                    c.node(name = f"pof-{bf['box']}", fontcolor = "white")
                    c.attr(label = str(pof.get('name')))
                else: 
                    pof['node'] = f"pof-{bf['box']}"
                    c.node(name = f"pof-{bf['box']}", fontcolor = "white")
        #print(data.get('pof'))

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
            # print(pif)

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

def drawWFF(data, g):
    if data.get('wff') != None:
        for wff in data['wff']:
            for pif in data['pif']:
                for pof in data['pof']:
                    if wff['src'] == pif['id'] and wff['tgt'] == pof['id']:
                        g.edge(pif.get('node'), pof.get('node'))

def drawWFOPO(data, g):
        if data.get('wfopo') != None:
            for wfopo in data.get('wfopo'):
                if data.get('opo') != None:
                    for opo in data.get('opo'):
                        for pof in data.get('pof'):
                            if wfopo['src'] == opo['id'] and wfopo['tgt'] == pof['id']:
                                g.edge(opo['node'], pof['node'])

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


def drawBF(data, a):
    if data.get('bf') != None:
        for bf in data.get('bf'):
            if bf.get('function_type') == 'EXPRESSION':
                with a.subgraph(name=f"cluster_expr_{bf['box']}") as b: 
                    bf['node'] = f"cluster_expr_{bf['box']}"
                    b.attr('node', shape='box')
                    drawPIF(bf, b, data)
                    drawPOF(bf, b, data)
            if bf.get('function_type') == 'LITERAL':
                if bf.get('value').get('value_type') == "Integer":
                    literal = str(bf.get('value').get('value'))
                    with a.subgraph(name=f"cluster_lit_{literal}_{bf['box']}") as c: 
                        bf['node'] = f"cluster_lit_{literal}_{bf['box']}"
                        c.attr(label=literal)
                        c.attr('node', shape='box')
                        drawPIF(bf, c, data)
                        drawPOF(bf, c, data)
                # if bf.get('value').get('value_type') == 'List':
            if bf['function_type'] == 'PRIMITIVE':
                primitive = str(bf['name'])
                with a.subgraph(name=f"cluster_prim_{primitive}_{bf['box']}") as d: 
                    bf['node'] = f"cluster_prim_{primitive}_{bf['box']}"
                    d.attr(label=primitive)
                    d.attr('node', shape='box')
                    drawPIF(bf, d, data)
                    drawPOF(bf, d, data)
            if bf.get('function_type') == 'FUNCTION':
                with a.subgraph(name=f"cluster_func_{bf['box']}") as e: #function
                    bf['node'] = f"cluster_func_{bf['box']}"                        
                    drawPOF(bf, e, data)
                    drawPIF(bf, e, data)   
            if bf.get('function_type') == 'PREDICATE':
                with a.subgraph(name=f"cluster_pred_{bf['box']}") as f:
                    bf['node'] = f"cluster_pred_{bf['box']}"
                    drawPIF(bf, f, data)
                    drawPOF(bf, f, data)

            
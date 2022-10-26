import json
import graphviz

def drawPOF(bf, c, data):
    if data.get('pof') != None:
        for pof in data['pof']:
            index = bf['box'].split('-')
            if str(pof['box']) == str(index[-1]):
                c.attr('node', shape = 'box')
                if pof.get('name') != None:
                    pof['node'] = str(pof.get('name'))
                    c.node(name = str(pof['name']))
                else: 
                    pof['node'] = f"pof-{bf['box']}"
                    c.node(name = f"pof-{bf['box']}", fontcolor = "white")
        print(data.get('pof'))



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
                    # print(pif.get('node'), pof.get('node'))
                    if wff['src'] == pif['id'] and wff['tgt'] == pof['id']:
                        g.edge(pif.get('node'), pof.get('node'), fontcolor = "white")

def drawWFOPO(data, g):
        if data.get('wfopo') != None:
            for wfopo in data.get('wfopo'):
                if data.get('opo') != None:
                    for opo in data.get('opo'):
                        for pof in data.get('pof'):
                            if wfopo['src'] == opo['id'] and wfopo['tgt'] == pof['id']:
                                g.edge(opo['node'], pof['node'], fontcolor = "white")

def drawWFOPI(data, g):
    if data.get('wfopi') != None:
        for wfopi in data.get('wfopi'):
            if data.get('opi') != None:
                for opi in data.get('opi'):
                    for pif in data.get('pif'):
                        if wfopi['src'] == opi['id'] and wfopi['tgt'] == pif['id']:
                            g.edge(opi['node'], pif['node'], fontcolor = "white")

def drawBF(data, a):
    if data.get('value').get('bf') != None:
        for bf in data.get('value').get('bf'):
            if bf.get('function_type') == 'EXPRESSION':
                with a.subgraph(name=f"cluster_expr_{bf['box']}") as b: 
                    bf['node'] = f"cluster_expr_{bf['box']}"
                    b.attr('node', shape='box')
                    drawPIF(bf, b, data.get('value'))
                    drawPOF(bf, b, data.get('value'))
            if bf.get('function_type') == 'LITERAL':
                literal = str(bf.get('value').get('value'))
                with a.subgraph(name=f"cluster_lit_{literal}") as c: 
                    bf['node'] = f"cluster_lit_{bf['box']}"
                    c.attr(label=literal)
                    c.attr('node', shape='box')
                    drawPIF(bf, c, data.get('value'))
                    drawPOF(bf, c, data.get('value'))
            if bf['function_type'] == 'PRIMITIVE':
                primitive = str(bf['name'])
                with a.subgraph(name=f"cluster_prim_{primitive}") as d: 
                    bf['node'] = f"cluster_prim_{bf['box']}"
                    d.attr(label=primitive)
                    d.attr('node', shape='box')
                    drawPIF(bf, d, data.get('value'))
                    drawPOF(bf, d, data.get('value'))
            if bf.get('function_type') == 'FUNCTION':
                with a.subgraph(name=f"cluster_func_{bf['box']}") as e: #function
                    bf['node'] = f"cluster_func_{bf['box']}"                        
                    drawPOF(bf, e, data.get('value'))
                    drawPIF(bf, e, data.get('value'))    


# def drawWOPIO(data, g):


def draw_graph(PROGRAM_NAME):
    f = open (f"{PROGRAM_NAME}--Gromet-FN-auto.json", "r")
    data = json.loads(f.read())

    i = 1
    for item in data['fn']['bf']:
        item['box'] = f"fn-bf-{i}"
        i += 1

    i = 1
    for pof in data['fn']['pof']:
        pof["id"] = i
        i += 1

    i = 1
    if data.get('fn').get('pif') != None: 
        for pif in data['fn']['pif']:
            pif["id"] = i
            i += 1

    i = 1
    for attribute in data['attributes']:
        if attribute.get('value').get('b') != None:
            b_list = attribute.get('value').get('b')
            j=1
            
            for b_dict in b_list:
                b_dict['box'] = "attr-b-"+str(i)+"-"+str(j)
                j+=1
            i += 1

    i = 1
    for attribute in data['attributes']:
        if attribute.get('value').get('bf') != None:
            bf_list = attribute.get('value').get('bf')
            j=1 
            for bf_dict in bf_list:
                bf_dict['box'] = "attr-bf-"+str(i)+"-"+str(j)
                j += 1
            i+= 1
    
    for attribute in data['attributes']:
        j = 1
        if attribute.get('value').get('pof') != None: 
            pof_list = attribute.get('value').get('pof')
            for pof_dict in pof_list:
                pof_dict['id'] =  j
                j += 1
    
    i = 1
    for bf in data.get('fn').get('bf'):
        bf['contents'] = i
        i += 1
    
    #print(data.get('fn').get('bf'))

    g = graphviz.Graph('G', filename=PROGRAM_NAME, engine='fdp', format='png', directory='static')


    #LHS
    for b in data['fn']['b']:
        if b['function_type'] == 'MODULE':
            with g.subgraph(name='clusterA') as a: #module
                for bf in data['fn']['bf']:
                    if bf.get('function_type') == 'EXPRESSION':
                        with a.subgraph(name=f"cluster_expr_{bf['box']}") as c: #expression
                            bf['node'] = f"cluster_expr_{bf['box']}"                        
                            drawPOF(bf, c, data['fn'])
                            drawPIF(bf, c, data['fn'])
                    if bf.get('function_type') == 'LITERAL':
                        literal = str(bf.get('value').get('value'))
                        with a.subgraph(name=f"cluster_lit_{bf['box']}") as c: #literal
                            bf['node'] = f"cluster_lit_{bf['box']}"
                            c.attr(label=literal)
                            c.attr('node', shape='box')
                            drawPOF(bf, c, data['fn'])
                            drawPIF(bf, c, data['fn'])
                    if bf.get('function_type') == 'FUNCTION':
                        with a.subgraph(name=f"cluster_func_{bf['box']}") as c: #function
                            bf['node'] = f"cluster_func_{bf['box']}"                        
                            drawPOF(bf, c, data['fn'])
                            drawPIF(bf, c, data['fn'])


    drawWFF(data['fn'], g)

    #RHS
    for attribute in data.get('attributes'):
        for b in attribute.get('value').get('b'):
            if b.get('function_type') == 'EXPRESSION':
                with g.subgraph(name=f"cluster_expr_{b.get('box')}") as a:
                    b['node'] = f"cluster_expr_{b.get('box')}"
                    drawOPO(b, a, attribute.get('value'))
                    drawOPI(b, a, attribute.get('value'))
                    drawBF(attribute, a)
            if b.get('function_type') == 'FUNCTION':
                with g.subgraph(name=f"cluster_func_{b.get('box')}") as a:
                    a.attr(label = str(b.get('name')))
                    b['node'] = f"cluster_func_{b.get('box')}"
                    drawOPO(b, a, attribute.get('value'))
                    drawOPI(b, a, attribute.get('value'))
                    drawBF(attribute, a)
        drawWFF(attribute.get('value'), g)  
        drawWFOPO(attribute.get('value'), g)
        drawWFOPI(attribute.get('value'), g)

    #connecting LHS and RHS
    for bf in data.get('fn').get('bf'):
        for attribute in data.get('attributes'):
            for b in attribute.get('value').get('b'):
                index = b.get('box').split('-')
                if index[2] == str(bf['contents']):
                    g.edge(bf['node'], b['node'])

    g.view()
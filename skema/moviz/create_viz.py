import json
import graphviz

def draw_graph(PROGRAM_NAME):
    f = open (f"{PROGRAM_NAME}--Gromet-FN-auto.json", "r")
    data = json.loads(f.read())

    i = 1
    for item in data['fn']['bf']:
        item['box'] = 1
        i += 1

    i = 1
    for item in data['attributes'][0]['value']['b']:     
        item["box"] = i
        i += 1
        # print(item)

    i = 1
    for item in data['attributes'][0]['value']['bf']:     
        item["box"] = i
        i += 1
        # print(item)
    
    i = 1
    for pof in data['attributes'][0]['value']['pof']:
        # pof.update({'id': i})
        pof["id"] = i
        i += 1
        # print(pof)
    

    g = graphviz.Graph('G', filename=PROGRAM_NAME, engine='fdp', format='png', directory='static')

    #if(data['fn']['b'][0]['function_type'] == 'MODULE'):
    for b in data['fn']['b']:
        if b['function_type'] == 'MODULE':
            with g.subgraph(name='clusterA') as a: #module
                for bf in data['fn']['bf']:
                    if bf['function_type'] == 'EXPRESSION':
                        with a.subgraph(name='clusterC') as c: #expression
                            for pof in data['fn']['pof']:
                                if pof['box'] == bf['box']:
                                    c.attr('node', shape = 'box')
                                    c.node(name = str(pof['name']))                           
    
    # if(data['attributes'][0]['value']['b'][0]['function_type'] == 'EXPRESSION'):
    for b in data['attributes'][0]['value']['b']:
        if b['function_type'] == 'EXPRESSION':
            with g.subgraph(name='clusterB') as a:
                a.attr('node', shape='box')
                for opo in data['attributes'][0]['value']['opo']:
                    if opo['box'] == data['attributes'][0]['value']['b'][0]['box']:
                        a.node(name = f"opo-{opo['id']}", fontcolor = "white")
                for bf in data['attributes'][0]['value']['bf']:
                    if bf['function_type'] == 'LITERAL':
                        literal = str(bf['value']['value'])
                        # print(literal)
                        with a.subgraph(name=f"cluster-{literal}") as c: 
                            c.attr(label=literal)
                            c.attr('node', shape='box')
                            if 'pif' in data['attributes'][0]['value'].keys():                           
                                for pif in data['attributes'][0]['value']['pif']:
                                    if pif['box'] == bf['box']:
                                        c.node(name = f"pif-{pif['id']}", fontcolor = "white")
                            for pof in data['attributes'][0]['value']['pof']:
                                if pof['box'] == bf['box']:
                                    c.node(name = f"pof-{pof['id']}", fontcolor = "white") 
                    if bf['function_type'] == 'PRIMITIVE':
                        primitive = str(bf['name'])
                        # print(primitive)
                        with a.subgraph(name=f"cluster-{primitive}") as d: 
                            d.attr(label=primitive)
                            d.attr('node', shape='box')
                            for pif in data['attributes'][0]['value']['pif']:
                                if pif['box'] == bf['box']:
                                    d.node(name = f"pif-{pif['id']}", fontcolor = "white")
                            for pof in data['attributes'][0]['value']['pof']:
                                if pof['box'] == bf['box']:
                                    d.node(name = f"pof-{pof['id']}", fontcolor = "white")


    if 'wff' in data['attributes'][0]['value'].keys():
        print("hi")
        for wff in data['attributes'][0]['value']['wff']:
            for pif in data['attributes'][0]['value']['pif']:   
                for pof in data['attributes'][0]['value']['pof']:
                    if wff['src'] == pif['id'] and wff['tgt'] == pof['id']:
                        g.edge(f"pif-{pif['id']}", f"pof-{pof['id']}")

    for wfopo in data['attributes'][0]['value']['wfopo']:
        for opo in data['attributes'][0]['value']['opo']:
            for pof in data['attributes'][0]['value']['pof']:
                if wfopo['src'] == opo['id'] and wfopo['tgt'] == pof['id']:
                    g.edge(f"opo-{opo['id']}", f"pof-{pof['id']}")

    g.edge('clusterC', 'clusterB')

    g.view()
def init(data):

    i=1
    if data.get('fn').get('b') != None:
        for item in data.get('fn').get('b'):
            item['box'] = f"fn-b-{i}"
            i += 1

    i = 1
    if data.get("fn").get("bf") != None:
        for item in data.get('fn').get('bf'):
                item["box"] = f"fn-bf-{i}"
                i += 1

    i = 1
    if data.get("fn").get("pof") != None:
        for pof in data["fn"]["pof"]:
                pof["id"] = i
                i += 1

    # print(data.get('fn').get('pof'))
    i = 1
    if data.get("fn").get("pif") != None:
        for pif in data["fn"]["pif"]:
            pif["id"] = i
            i += 1

    i = 1
    for attribute in data["fn_array"]:
        if attribute.get("b") != None:
            b_list = attribute.get("b")
            j = 1

            for b_dict in b_list:
                b_dict["box"] = "attr-b-" + str(i) + "-" + str(j)
                j += 1
            i += 1

    i = 1
    for attribute in data["fn_array"]:
        if attribute.get("bf") != None:
            bf_list = attribute.get("bf")
            j = 1
            for bf_dict in bf_list:
                bf_dict["box"] = "attr-bf-" + str(i) + "-" + str(j)
                j += 1
            i += 1

    for attribute in data["fn_array"]:
        j = 1
        if attribute.get("pof") != None:
            pof_list = attribute.get("pof")
            for pof_dict in pof_list:
                pof_dict["id"] = j
                j += 1

    for attribute in data["fn_array"]:
        j = 1
        if attribute.get("pif") != None:
            pif_list = attribute.get("pif")
            for pif_dict in pif_list:
                pif_dict["id"] = j
                j += 1

    if data.get("fn").get("bc") != None:
        i = 1
        for bc in data.get("fn").get("bc"):
            bc["box"] = f"fn-bc-{i}"

    if data.get("fn").get("bl") != None:
        i = 1
        for bl in data.get("fn").get("bl"):
            bl["box"] = f"fn-bl-{i}"

    if data.get("fn").get("wfc") != None:
        for wfc in data.get("fn").get("wfc"):
            wfc["src"] = 1
    
    for attribute in data["fn_array"]:
        if attribute.get('wfopo') != None:
            for wfopo in attribute.get('wfopo'):
                wfopo['drawn'] = False
        if attribute.get('wfopi') != None:
            for wfopi in attribute.get('wfopi'):
                wfopi['drawn'] = False
        if attribute.get('wff') != None:
            for wff in attribute.get('wff'):
                wff['drawn'] = False
        

    if data.get("fn").get("wff") != None:
        for wff in data.get("fn").get('wff'):
            wff['drawn'] = False

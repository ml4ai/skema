import os, re, json

def count(eqn, e):
    c=0
    for word in eqn.split():
        if e in word:
            c+=1
    return c

def isfloat(num):
    #return (re.match("[|-|+]\d+.\d+$", num))
    try:
        float(num)
        return True
    except:
        return False

def isint(num):
    #return (re.match("[|-|+]\d+$", num))
    try:
        int(num)
        return True
    except:
        return False

def isfrac(num):
    return (re.match("[|-|+]?\d+\/\d+$", num))

def remove_unecc_tokens(eqn):
    eliminate = ['mspace', 'mtable', 'mtext', 'mathvariant', 'class', 'mpadded',
                'symmetric', 'fence', 'rspace', 'lspace', 'displaystyle', 'scriptlevel',
                'stretchy','form', 'movablelimits', 'maxsize', 'minsize', 'linethickness', 'mstyle']

    # An additional '&#xA0;' token has been removed during the selection of eqns, due to it repetative behaviour
    # which causes unneccesary complexity. Removing it doesn't affect the structure of the equation.

    keep = ['mo', 'mi', 'mfrac', 'mn', 'mrow']

    for e in eliminate:
        if e in eqn:
            c=count(eqn, e)
            for _ in range(c):
                idx = eqn.find(e)
                # find the '<' just before the e
                temp1 = eqn[:idx+1]
                temp2 = eqn[idx+1:]
                open_angle = [idx_open for idx_open, angle in enumerate(temp1) if angle == '<']
                close_angle = [idx_close for idx_close, angle in enumerate(temp2) if angle == '>']
                filtered = temp1[open_angle[-1]:]+temp2[:close_angle[0]+1]
                flag = False
                for k in keep:
                    if k in filtered:
                          flag=True
                          if e in ["movablelimits", "minsize"] and k in ["mo", "mi"]:
                              true_k = [k for f in filtered.split() if k in f and e not in f]
                              if len(true_k)>0: keep_token = true_k[0]
                          else:
                              keep_token = k
                if flag == True:
                    eqn = temp1[:open_angle[-1]]+f' <{keep_token}>'+temp2[close_angle[0]+1:]
                else:
                    eqn = temp1[:open_angle[-1]]+temp2[close_angle[0]+1:]

    return eqn

def remove_additional_tokens(eqn):
    if 'mtext' in eqn:
        mtext_pos = eqn.find('mtext')
        eqn  = eqn.replace(eqn[mtext_pos-1:], ' </math>')

    if '<mrow>' in eqn:
        try:
            eqn_arr = eqn.split()
            temp_eqn = list()

            idxs_close = []
            idxs_open = []
            for ind, i in enumerate(eqn_arr):
                if i == '<mrow>':
                    idxs_open.append(ind)
                if i == '</mrow>':
                    idxs_close.append(ind)

            if len(idxs_open) != len(idxs_close):
                if len(idxs_close)>len(idxs_open):
                    idxs_close = idxs_close[:len(idxs_open)]
                else:
                    idxs_open = idxs_open[:len(idxs_close)]

            c_begin = 0
            for c_end in idxs_close:
                _eqn_arr = eqn_arr[c_begin:c_end+1]
                begin_idx = _eqn_arr.index("<mrow>")
                end_idx = _eqn_arr.index("</mrow>")
                if begin_idx+2==end_idx:
                    temp_eqn+= _eqn_arr[:begin_idx] + [_eqn_arr[begin_idx+1]]
                else:
                    temp_eqn+=eqn_arr[c_begin:c_end+1]

                c_begin = c_end+1
            temp_eqn+= eqn_arr[c_begin:]
            return " ".join(temp_eqn)

        except:
            f=''
            for F in eqn.split():
                f=f+F+' '
            return f

    else:
        f=''
        for F in eqn.split():
            f=f+F+' '

        return f

def remove_hexComments(eqn):
    temp_arr = []
    eqn_split = eqn.split()

    skip_idx = None
    for _idx, _o in enumerate(eqn_split):
        if _idx!=skip_idx:
            if "&#x" in _o:
                temp_arr.append(_o.split(";")[0].strip())
                if _idx+1!=len(eqn_split)-1:
                    skip_idx = _idx+1

            elif "-->" in _o:
                temp_arr.append(_o.split("-->")[-1].strip())

            else:
                temp_arr.append(_o)

    final = " ".join(temp_arr)

    return final

def cleaning_mml(eqn):
    eqn = remove_unecc_tokens(eqn)
    eqn = remove_additional_tokens(eqn)
    if "&#x" in eqn:
        eqn = remove_hexComments(eqn)
    return eqn

def extract_inbetween_tokens(mml_eqn):
    mmls = [m for m in mml_eqn.split(' ') if m != '']
    mmlss = [m for m in mmls if '<' in m and len([t for t in m if t=='<']) ==2]
    mmls3 = []
    for i in mmlss:
        if '&#x' not in i:
            imml = [im for im in re.split('>|<',i) if im != '']
            if len(imml)==3 and imml[-1] != '/math':
                if len(imml[1])>1:
                    mmls3.append(imml[1])

    return mmls3

def tokenize(mml_eqn):
    mml_split = re.split('>|<',mml_eqn)
    tokenized_mml=''

    inbetween_tokens = extract_inbetween_tokens(mml_eqn)

    for token in mml_split:
        token = token.strip()

        if len(token)>0:
            if '&#x' in  token or len(token)==1:
                tokenized_mml += token

            elif token.isdigit():   # entire number is made up integers e.g. 12345
                for intgr in list(map(int, token)):
                    tokenized_mml += f' {intgr} '

            elif isfloat(token):  # eg. 120.456
                try:
                    token_arr = token.split('.')
                    for tok_idx, tok in enumerate(token_arr):
                        if tok_idx==1: tokenized_mml += '.'

                        for intgr in list(map(int, token_arr[tok_idx])):
                            tokenized_mml += f' {intgr} '
                except: pass

            elif isfrac(token):
                token_arr = token.split('/')

                for tok_idx, tok in enumerate(token_arr):
                    if tok_idx==1: tokenized_mml += '/'
                    for intgr in list(map(int, token_arr[tok_idx])):
                        tokenized_mml += f' {intgr} '

            elif token in inbetween_tokens:
                tokenized_mml += token
            else:
                tokenized_mml += ' <' + token +'> '

    return tokenized_mml

def reduce(mml):
    _arr = mml.split()
    _temp = list()

    skip_idx = 0
    for _idx, _a in enumerate(_arr):
        if _idx > skip_idx:
            if (_a in ["<mn>", "<mi>", "<mo>"]) \
                    and (_arr[_idx+2] in ["</mn>", "</mi>", "</mo>"]):

                _temp.append(_arr[_idx+1])
                _temp.append(f"__{_a}")
                skip_idx = _idx+2

            else:
                _temp.append(_a)

    return ("<math> " + " ".join(_temp))

if __name__ == '__main__':
    # paths
    with open("config.json", "r") as cfg:
        config = json.load(cfg)

    data_path = f"{config['data_path']}/{config['dataset_type']}"
    modified_mml_file = f"{data_path}/modified_mml.txt"
    train = open(f'{data_path}/original_mml.txt', 'r').readlines()
    train_new = open(modified_mml_file, 'w')
    c=0
    for ei, eqn in enumerate(train):
        if len(eqn)>1:
            try:
                # cleaning
                mml_eqn = cleaning_mml(eqn)
                # tokenize
                mml_eqn = tokenize(mml_eqn)
                # reducing length of mml by merginh tokens
                # mml_eqn = reduce(mml_eqn)
                # writing
                if '\n' not in mml_eqn:
                    mml_eqn+="\n"
                train_new.write(mml_eqn)

            except:
                c+=1
                train_new.write("REJECTED MATHML"+"\n")

    print(c)
    os.rename(f'{data_path}/modified_mml.txt', f'{data_path}/mml.txt')

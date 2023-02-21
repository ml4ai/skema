import os, re

def count(eqn, e):
    c=0
    for word in eqn.split():
        if e in word:
            c+=1
    return c

def cleaning_mml(eqn):

    eliminate = ['mspace', 'mtable', 'mtext', 'mathvariant', 'class', 'mpadded',
                'symmetric', 'fence', 'rspace', 'lspace', 'displaystyle', 'scriptlevel',
                'stretchy','form', 'movablelimits', 'maxsize', 'minsize', 'linethickness', 'mstyle']

    # An additional '&#xA0;' token has been removed during the selection of eqns, due to it repetative behaviour
    # which causes unneccesary complexity. Removing it doesn't affect the structure of the equation.

    keep = ['mo', 'mi', 'mfrac', 'mn', 'mfrac','mrow']

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
                          keep_token = k
                if flag == True:
                    eqn = temp1[:open_angle[-1]]+f' <{keep_token}> '+temp2[close_angle[0]+1:]
                else:
                    eqn = temp1[:open_angle[-1]]+temp2[close_angle[0]+1:]

    if 'mtext' in eqn:
        mtext_pos = eqn.find('mtext')
        eqn  = eqn.replace(eqn[mtext_pos-1:], ' </math>')

    if '<mrow>' in eqn:
        f=''
        for F in eqn.split():
            f=f+F+' '
        idxs_open = []
        idxs_close = []
        for ind, i in enumerate(f.split()):
            if i == '<mrow>':
                idxs_open.append(ind)
            if i == '</mrow>':
                idxs_close.append(ind)
        for o,c in zip(idxs_open, idxs_close):
            if len(f.split()[o:c+1])==3:
                to_replace = ''
                replace_with = ''
                for fs in f.split()[o:c+1]:
                    to_replace+=fs+' '
                replace_with = f.split()[o:c+1][1]+' '
                f=f.replace(to_replace, replace_with)
         #train_new.write(f+'\n')
    else:
        f=''
        for F in eqn.split():
            f=f+F+' '
        #train_new.write(f+'\n')

    return f

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
                #print(token)
                for intgr in list(map(int, token)):
                    tokenized_mml += f' {intgr} '

            elif isfloat(token):  # eg. 120.456
                #print(token)
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

if __name__ == '__main__':
    # paths
    with open("config.json", "r") as cfg:
        config = json.load(cfg)

    data_path = f"{config['data_path']}/{config['dataset_type']}"
    modified_mml_file = f"{data_path}/modified_mml.txt"
    train = open(f'{data_path}/original_mml.txt', 'r').readlines()
    train_new = open(modified_mml_file, 'w')

    for eqn in train:
        if len(eqn)>1:
            # cleaning
            mml_eqn = cleaning_mml(eqn)
            # tokenize
            tokenized_mml = tokenize(mml_eqn)

            # writing
            if '\n' not in tokenized_mml:
                train_new.write(tokenized_mml+'\n')

    os.rename(f'{data_path}/modified_mml.txt', f'{data_path}/mml.txt')

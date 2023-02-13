# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:36:21 2020

JSON to javascript converter
"""

import os

def json2js(json_data, output_file, var_name='eqn_src'):
    
    with open(output_file, 'w') as fout:
        fout.write(f'{var_name} = [\n')
        for i, datum in enumerate(json_data):
            fout.write('  {\n')
            fout.write(f'    eqn_num: {repr(datum["eqn_num"])},\n')
            fout.write(f'    src: {repr(datum["src"])},\n')
            fout.write(f'    mml: {repr(datum["mml"])}\n')
            fout.write('  }')
            if i < len(json_data):
                fout.write(',')
            fout.write('\n')
        fout.write('];')
    
    

if __name__ == '__main__':
    
    dir = "1402.0091"
    
    # json_data --> array of the dictionaries in a format like {'src': ---latex eqn---, 'mml': MathML code}
    json_data = []
    
    SRC_dir = f'/projects/temporary/automates/er/gaurav/1402_results/latex_equations/{dir}/Large_eqns'
    MML_dir = f'/projects/temporary/automates/er/gaurav/1402_results/Mathjax_mml/{dir}/Large_MML'
    
    for MML_file in os.listdir(MML_dir):
        
        temp_dict = {}
        
        eqn_num = MML_file.split(".")[0]
        eqn_num = eqn_num.replace("eqn", "")
        print(eqn_num)
        src_text = open(os.path.join(SRC_dir, MML_file), 'r').readlines()[0]
        mml_text = open(os.path.join(MML_dir, MML_file), 'r').readlines()[0]
        
        temp_dict['eqn_num'] = int(eqn_num)
        temp_dict['src'] = src_text
        temp_dict['mml'] = mml_text
        
        json_data.append(temp_dict)
    
    destination = f'/projects/temporary/automates/er/gaurav/1402_results/{dir}_json2js_File.js'
    
    json2js(json_data, destination)
    

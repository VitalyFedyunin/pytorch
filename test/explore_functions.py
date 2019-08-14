import re
import os
import json
import yaml
from collections import OrderedDict

path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
function_test_metadata_yaml = os.path.join(path, 'native_functions_test_metadata.yaml')
declarations_yaml =  os.path.join(path, '../aten/src/ATen/Declarations.cwrap')
under_test = dict()
declarations_th = dict()
allowed_fields = [
('aten_ported_cpu', 'If operator ported from TH to Aten', 'False')
]
declarations_cnames = dict()

def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)
yaml.add_representer(OrderedDict, represent_ordereddict)


def test_meta_add_function(fname):
    global under_test
    if fname not in under_test:
        under_test[fname]  = dict(func = fname)

def test_meta_modify_functions(fnames, **kwargs):
    global under_test
    if not kwargs:
        return
    if isinstance(fnames, str):
        fnames = [fnames]
    for fname in fnames:
        if fname not in under_test:
            under_test[fname]  = dict(func = fname)
        for k, v in kwargs.items():
            under_test[fname][k] = v

def update_test_meta():
    global under_test
    result = list()
    for k in sorted(under_test.keys()):
     result.append(OrderedDict(sorted(under_test[k].items(), key=lambda t: t[0] if t[0] != 'func' else '')))
    with open(function_test_metadata_yaml, 'w') as outfile:
        yaml.dump(result, outfile)


def get_all_functions():
    import json
    global under_test
    global declarations_th
    global declarations_cnames

    with open(declarations_yaml, 'r') as file:
        content = file.read()
        for match in re.finditer('\[\[(.*?)\]\]', content, re.S):
            # print(match.group(1))
            a = yaml.load(match.group(1))
            declarations_th[a['name']] = a
            if 'cname' in a:
                declarations_cnames[a['cname']] = a
                declarations_cnames[a['cname']+'_'] = a
            if 'options' in a:
                for o in a['options']:
                    if 'cname' in o:
                        # print(json.dumps(o,indent=4))
                        declarations_cnames[o['cname']] = a
                        declarations_cnames[o['cname']+'_'] = a

    with open(function_test_metadata_yaml, 'r') as file:
        for f in yaml.load(file.read()):
            under_test[f['func']] = f

    native = dict()
    with open(aten_native_yaml, 'r') as file:
        for f in yaml.load(file.read()):
            m = re.search(r'^([^(.]+)', f['func'])
            # m = re.search(r'^([^.(]+)', f['func'])
            if m:
                short_name = m.group(0)
                # if short_name[-1] == '_':
                #     short_name = short_name[:-1]
                f['short_name'] = short_name
                if short_name in native:
                    native[short_name].append(f)
                else:
                    native[short_name] = [f]
    # print(json.dumps(len(native.keys()),indent=4))
    print(json.dumps(native['sin_'],indent=4))
    print(json.dumps(declarations_cnames['zero_'],indent=4))
    # print(json.dumps(under_test['sin'],indent=4))

    # print(json.dumps(len(declarations_th.keys()), indent=4))
    total = 0
    for k, v in declarations_th.items():
        inc = 1
        if 'backends' in v:
            inc *= len(v['backends'])
        else:
            inc *= 2
        if 'options' in v:
            inc *= len(v['options'])
        total += inc
    print("Total declarations:", total)


    test_meta_modify_functions(under_test.keys(), requires_porting_from_th_cpu = False, requires_porting_from_th_cuda = False)

    total = 0
    for k, v in declarations_cnames.items():
        if k in native:
            if 'backends' in v:
                if 'CPU' in v['backends']:
                    total += 1
                    test_meta_modify_functions(k, requires_porting_from_th_cpu = True)
                if 'CUDA' in v['backends']:
                    total += 1
                    test_meta_modify_functions(k, requires_porting_from_th_cuda = True)
            else:
                total += 2
                test_meta_modify_functions(k, requires_porting_from_th_cpu = True, requires_porting_from_th_cuda = True)
        else:
            # print(json.dumps(v, indent=4))
            pass
    print("Total recorded declarations:", total)

    for k, v in native.items():
        for item in v:
            if 'dispatch' in item and type(item['dispatch'])  == dict:
                for dk, dv in item['dispatch'].items():
                    if dv.startswith('legacy::cpu::'):
                        test_meta_modify_functions(k, requires_porting_from_th_cpu = True)
                    if dv.startswith('legacy::cuda::'):
                        test_meta_modify_functions(k, requires_porting_from_th_cuda = True)


get_all_functions()
update_test_meta()

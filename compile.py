from textx import metamodel_from_str, get_children_of_type
from collections import namedtuple
import logging
import shutil
from functools import lru_cache
import os, textx, re, json, sys

dirname = 'functions'

try:
    shutil.rmtree(dirname)
except:
    pass
os.mkdir(dirname)

grammer = open('grammer').read()

mm = metamodel_from_str(grammer)
code = [open(x).read() for x in sys.argv[1:]]
code = '\n'.join(code)
code = re.sub(r'(//.*)|(/\*[\s\S]*?\*/)', '', code)
print(code)
model = mm.model_from_str(code)

cname = lambda o: o.__class__.__name__
classes_info = {

}
MAX_SIZE = 16

OBJECTIVE = 'cmc'
namespace = 'cmc'
ti = 0

load_file = open(f'{dirname}/load.mcfunction', 'w')
abcdefg = []
Argument = namedtuple('Argument', ['type', 'name'])
FMethod = namedtuple('Method', ['static', 'returnType', 'name', 'template', 'arguments', 'cb', 'author'])
left = model.classes[:]


def add_iby(e, o):
    for c in classes_info[e]['inherits']:
        if c == e:
            classes_info[c]['iby'].append(o)
        else:
            add_iby(c, o)


while left:
    cleft = left[:]
    for kklass in left:
        if kklass.e != '':
            if kklass.e not in classes_info:
                continue
        kname = kklass.name
        classes_info[kname] = {
            'static': [],
            'field': [],
            'inherits': [kname],
            'methods': {},
            'iby': []
        }
        if kklass.e != '':
            classes_info[kname]['static'].extend(classes_info[kklass.e]['static'])
            classes_info[kname]['field'].extend(classes_info[kklass.e]['field'])
            classes_info[kname]['inherits'].extend(classes_info[kklass.e]['inherits'])
            classes_info[kname]['methods'].update(classes_info[kklass.e]['methods'])
            add_iby(kklass.e, kname)
        for thing in kklass.things:
            if cname(thing) == 'Field':
                if thing.static:
                    classes_info[kname]['static'].append(thing)
                else:
                    classes_info[kname]['field'].append(thing)
            elif cname(thing) == 'Method':
                classes_info[kname]['methods'][thing.name] = FMethod(thing.static, thing.returnType, thing.name,
                                                                     thing.template, thing.arguments, thing.cb, kname)
        abcdefg.append([0] * len(classes_info[kname]['static']))
        cleft.remove(kklass)
    # print(left)
    left = cleft
    print(left)


# print(classes_info)


def get_size(class_name):
    if class_name == 'int': return 1
    assert class_name in classes_info, f'Undefined class {class_name}'
    return len(classes_info[class_name]['field'])


load_file.write(f'gamerule maxCommandChainLength 1000000\n')
load_file.write(f'data modify storage {namespace}:main static set value {repr(abcdefg)}\n')
load_file.write(
    f'data modify storage {namespace}:main functions set value [[0]]\n')
load_file.write(f'data modify storage {namespace}:main temps set value []\n')
load_file.write(f'data modify storage {namespace}:main i_stack set value []\n')
load_file.write(f'data modify storage {namespace}:main temp_s set value []\n')
load_file.write(f'data modify storage {namespace}:main loops set value []\n')
load_file.write(f'scoreboard objectives add {OBJECTIVE} dummy\n')
load_file.write(f'scoreboard players set $hi {OBJECTIVE} 0\n')
load_file.write(f'scoreboard players set $returned {OBJECTIVE} 0\n')
load_file.write(f'scoreboard players set $error {OBJECTIVE} 0\n')
load_file.write(f'function {namespace}:load_heap\n')
load_file.write(f'data modify storage {namespace}:main heap[0].value[0] set value {repr([0] * get_size("Main"))}\n')
load_file.write(f'function {namespace}:method_main_main\n')


# load_file.write(f'function {namespace}:method_main_start\n')


def get_temp():
    global ti
    ti += 1
    return f'$t{ti}'


def opname(op):
    if op == '+':
        return 'add'
    if op == '-':
        return 'sub'
    if op == '*':
        return 'mul'
    if op == '/':
        return 'div'
    if op == '%':
        return 'mod'
    raise NotImplementedError(op)


def generate_expression(l1o, local, klass, cur_m, lvals):
    comp = l1o.expr
    out_type = ''
    temp = ''
    # if tempss is None:
    #     temps = []
    # else:
    #     temps = tempss
    a = ''
    for l1i, l1o in enumerate(comp.comp):
        ops = l1o.op
        prods = l1o.prod
        l1_temp = ''
        l1_type = ''
        for l2i, l2o in enumerate(prods):
            l2_temp = ''
            l2_type = ''
            for l3i, l3o in enumerate(l2o.bools):
                l3_temp = ''
                l3_type = ''
                for l4i, l4o in enumerate(l3o.vals):
                    a, l4_temp, l4_type = evaluate_final(a, cur_m, klass,
                                                         l4o, local,
                                                         list(filter(lambda x: x != '', [l1_temp, l2_temp, l3_temp])),
                                                         lvals)
                    if l4i == 0:
                        l3_type = l4_type
                        l3_temp = l4_temp
                    if l4i != 0:
                        assert l4_type == 'int', 'Can only perform boolean operations on integers'
                        # print(l3o, l4i, l3o.vals, l3o.op, code[l3o._tx_position: l3o._tx_position_end])
                        op = l3o.op[l4i - 1]
                        if op == '&&':
                            a += f'scoreboard players operation {l3_temp} {OBJECTIVE} *= {l4_temp} {OBJECTIVE}\n'
                        elif op == '||':
                            a += f'execute unless score {l4_temp} {OBJECTIVE} matches 0 run scoreboard players set {l3_temp} 1\n'
                if l3i == 0:
                    l2_type = l3_type
                    l2_temp = l3_temp
                if l3i != 0:
                    # assert 'int' == l2_type == l3_type, f'Trying to multiply/divide/modulus not-integer values'
                    # a += f'scoreboard players operation {l2_temp} {OBJECTIVE} {l2o.op[l3i - 1]}= {l3_temp} {OBJECTIVE}\n'
                    # print(l3_type, l2_type, cur_m, code[l1o._tx_position:l1o._tx_position_end])
                    assert l3_type == l2_type, f'Trying to multiply/divide objects of different types'
                    if l3_type == 'int':
                        a += f'scoreboard players operation {l2_temp} {OBJECTIVE} {l2o.op[l3i - 1]}= {l3_temp} {OBJECTIVE}\n'
                    else:
                        mklass = l3_type
                        m = classes_info[mklass]['methods'][opname(l2o.op[l3i - 1])]
                        a = static_call(a, l2_temp, l3_temp, m, mklass, [l1_temp])
            if l2i == 0:
                l1_temp = l2_temp
                l1_type = l2_type
            if l2i != 0:
                assert l1_type == l2_type, f'Trying to add/subtract objects of different types'
                if l1_type == 'int':
                    a += f'scoreboard players operation {l1_temp} {OBJECTIVE} {ops[l2i - 1]}= {l2_temp} {OBJECTIVE}\n'
                else:
                    mklass = l1_type
                    m = classes_info[mklass]['methods'][opname(ops[l2i - 1])]
                    a = static_call(a, l1_temp, l2_temp, m, mklass, [])
        if l1i == 0:
            temp = l1_temp
            out_type = l1_type
        assert out_type == l1_type, 'Comparison of values of different types is always false'
        if l1i != 0:
            assert l1_type == 'int', 'Can only compare integers (for now)'
            op = comp.op[l1i - 1]
            if op == '==':
                a += f'execute store success score {temp} {OBJECTIVE} if score {temp} {OBJECTIVE} = {l1_temp} {OBJECTIVE}\n'
            elif op == '!=':
                a += f'execute store success score {temp} {OBJECTIVE} unless score {temp} {OBJECTIVE} = {l1_temp} {OBJECTIVE}\n'
            else:
                a += f'execute store success score {temp} {OBJECTIVE} if score {temp} {OBJECTIVE} {op} {l1_temp} {OBJECTIVE}\n'
    return a, temp, out_type


def static_call(code, val1, val2, method, mklass, used_temps):
    fcall = f'function {namespace}:method_{mklass.lower()}_{method.name.lower()}'
    code += f'data modify storage {namespace}:main temp_s append value [0,0]\n'
    code += f'execute store result storage {namespace}:main temp_s[-1][0] int 1 run scoreboard players get {val1} {OBJECTIVE}\n'
    code += f'execute store result storage {namespace}:main temp_s[-1][1] int 1 run scoreboard players get {val2} {OBJECTIVE}\n'
    code += f'data modify storage {namespace}:main functions append from storage {namespace}:main temp_s[-1]\n'
    code += save_temps(used_temps)
    code += f'{fcall}\n'
    code += f'data remove storage {namespace}:main functions[-1]\n'
    code += f'data remove storage {namespace}:main temp_s[-1]\n'
    code += unsave_temps(used_temps)
    code += f'scoreboard players operation {val1} {OBJECTIVE} = $r {OBJECTIVE}\n'
    return code


def save_temps(used_temps):
    a = ''
    if len(used_temps) > 0:
        a += f'data modify storage {namespace}:main temps append value {repr([0] * len(used_temps))}\n'
        for i, t in enumerate(used_temps):
            a += f'execute store result storage {namespace}:main temps[-1][{i}] int 1 run scoreboard players get {t} {OBJECTIVE}\n'
    return a


def unsave_temps(used_temps):
    a = ''
    if len(used_temps) > 0:
        for i, t in enumerate(used_temps):
            a += f'execute store result score {t} {OBJECTIVE} run data get storage {namespace}:main temps[-1][{i}]\n'
        a += f'data remove storage {namespace}:main temps[-1]\n'
    return a


def get_code(value_expr):
    return code[value_expr._tx_position - 10: value_expr._tx_position_end + 10]


def evaluate_final(a, cur_m, klass, value_expr, local, used_temps, lvals):
    if cname(value_expr) == 'NUMBER':
        l3_temp = get_temp()
        l3_type = 'int'
        a += f'scoreboard players set {l3_temp} {OBJECTIVE} {value_expr.nval}\n'
    elif cname(value_expr) == 'NOT':
        out_code, out_temp, temp_type = generate_expression(value_expr.negate, local, klass, cur_m, lvals)
        assert temp_type == 'int', 'Can only negate integer operations'
        a += out_code
        l3_temp = out_temp
        l3_type = temp_type
        a += f'execute store success score {l3_temp} {OBJECTIVE} if score {l3_temp} {OBJECTIVE} matches 0\n'
    elif cname(value_expr) == 'FUNCTION_CALL':
        fname = value_expr.func_name
        this = get_temp()
        if fname.var != '':
            assert fname.var in classes_info[klass]['methods'], f'Method {fname.var} not found in class {klass}'
            m = classes_info[klass]['methods'][fname.var]
        else:
            if fname.expr in classes_info and len(fname.props) == 1:  # static function
                mklass = fname.expr
                assert fname.props[0] in classes_info[mklass][
                    'methods'], f'Method {fname.var} not found in class {mklass}'
                m = classes_info[mklass]['methods'][fname.props[0]]
                assert m.static, 'Can\'t call non-static method by referencing class-name.'
            else:
                callee = fname.props[:-1]
                function = fname.props[-1]
                org = ''
                if cname(fname.expr) == 'str':
                    org = fname.expr
                    name = list(filter(lambda x: x.name == org, local))
                    assert len(name) != 0, f'Undefined variable {org} {get_code(value_expr)}'
                    arg_stuff = name[0]
                    setup, path, save = get_path(arg_stuff, local, klass, 'functions[-1][0]')
                    cur_klass = arg_stuff.type
                    a += setup
                else:
                    out_code, out_temp, out_type = generate_expression(fname.expr.expr, local, klass, cur_m, lvals)
                    cur_klass = out_type
                    a += out_code
                    path = out_temp
                for caller in callee:
                    args = classes_info[cur_klass]['field']
                    name = list(filter(lambda x: x.name == caller, args))
                    assert len(name) != 0, f'Can\'t find property {caller}'
                    arg_stuff = name[0]
                    setup, path, save = get_path(arg_stuff, args, cur_klass, path.split()[-1])
                    cur_klass = arg_stuff.type
                    a += setup
                assert function in classes_info[cur_klass][
                    'methods'], f'Method {function} not found in class {cur_klass}'
                m = classes_info[cur_klass]['methods'][function]
                if path.startswith('$'):
                    a += f'scoreboard players operation {this} {OBJECTIVE} = {path} {OBJECTIVE}\n'
                else:
                    a += f'execute store result score {this} {OBJECTIVE} run data get storage {path}\n'
        mklass = m.author
        l3_type = m.returnType
        l3_temp = get_temp()
        fcall = f'function {namespace}:method_{mklass.lower()}_{m.name.lower()}'
        i = 0
        a += f'data modify storage {namespace}:main temp_s append value {repr([0] * (len(value_expr.func_call) + (not m.static)))}\n'
        if not m.static:
            if fname.var != '':
                assert not cur_m.static, f'Static method {cur_m.name} can\'t call not-static method.'
                a += f'data modify storage {namespace}:main temp_s[-1][{i}] set from storage {namespace}:main ' \
                     'functions[-1][0]\n'
            else:
                a += f'execute store result storage {namespace}:main temp_s[-1][{i}] int 1 run scoreboard players get {this} {OBJECTIVE}\n'
            i += 1
        for iiii, arg in enumerate(value_expr.func_call):
            o_c, o_t, t_t = generate_expression(arg, local, klass, cur_m, lvals)
            a += o_c
            # print(m.arguments[iiii].type, t_t)
            assert m.arguments[iiii].type == t_t, 'Invalid type!'
            a += f'execute store result storage {namespace}:main temp_s[-1][{i}] int 1 run scoreboard players get {o_t} {OBJECTIVE}\n'
            i += 1
        a += f'data modify storage {namespace}:main functions append from storage {namespace}:main temp_s[-1]\n'
        a += save_temps(used_temps)
        a += f'{fcall}\n'
        a += f'data remove storage {namespace}:main functions[-1]\n'
        a += f'data remove storage {namespace}:main temp_s[-1]\n'
        a += unsave_temps(used_temps)
        a += f'scoreboard players operation {l3_temp} {OBJECTIVE} = $r {OBJECTIVE}\n'

    elif cname(value_expr) == 'PROP':
        l3_temp = get_temp()
        if value_expr.var != '':
            if value_expr.var == 'this':
                a += f'execute store result score {l3_temp} {OBJECTIVE} run data get storage {namespace}:main functions[-1][0]\n'
                l3_type = klass
            else:
                name = list(filter(lambda x: x.name == value_expr.var, local))
                assert len(name) != 0, f'Undefined variable {value_expr.var} {get_code(value_expr)}'
                arg_stuff = name[0]
                setup, path, save = get_path(arg_stuff, local, klass, 'functions[-1][0]')
                a += setup
                a += f'execute store result score {l3_temp} {OBJECTIVE} run data get storage {path}\n'
                l3_type = arg_stuff.type
        else:
            var1 = value_expr.props[0]
            callee = value_expr.props[1:]
            org = ''
            # print(value_expr, code[value_expr._tx_position:value_expr._tx_position_end])
            if cname(value_expr.expr) == 'str':
                org = value_expr.expr
                if org in classes_info:
                    name = list(filter(lambda x: x.name == var1, classes_info[org]['static']))
                    assert len(name) != 0, f'Undefined variable {var1} {get_code(value_expr)}'
                    arg_stuff = name[0]
                    setup, path, save = get_path(arg_stuff, local, org, 'functions[-1][0]')
                    l3_type = arg_stuff.type
                    a += setup
                    # raise NotImplementedError('static things are hard')
                else:
                    name = list(filter(lambda x: x.name == org, local))
                    assert len(name) != 0, f'Undefined variable {org}'
                    arg_stuff = name[0]
                    setup, path, save = get_path(arg_stuff, local, klass, 'functions[-1][0]')
                    l3_type = arg_stuff.type
                    a += setup
                    callee = [var1] + callee
            else:
                out_code, out_temp, out_type = generate_expression(value_expr.expr.expr, local, klass, cur_m, lvals)
                l3_type = out_type
                a += out_code
                path = out_temp
                callee = [var1] + callee
            for caller in callee:
                # print(path, path.split()[-1])
                args = classes_info[l3_type]['field']
                name = list(filter(lambda x: x.name == caller, args))
                assert len(name) != 0, (
                    f'Can\'t find property {caller}', code[value_expr._tx_position:value_expr._tx_position_end])
                arg_stuff = name[0]
                setup, path, save = get_path(arg_stuff, args, l3_type, path.split()[-1])
                l3_type = arg_stuff.type
                a += setup
            # print(path)
            if path.startswith('$'):
                a += f'scoreboard players operation {l3_temp} {OBJECTIVE} = {path} {OBJECTIVE}\n'
            else:
                a += f'execute store result score {l3_temp} {OBJECTIVE} run data get storage {path}\n'
    elif cname(value_expr) == 'PARENTETHIS':
        out_code, l3_temp, l3_type = generate_expression(value_expr.expr, local, klass, cur_m, lvals)
        a += out_code
    elif cname(value_expr) == 'CONSTRUCTOR':
        mklass = value_expr.class_name
        assert mklass in classes_info[mklass][
            'methods'], f'Constructor not found in class {mklass} {get_code(value_expr)} {value_expr.class_name}'
        m = classes_info[mklass]['methods'][mklass]
        l3_type = mklass
        l3_temp = get_temp()
        fcall = f'function {namespace}:method_{mklass.lower()}_{m.name.lower()}'
        a += f'data modify storage {namespace}:main temp_s append value {repr([0] * (len(value_expr.args) + (not m.static)))}\n'
        i = 1
        assert not m.static, 'A constructor can\'t be static'
        a += f'scoreboard players add $hi {OBJECTIVE} 1\n'
        a += f'scoreboard players operation $index {OBJECTIVE} = $hi {OBJECTIVE}\n'
        a += f'data modify storage {namespace}:main obj set value {repr([0] * get_size(mklass) + [get_index(mklass)])}\n'
        a += f'function {namespace}:set_heap\n'
        a += f'data modify storage {namespace}:main temp_s[-1] append value 0\n'
        a += f'execute store result storage {namespace}:main temp_s[-1][0] int 1 run scoreboard players get $index {OBJECTIVE}\n'
        for iiii, arg in enumerate(value_expr.args):
            o_c, o_t, t_t = generate_expression(arg, local, klass, cur_m, lvals)
            a += o_c
            assert m.arguments[iiii].type == t_t, 'Invalid type!'
            a += f'execute store result storage {namespace}:main temp_s[-1][{i}] int 1 run scoreboard players get {o_t} {OBJECTIVE}\n'
            i += 1
        a += f'data modify storage {namespace}:main functions append from storage {namespace}:main temp_s[-1]\n'
        a += save_temps(used_temps)
        a += f'{fcall}\n'
        a += f'execute store result score {l3_temp} {OBJECTIVE} run data get storage {namespace}:main functions[-1][0]\n'
        a += f'data remove storage {namespace}:main functions[-1]\n'
        a += f'data remove storage {namespace}:main temp_s[-1]\n'
        a += unsave_temps(used_temps)
    elif cname(value_expr) == 'ACONSTRUCTOR':
        out_code, out_temp, out_type = generate_expression(value_expr.length, local, klass, cur_m, lvals)
        a += out_code
        assert out_type == 'int', 'Length may only be int'
        l3_temp = get_temp()
        temp_temp = get_temp()
        l3_type = value_expr.class_name + '[]'
        # take length, divide by 128, add 1, create N 128 arrays. encapsulate temp, and prepend it. save everything to heap, done!
        a += f'data modify storage {namespace}:main obj set value [[0]]\n'
        a += f'execute store result storage {namespace}:main obj[0][0] int 1 run scoreboard players operation $length {OBJECTIVE} = {out_temp} {OBJECTIVE}\n'
        a += f'scoreboard players set {temp_temp} {OBJECTIVE} 128\n'
        a += f'scoreboard players operation $length {OBJECTIVE} /= {temp_temp} {OBJECTIVE}\n'
        a += f'scoreboard players add $length {OBJECTIVE} 1\n'
        a += f'function {namespace}:create_array\n'
        a += f'scoreboard players add $hi {OBJECTIVE} 1\n'
        a += f'execute store result score {l3_temp} {OBJECTIVE} run scoreboard players operation $index {OBJECTIVE} = $hi {OBJECTIVE}\n'
        a += f'function {namespace}:set_heap\n'
        # raise Exception('Unhandled type: ', cname(value_expr))
    elif cname(value_expr) == 'INDEX':
        a, out_temp, out_type = evaluate_final(a, cur_m, klass, value_expr.arr, local, used_temps, lvals)
        a2, out_temp2, out_type2 = generate_expression(value_expr.i, local, klass, cur_m, lvals)
        a += a2
        assert out_type.endswith('[]'), 'Can\'t index non-array object'
        l3_type = out_type[:-2]
        l3_temp = get_temp()
        temp_temp = get_temp()
        a += f'scoreboard players operation $index {OBJECTIVE} = {out_temp} {OBJECTIVE}\n'
        a += f'function {namespace}:get_heap\n'
        a += f'execute store result score $index {OBJECTIVE} run data get storage {namespace}:main obj\n'
        a += f'scoreboard players set {temp_temp} {OBJECTIVE} 128\n'
        a += f'scoreboard players remove $index {OBJECTIVE} 1\n'
        a += f'scoreboard players operation $index {OBJECTIVE} *= {temp_temp} {OBJECTIVE}\n'
        a += f'scoreboard players operation $index {OBJECTIVE} -= {out_temp2} {OBJECTIVE}\n'
        a += f'scoreboard players remove $index {OBJECTIVE} 1\n'
        a += f'function {namespace}:array_access\n'
        a += f'scoreboard players operation $index {OBJECTIVE} %= {temp_temp} {OBJECTIVE}\n'
        a += f'function {namespace}:subarray_access\n'
        a += f'scoreboard players operation {l3_temp} {OBJECTIVE} = $arr {OBJECTIVE}\n'
    elif cname(value_expr) == 'SCOREBOARD':
        l3_type = 'int'
        l3_temp = get_temp()
        a += f'scoreboard players operation {l3_temp} {OBJECTIVE} = {value_expr.name} {value_expr.objective}\n'
    elif cname(value_expr) == 'TEMPLATE':
        fname = value_expr.func_name
        this = get_temp()
        if fname.var != '':
            assert fname.var in classes_info[klass]['methods'], f'Method {fname.var} not found in class {klass}'
            m = classes_info[klass]['methods'][fname.var]
        else:
            if fname.expr in classes_info and len(fname.props) == 1:  # static function
                mklass = fname.expr
                assert fname.props[0] in classes_info[mklass][
                    'methods'], f'Method {fname.var} not found in class {mklass}'
                m = classes_info[mklass]['methods'][fname.props[0]]
                assert m.static, 'Can\'t call non-static method by referencing class-name.'
            else:
                callee = fname.props[:-1]
                function = fname.props[-1]
                org = ''
                if cname(fname.expr) == 'str':
                    org = fname.expr
                    name = list(filter(lambda x: x.name == org, local))
                    assert len(name) != 0, f'Undefined variable {org}'
                    arg_stuff = name[0]
                    setup, path, save = get_path(arg_stuff, local, klass, 'functions[-1][0]')
                    cur_klass = arg_stuff.type
                    a += setup
                else:
                    out_code, out_temp, out_type = generate_expression(fname.expr.expr, local, klass, cur_m, lvals)
                    cur_klass = out_type
                    a += out_code
                    path = out_temp
                for caller in callee:
                    args = classes_info[cur_klass]['field']
                    name = list(filter(lambda x: x.name == caller, args))
                    assert len(name) != 0, f'Can\'t find property {caller}'
                    arg_stuff = name[0]
                    setup, path, save = get_path(arg_stuff, args, cur_klass, path.split()[-1])
                    cur_klass = arg_stuff.type
                    a += setup
                assert function in classes_info[cur_klass][
                    'methods'], f'Method {fname.var} not found in class {cur_klass}'
                m = classes_info[cur_klass]['methods'][function]
                if path.startswith('$'):
                    a += f'scoreboard players operation {this} {OBJECTIVE} = {path} {OBJECTIVE}\n'
                else:
                    a += f'execute store result score {this} {OBJECTIVE} run data get storage {path}\n'
        mklass = m.author
        l3_type = m.returnType
        l3_temp = get_temp()
        assert len(m.template) == len(
            value_expr.tvals), f'Template doesn\'t contain the same number of arguments {code[value_expr._tx_position:value_expr._tx_position_end]} {value_expr.tvals} {m.template}'
        print(value_expr.tvals, m.template)
        tname = f'template_{get_temp()[1:]}'
        file = open(tname + '.mcfunction', 'w')
        generate(m.cb, ([] if m.static else [Argument(a, 'this')]) + m.arguments, file, tname, mklass, m,
                 {'json': json, **dict(zip(m.template, map(lambda x: eval(x, lvals), value_expr.tvals)))})
        file.write(
            f'execute unless score $error {OBJECTIVE} matches 1 run scoreboard players set $returned {OBJECTIVE} 0')
        file.close()
        fcall = f'function {namespace}:{tname}'
        i = 0
        a += f'data modify storage {namespace}:main temp_s append value {repr([0] * (len(value_expr.func_call) + (not m.static)))}\n'
        if not m.static:
            if fname.var != '':
                assert not cur_m.static, f'Static method {cur_m.name} can\'t call not-static method.'
                a += f'data modify storage {namespace}:main temp_s[-1][{i}] set from storage {namespace}:main ' \
                     'functions[-1][0]\n'
            else:
                a += f'data modify storage {namespace}:main temp_s[-1] append value 0\n'
                a += f'execute store result storage {namespace}:main temp_s[-1][{i}] int 1 run scoreboard players get {this} {OBJECTIVE}\n'
            i += 1
        for iiii, arg in enumerate(value_expr.func_call):
            o_c, o_t, t_t = generate_expression(arg, local, klass, cur_m, lvals)
            a += o_c
            assert m.arguments[iiii].type == t_t, 'Invalid type!'
            a += f'execute store result storage {namespace}:main temp_s[-1][{i}] int 1 run scoreboard players get {o_t} {OBJECTIVE}\n'
            i += 1
        a += f'data modify storage {namespace}:main functions append from storage {namespace}:main temp_s[-1]\n'
        a += save_temps(used_temps)
        a += f'{fcall}\n'
        a += f'data remove storage {namespace}:main functions[-1]\n'
        a += f'data remove storage {namespace}:main temp_s[-1]\n'
        a += unsave_temps(used_temps)
        a += f'scoreboard players operation {l3_temp} {OBJECTIVE} = $r {OBJECTIVE}\n'
    elif cname(value_expr) == 'CAST':
        out_code, out_temp, temp_type = generate_expression(value_expr.tocast, local, klass, cur_m, lvals)
        a += out_code
        l3_temp = out_temp
        l3_type = value_expr.ctype
        # print(classes_info[value_expr.ctype]['inherits'])
        # print(value_expr.ctype, l3_type, l3_temp)
        # raise NotImplementedError('hi')
    else:
        # print(l3o)
        raise NotImplementedError('Unhandled type: ', cname(value_expr))
    return a, l3_temp, l3_type


def get_index(klass_name):
    return list(classes_info.keys()).index(klass_name)


def get_path(variable, variables, klass, this_path):
    if cname(variable) == 'Argument':
        return '', f'{namespace}:main functions[-1][{variables.index(variable)}]', ''
    elif cname(variable) == 'Field':
        if not variable.static:
            index = classes_info[klass]["field"].index(variable)
            # print(this_path)
            if this_path.startswith('$'):
                return f'''scoreboard players operation $index {OBJECTIVE} = {this_path} {OBJECTIVE}
function {namespace}:get_heap\n''', f'{namespace}:main obj[{index}]', f'function {namespace}:set_heap\n'
            else:
                return f'''execute store result score $index {OBJECTIVE} run data get storage {namespace}:main {this_path}
function {namespace}:get_heap
''', f'{namespace}:main obj[{index}]', f'function {namespace}:set_heap\n'
        else:
            return '', f'{namespace}:main static[{list(classes_info).index(klass)}][{classes_info[klass]["static"].index(variable)}]', ''
    else:
        raise NotImplementedError(variable)


def assign(to_assign, assignee, args, klass, file, cur_m, lvals):
    out_code, out_temp, out_type = to_assign
    file.write(out_code)
    a = ''
    # print(assignee)
    if cname(assignee) == 'str':
        t = namedtuple('t', ['var'])
        assignee = t(assignee)
    if cname(assignee) == 'INDEX':
        a, out_temp2, out_type2 = evaluate_final(a, cur_m, klass, assignee.arr, args, [], lvals)
        out_code, out_temp3, out_type3 = generate_expression(assignee.i, args, klass, cur_m, lvals)
        a += out_code
        assert out_type2.endswith('[]'), 'Can\'t index non-array object'
        l3_type = out_type2[:-2]
        # print(l3_type, out_type2)
        assert l3_type == out_type, 'Unmatching types'
        temp_temp = get_temp()
        temp_temp2 = get_temp()
        a += f'scoreboard players operation $index {OBJECTIVE} = {out_temp2} {OBJECTIVE}\n'
        a += f'function {namespace}:get_heap\n'
        a += f'execute store result score $index {OBJECTIVE} run data get storage {namespace}:main obj\n'
        a += f'scoreboard players remove $index {OBJECTIVE} 1\n'
        a += f'scoreboard players set {temp_temp} {OBJECTIVE} 128\n'
        a += f'scoreboard players operation $index {OBJECTIVE} *= {temp_temp} {OBJECTIVE}\n'
        a += f'scoreboard players operation $index {OBJECTIVE} -= {out_temp3} {OBJECTIVE}\n'
        a += f'execute store result score {temp_temp2} {OBJECTIVE} run scoreboard players remove $index {OBJECTIVE} 1\n'
        a += f'function {namespace}:array_access\n'
        a += f'scoreboard players operation $index {OBJECTIVE} %= {temp_temp} {OBJECTIVE}\n'
        a += f'scoreboard players operation $arr {OBJECTIVE} = {out_temp} {OBJECTIVE}\n'
        a += f'function {namespace}:subarray_set\n'
        a += f'scoreboard players operation $index {OBJECTIVE} = {temp_temp2} {OBJECTIVE}\n'
        a += f'function {namespace}:array_set\n'
        a += f'scoreboard players operation $index {OBJECTIVE} = {out_temp2} {OBJECTIVE}\n'
        a += f'function {namespace}:set_heap\n'
    elif assignee.var != '':
        name = list(filter(lambda x: x.name == assignee.var, args))
        assert len(name) != 0, f'Undefined variable {assignee.var}'
        arg_stuff = name[0]
        if arg_stuff.type == 'int':
            assert out_type == 'int', f'Invalid assignment, can\'t assign int to {arg_stuff.type}'
        else:
            assert arg_stuff.type in classes_info[out_type][
                'inherits'], f'Invalid assignment, can\'t assign {out_type} to {arg_stuff.type}'
        setup, path, save = get_path(arg_stuff, args, klass, 'functions[-1][0]')
        a += setup
        a += f'execute store result storage {path} int 1 run scoreboard players get {out_temp} {OBJECTIVE}\n'
        a += save
        # a += f'execute store result score {temp_val} {OBJECTIVE} run data get storage {path}\n'
    else:
        var1 = assignee.props[0]
        callee = assignee.props[1:]
        org = ''
        if cname(assignee.expr) == 'str':
            org = assignee.expr
            if org in classes_info:
                name = list(filter(lambda x: x.name == var1, classes_info[org]['static']))
                assert len(name) != 0, f'Undefined variable {var1}'
                arg_stuff = name[0]
                setup, path, save = get_path(arg_stuff, args, org, 'functions[-1][0]')
                l3_type = arg_stuff.type
                a += setup
                # raise NotImplementedError('static things are hard')
            else:
                name = list(filter(lambda x: x.name == org, args))
                assert len(name) != 0, f'Undefined variable {org}'
                arg_stuff = name[0]
                setup, path, save = get_path(arg_stuff, args, klass, 'functions[-1][0]')
                l3_type = arg_stuff.type
                a += setup
                callee = [var1] + callee
        else:
            print(code[assignee.expr._tx_position:assignee.expr._tx_position_end])
            raise RuntimeError('Can\'t assign to expression')
        for caller in callee:
            # print(path, path.split()[-1])
            args = classes_info[l3_type]['field']
            name = list(filter(lambda x: x.name == caller, args))
            assert len(name) != 0, f'Can\'t find property {caller}'
            arg_stuff = name[0]
            setup, path, save = get_path(arg_stuff, args, l3_type, path.split()[-1])
            l3_type = arg_stuff.type
            a += setup
        # print(path)
        if path.startswith('$'):
            raise RuntimeError('Invalid assignment target')
        else:
            a += f'execute store result storage {path} int 1 run scoreboard players get {out_temp} {OBJECTIVE}\n'
        a += save
    file.write(a)
    # if cname(assignee) == 'PROP':
    #     if assignee.var != '':
    #         assignee = assignee.var
    #     else:
    #         raise NotImplementedError('not local variables are not supported')
    # if cname(assignee) == 'str':
    #     name = list(filter(lambda x: x.name == assignee, args))
    #     assert len(name) != 0, f'Undefined variable {assignee}'
    #     arg_stuff = name[0]
    #     assert out_type == arg_stuff.type, 'Assign type isn\'t equals to expression type'
    #     setup, path, save = get_path(arg_stuff, args, klass, 'functions[-1][0]')
    #     file.write(setup)
    #     file.write(
    #         f'execute store result storage {path} int 1 run scoreboard players get {out_temp} {OBJECTIVE}\n')
    #     file.write(save)


def generate(cb, arguments, file, name, klass, method, lvals):
    arguments = arguments[:]
    props = []
    props += classes_info[klass]['static']
    if not method.static:
        props += classes_info[klass]['field']
    b = ''
    for ins in cb.instructions:
        if cname(ins) == 'Assignment':
            if 'type' in dir(ins) and ins.type is not None and ins.type != '':
                assert len(
                    list(filter(lambda x: x.name == ins.name,
                                arguments + props))) == 0, f'Variable {ins.name} defined twice'
                arguments.append(Argument(ins.type, ins.name))
                file.write(
                    f'data modify storage {namespace}:main functions[-1] append value 0\n')
                b += f'data remove storage {namespace}:main functions[-1][-1]\n'
    for i, inst in enumerate(cb.instructions):
        file2 = open((name + f'_line{i}.mcfunction').lower(), 'w')
        if cname(inst) == 'Assignment':
            assignee = inst.name
            expr = inst.value
            assign(generate_expression(expr, arguments + props, klass, method, lvals), assignee, arguments + props,
                   klass, file2,
                   method, lvals)
        elif cname(inst) == 'Expression':
            file2.write(generate_expression(inst, arguments + props, klass, method, lvals)[0])
        elif cname(inst) == 'Return':
            if inst.rval is None:
                assert method.returnType in ['void', 'init'], 'Not void-methods must return something'
            else:
                out_code, out_temp, out_type = generate_expression(inst.rval, arguments + props, klass, method, lvals)
                assert out_type == method.returnType, 'Invalid return type'
                file2.write(out_code)
                file2.write(f'scoreboard players operation $r {OBJECTIVE} = {out_temp} {OBJECTIVE}\n')
            file2.write(f'scoreboard players set $returned {OBJECTIVE} 1')
        elif cname(inst) == 'MCCommand':
            c = inst.command
            r = r'\${([^}]*?)}'
            r2 = r'`([^}]*?)`'

            def eval_rep(match):
                match = match.group(1)
                return str(eval(match, lvals))

            # print(c, r2)
            c = re.sub(r2, eval_rep, c)

            def replacement(match):
                match = match.group(1)
                exprr = mm.model_from_str('class a{int a(){' + match + ';}}')
                exprr = exprr.classes[0].things[0].cb.instructions[0]
                o_c, o_temp, o_type = generate_expression(exprr, arguments + props, klass, method, lvals)
                if o_type != 'int':
                    logging.warning('The value of not-integers will be a pointer to the heap.')
                file2.write(o_c)
                return o_temp

            file2.write(re.sub(r, replacement, c) + '\n')

        elif cname(inst) == 'If':
            print('start', arguments + props)
            out_code, out_temp, out_type = generate_expression(inst.exp, arguments + props, klass, method, lvals)
            print('end')
            file2.write(out_code)
            assert out_type == 'int', 'Can only check If on an integer'
            file2.write(
                f'execute unless score {out_temp} {OBJECTIVE} matches 0 run function {namespace}:{name}_if{i}')
            name3 = name + f'_if{i}'
            print('if:', name3)
            file3 = open((name + f'_if{i}.mcfunction').lower(), 'w')
            generate(inst.code, arguments, file3, name3, klass, method, lvals)
            file3.close()
        elif cname(inst) == 'While':
            out_code, out_temp, out_type = generate_expression(inst.exp, arguments + props, klass, method, lvals)
            file2.write(out_code)
            assert out_type == 'int', 'Can only check If on an integer'
            file2.write(
                f'execute unless score {out_temp} {OBJECTIVE} matches 0 run function {namespace}:{name}_while{i}')
            name3 = name + f'_while{i}'
            file3 = open((name + f'_while{i}.mcfunction').lower(), 'w')
            generate(inst.code, arguments, file3, name3, klass, method, lvals)
            file3.write(f'execute unless score $returned {OBJECTIVE} matches 1 run function {namespace}:{name}_line{i}')
            file3.close()
        elif cname(inst) == 'For':
            out_code, out_temp, out_type = generate_expression(inst.arr, arguments + props, klass, method, lvals)
            assert out_type.endswith('[]'), 'Can only loop over a list'
            assert out_type[:-2] == inst.type, 'Unmatching types'
            file2.write(out_code)
            file2.write(f'scoreboard players operation $index {OBJECTIVE} = {out_temp} {OBJECTIVE}\n')
            file2.write(f'function {namespace}:get_heap\n')
            file2.write(f'data modify storage {namespace}:main loops append from storage {namespace}:main obj\n')
            file2.write(f'data modify storage {namespace}:main functions[-1] append value 0\n')
            file2.write(
                f'execute store result score {out_temp} {OBJECTIVE} run data get storage {namespace}:main loops[-1][0][0]\n')
            file2.write(
                f'execute unless score {out_temp} {OBJECTIVE} matches 0 run function {namespace}:{name}_for{i}\n')
            file2.write(
                f'data remove storage {namespace}:main loops[-1]\n')
            file2.write(
                f'data remove storage {namespace}:main functions[-1][-1]\n')
            name3 = name + f'_for{i}'
            file3 = open((name + f'_for{i}.mcfunction').lower(), 'w')
            thingi = len(arguments)
            file3.write(
                f'data modify storage {namespace}:main functions[-1][{thingi}] set from storage {namespace}:main loops[-1][-1][-1]\n')
            arguments = arguments + [Argument(type=inst.type, name=inst.name)]
            generate(inst.code, arguments, file3, name3, klass, method, lvals)
            file3.write(f'data remove storage {namespace}:main loops[-1][-1][-1]\n')
            file3.write(
                f'execute store result score $index {OBJECTIVE} run data get storage {namespace}:main loops[-1][0][0]\n')
            file3.write(
                f'execute store result storage {namespace}:main loops[-1][0][0] int 1 run scoreboard players remove $index {OBJECTIVE} 1\n')
            file3.write(
                f'execute unless data storage {namespace}:main loops[-1][-1][] run data remove storage {namespace}:main loops[-1][-1]\n')
            file3.write(
                f'execute unless score $returned {OBJECTIVE} matches 1 if score $index {OBJECTIVE} matches 1.. run function {namespace}:{name3}')
            file3.close()
            # file3.write(f'function {namespace}:{name}_line{i}')
        elif cname(inst) == 'Error':
            error_msg = json.dumps('Error: ' + inst.msg)
            file2.write(f'tellraw @a {{"color": "red", "text": {error_msg}}}\n')
            file2.write(f'scoreboard players set $error {OBJECTIVE} 1\n')
            file2.write(f'scoreboard players set $returned {OBJECTIVE} 1\n')
        elif cname(inst) == 'As':
            file2.write(
                f'execute as {inst.selector} run function {namespace}:{name}_as{i}')
            name3 = name + f'_as{i}'
            # print('if:', name3)
            file3 = open((name + f'_as{i}.mcfunction').lower(), 'w')
            generate(inst.cb, arguments, file3, name3, klass, method, lvals)
            file3.close()
        else:
            raise NotImplementedError('oops', cname(inst))
        iname = name + f'_line{i}'
        file.write(f'execute unless score $returned {OBJECTIVE} matches 1 run function {namespace}:{iname}\n')
        file2.close()
    file.write(b)


def main():
    os.chdir(dirname)
    for a in classes_info:
        for m in classes_info[a]['methods']:
            method = classes_info[a]['methods'][m]
            if len(method.template) == 0:
                f = open(f'method_{a}_{m}.mcfunction'.lower(), 'w')
                args = method.arguments[:]
                # print('args: ', args)
                if not method.static:
                    args = [Argument(a, 'this')] + args
                    # print(a, m)
                    # print(method.author)
                    if list(filter(lambda x: classes_info[x]['methods'][m].author == x, classes_info[a]['iby'])):
                        print(list(filter(lambda x: classes_info[x]['methods'][m].author == x, classes_info[a]['iby'])))
                        print(classes_info[a]['iby'])
                        f.write(
                            f'execute store result score $index {OBJECTIVE} run data get storage {namespace}:main functions[-1][0]\n')
                        f.write(
                            f'function {namespace}:get_heap\n')
                        f.write(
                            f'data modify storage {namespace}:main i_stack append from storage {namespace}:main obj[-1]\n')
                        for c in filter(lambda x: classes_info[x]['methods'][m].author == x, classes_info[a]['iby']):
                            f.write(
                                f'execute store result score $type {OBJECTIVE} run data get storage {namespace}:main i_stack[-1]\n')
                            f.write(
                                f'execute if score $type {OBJECTIVE} matches {get_index(c)} run function {namespace}:method_{c.lower()}_{m.lower()}\n')
                        f.write(
                            f'execute store result score $type {OBJECTIVE} run data get storage {namespace}:main i_stack[-1]\n')
                        f.write(
                            f'execute unless score $type {OBJECTIVE} matches {get_index(a)} run scoreboard players set $returned {OBJECTIVE} 1\n')
                        f.write(
                            f'data remove storage {namespace}:main i_stack[-1]\n')
                generate(method.cb, args, f, f'method_{a}_{m}'.lower(), a, method, {'json': json})
                f.write(
                    f'execute unless score $error {OBJECTIVE} matches 1 run scoreboard players set $returned {OBJECTIVE} 0')
                f.close()
    from gen_heap import generate_heap
    generate_heap(namespace, OBJECTIVE)


if __name__ == '__main__':
    main()

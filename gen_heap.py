import re, random
from tree_gen import generate

SIZE = 255


def generate_heap(namespace, objective):
    # return
    # rval = str(random.randint(10 ** 5, 10 ** 6))
    with open('load_heap.mcfunction', 'w') as file:
        file.write(re.sub(r'namespace', namespace, open('../heap_load').read()))
    with open('create_array.mcfunction', 'w') as file:
        file.write(re.sub(r'objectivee', objective, re.sub(r'namespace', namespace, open('../create_array').read())))

    # with open('get_heap.mcfunction', 'w') as file:  # function tree, get namespace:main heap[i]
    def callback(pos):
        return f'data modify storage {namespace}:main obj set from storage {namespace}:main heap[{{selected:1}}].value[{pos}]'

    generate(0, SIZE, namespace, callback, objective, 'get_selected', score='$t')

    def callback(pos):
        return f'data modify storage {namespace}:main heap[{pos}].selected set value 1'

    f = open('get_heap.mcfunction', 'w')
    f.write(f'data modify storage {namespace}:main heap[].selected set value 0\n')
    f.close()
    generate(0, SIZE, namespace, callback, objective, 'get_heap', 256)
    f = open('get_heap.mcfunction', 'a')
    f.write(f'\nscoreboard players operation $t {objective} = $index {objective}\nscoreboard players set $i {objective} '
            f'256\nscoreboard players operation $t {objective} %= $i {objective}\nfunction {namespace}:get_selected')
    f.close()

    # with open('set_heap.mcfunction', 'w') as file:  # function tree, set namespace:main heap[i]
    def callback(pos):
        return f'data modify storage {namespace}:main heap[{{selected:1}}].value[{pos}] set from storage {namespace}:main obj'

    generate(0, SIZE, namespace, callback, objective, 'set_selected', score='$t')

    def callback(pos):
        return f'data modify storage {namespace}:main heap[{pos}].selected set value 1'

    f = open('set_heap.mcfunction', 'w')
    f.write(f'data modify storage {namespace}:main heap[].selected set value 0\n')
    f.close()
    generate(0, SIZE, namespace, callback, objective, 'set_heap', 256)
    f = open('set_heap.mcfunction', 'a')
    f.write(f'\nscoreboard players operation $t {objective} = $index {objective}\nscoreboard players set $i {objective} '
            f'256\nscoreboard players operation $t {objective} %= $i {objective}\nfunction {namespace}:set_selected')
    f.close()
    # def callback(pos):
    #     return f'data modify storage {namespace}:main heap[{pos}] set from storage {namespace}:main obj'
    #
    # generate(0, SIZE, namespace, callback, objective, 'set_heap')

    def callback(pos):
        return f'execute store result score $arr {objective} run data get storage {namespace}:main temp[{pos}]'

    generate(0, 127, namespace, callback, objective, 'subarray_access')

    def callback(pos):
        return f'execute store result storage {namespace}:main temp[{pos}] int 1 run scoreboard players get $arr {objective}'

    generate(0, 127, namespace, callback, objective, 'subarray_set')

    def callback(pos):
        return f'data modify storage {namespace}:main temp set from storage {namespace}:main obj[{pos + 1}]\n'

    generate(0, 128, namespace, callback, objective, 'array_access', 128)

    def callback(pos):
        return f'data modify storage {namespace}:main obj[{pos + 1}] set from storage {namespace}:main temp\n'

    generate(0, 128, namespace, callback, objective, 'array_set', 128)

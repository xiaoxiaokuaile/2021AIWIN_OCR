# -*- coding=utf-8 -*-


def get_lines():
    return len(open('recognizer/tools/chars.txt', 'r',encoding='utf-8').readlines())


num_class = get_lines() + 1

GPU_MEMORY_FRACTION = 0.8

TF_ALLOW_GROWTH = True

is_debug = True



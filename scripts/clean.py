# -*- coding: utf-8 -*-
import argparse
from functools import partial
import multiprocessing
from typing import Collection, Callable
nb_cores = multiprocessing.cpu_count()

import pythainlp
from pythainlp.util import normalize as th_normalize

from thai2transformers.preprocess import (
    fix_html,
    rm_useless_spaces,
    replace_spaces,
)

def _process_transformers(
    text: str,
    pre_rules: Collection[Callable] = [
        fix_html,
        rm_useless_spaces,
        replace_spaces,
    ]
) -> str:
    for rule in pre_rules:
        text = rule(text)
    return text

SPIECE = '▁'

def process_one(line: str, space_token: str, 
                do_fix_html: bool,
                do_rm_useless_spaces: bool,
                do_th_normalize: bool):

    line = line.replace(u'\x0a', u' ')
    line = line.replace(SPIECE, ' ')
    if do_th_normalize:
        line = th_normalize(line)
    
    pre_rules = []
    
    if do_fix_html:
        pre_rules.append(fix_html)
    if do_rm_useless_spaces:
        pre_rules.append(rm_useless_spaces)

    pre_rules.append(partial(replace_spaces, space_token=space_token))

    return _process_transformers(text=line, pre_rules=pre_rules)


def main(args):

    input_path = args.input_path
    output_path = args.output_path
    do_fix_html = args.do_fix_html
    do_rm_useless_spaces = args.do_rm_useless_spaces
    do_th_normalize = args.do_th_normalize
    space_token = args.space_token

    print(f'\n[INFO]: pythialnlp version: {pythainlp.__version__}\n')

    print(f'Begin loading files from {args.input_path}')
    
    print(f'\nReading file from `{input_path}`.')
    lines = open(input_path, 'r', encoding="utf8", errors='ignore').readlines()
    lines = list(map(lambda x: str(x).strip(), lines))
    lines = list(filter(lambda x: x != '\n', lines))
    print(f'\n  Total number of lines (before segment splitting): {len(lines)}')

    print('\nBegin text preprocessing.')
    print(f' fix_html = {do_fix_html}')
    print(f' rm_useless_spaces = {do_rm_useless_spaces}')
    print(f' th_normalize = {do_th_normalize}')
    print(f' space_token = {space_token}')
    print('...')

    with multiprocessing.Pool(nb_cores) as pool:

        processed_lines = pool.map(partial(process_one,
                                            space_token=space_token,
                                            do_fix_html=do_fix_html,
                                            do_rm_useless_spaces=do_rm_useless_spaces,
                                            do_th_normalize=do_th_normalize 
                                  ), lines)

    print('\nDone.')

    print(f'\nBegin writing result to the file: `{output_path}`')
    with open(output_path, 'w') as f:
        f.write('\n'.join(processed_lines))

    print(f'Done.\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help="Path to the file storing extracted text.")
    parser.add_argument('output_path', help="Output path to write the result file")
    parser.add_argument('--do_fix_html', default=True, action='store_true')
    parser.add_argument('--do_rm_useless_spaces', default=True, action='store_true')
    parser.add_argument('--do_th_normalize', default=True, action='store_true')
    parser.add_argument('--space_token', type=str, default=SPIECE)
    
    args = parser.parse_args()
    
    main(args)
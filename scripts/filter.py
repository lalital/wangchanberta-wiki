# -*- coding: utf-8 -*-
import argparse
from functools import partial
import pythainlp
from pythainlp.tokenize import word_tokenize
import multiprocessing
nb_cores = multiprocessing.cpu_count()

SPIECE = '▁'

def filter_one(line:str, min_n_tokens:int, max_n_tokens:int):
    n_tokens = len(word_tokenize(line.replace(SPIECE, ' '), engine='newmm', keep_whitespace=True))
    if n_tokens < min_n_tokens or min_n_tokens > max_n_tokens:
        return (False, line)
    
    return (True, line)

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    min_n_tokens = args.min_n_tokens
    max_n_tokens = args.max_n_tokens

    print(f'\n[INFO]: pythialnlp version: {pythainlp.__version__}\n')

    print(f'\nReading file from `{input_path}`.')
    lines = open(input_path, 'r', encoding="utf8", errors='ignore').readlines()
    lines = list(map(lambda x: str(x).strip(), lines))
    lines = list(filter(lambda x: x != '\n', lines))
  
    print(f'\n  Total number of lines (before filtering): {len(lines)}')

    with multiprocessing.Pool(nb_cores) as pool:

        _filtered_lines = pool.map(partial(filter_one,
                                          min_n_tokens=min_n_tokens,
                                          max_n_tokens=max_n_tokens),
                                  lines)
        filtered_lines = [ line for keep, line in _filtered_lines if keep ]

    print('\nDone.')

    print(f'\n  Total number of lines (agter filtering): {len(filtered_lines)}')


    print(f'\nBegin writing result to the file: `{output_path}`')

    with open(output_path, 'w') as f:
        f.write('\n'.join(filtered_lines))

    print(f'Done.\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help="Path to the file storing extracted text.")
    parser.add_argument('output_path', help="Output path to write the result file")
    parser.add_argument('--min_n_tokens', default=5, type=int)
    parser.add_argument('--max_n_tokens', default=500, type=int)
    
    
    args = parser.parse_args()
    
    main(args)
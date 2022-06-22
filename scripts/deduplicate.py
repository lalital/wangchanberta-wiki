# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import pandas as pd


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    
    print(f'\nReading file from `{input_path}`.')
    lines = open(input_path, 'r').readlines()
    lines = list(map(lambda x: str(x).strip(), lines))
    lines = list(filter(lambda x: x != '\n', lines))
    
    print(f'\nNumber of lines (filtered empty line): {len(lines)}')
    print(f'Done.')

    print('\nLoading text segments into Pandas DataFrame')
    df = pd.DataFrame().from_dict({'text': lines})
    print('Done.')
    
    print('\nPerform drop duplication (keep first).')
    df_dropdup = df.drop_duplicates(subset=['text'], keep='first')
    
    print(f'\nNumber of lines (after drop duplicated): {len(df_dropdup.text)}')
    print('Done.')

    print(f'\nWrite resulting file to {output_path}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(df_dropdup['text'].tolist()))
    print('Done.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)

    args = parser.parse_args()
    main(args)
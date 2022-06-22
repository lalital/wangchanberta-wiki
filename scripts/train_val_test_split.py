import argparse
import os
import math
import json
import random
from functools import partial
import multiprocessing
from typing import Collection, Callable, List, Dict, Union

from fsplit.filesplit import Filesplit
from tqdm import tqdm
from sklearn.model_selection import train_test_split

fs = Filesplit()

def get_file_size(f):
    """Get file size in GB from file pointer."""
    old_file_position = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(old_file_position, os.SEEK_SET)
    f.close()
    return float(size) / ( 2 ** ( 10 * 3 ))


def get_number_of_lines(f):
    """Get file size in GB from file pointer."""
    n_lines = 0
    for _ in f:
        n_lines += 1
    f.close()
    return n_lines


def load_data_source(input_path):
    lines = open(input_path, 'r', encoding="utf8", errors='ignore').readlines()
    lines = list(map(lambda x: str(x).lstrip(), lines))
    lines = list(filter(lambda x: x != '\n', lines))
    return lines


def compute_dist(mapping: Dict[str, float]):
    prob_dist = {}
    summation = float(sum(map(lambda x: x, list(mapping.values()))))
    for key, value in mapping.items():
        
        prob_dist[key] = value / summation
                      
    return prob_dist
    

def compute_smoothed_dist(dist, temperature=1.0):
    
    smooth_prob_dist = {}
    summation = float(sum(map(lambda x: math.pow(x, temperature),
                                  list(dist.values()))))

    for key, value in dist.items():
        smooth_prob_dist[key] = math.pow(value, temperature) / summation
                   
    return smooth_prob_dist
    

def main(args):

    print(f'\n\nINFO: Load data source config file from `{args.data_source_config_file_path}`\n')

    data_source_dict = json.load(open(args.data_source_config_file_path, 'r'))
    for name, path_postfix in data_source_dict.items():
        data_source_dict[name] = os.path.join(args.input_dir_prefix, path_postfix)
    print(f'\nINFO: Get file size and number of lines.')

    data_source_size_mapping = { name: get_file_size(open(path, 'r')) for name, path in data_source_dict.items() }
    data_source_lines_mapping = { name: get_number_of_lines(open(path, 'r')) for name, path in data_source_dict.items() }

    print('\nINFO: Data source statistics')
    for name, size in data_source_size_mapping.items():
        print(f'Source name: {name}')
        print(f' - path: {data_source_dict[name]}')
        print(f' - number of lines {data_source_lines_mapping[name]}')
        print(f' - data size: {size:.4f} GB')
    
    print(f'\nINFO: Total size of data: {sum(data_source_size_mapping.values())}')

    dist_by_size = compute_dist(data_source_size_mapping)
    print('\nINFO: Data source distribution by size\n')
    print(dist_by_size)
    print(f'\n\nINFO: Data source distribution after adjusted with temperature (T={args.temperature})')
    smoothed_dist = compute_smoothed_dist(dist_by_size, temperature=args.temperature)
    print(smoothed_dist)
    print('\n')

    print('--'*20)
    print(f'\nBegin to sample val/test sentences (number_of_val_sentneces={args.number_of_val_sentneces})')

    total_n_sampled = 0
    for source, prob in smoothed_dist.items():
        
        print('source:', source)
        print(' - prob:', prob)
        target_n_sampled = int(prob * args.number_of_val_sentneces)
        print(f' - # of sentences (target): {target_n_sampled:,}')
        
        data_source_n_lines = data_source_lines_mapping[source]    
        if target_n_sampled >= data_source_n_lines:
            real_n_sampled = data_source_n_lines
        else:
            real_n_sampled = target_n_sampled
        print(f' - total # of sentences: {data_source_n_lines:,}')
        print(f' - # of sentences (actual): {real_n_sampled:,}')
        
        print('')
        
        total_n_sampled += real_n_sampled

    print(f'Total: {total_n_sampled:,}')

    print('--'*20)

    print('\nLoad all files into memory and sample val/test sentences')

    sampled_lines_by_source = {}
    sampled_indices_by_source = {}
    lines_by_source = {}
    for source, input_path in data_source_dict.items():
        prob = smoothed_dist[source]
        print('source:', source)
        print(' - smoothed prob:', prob)
        
        number_of_samples = int(prob * args.number_of_val_sentneces)
        print(f' - # of sampled sentences (target): {number_of_samples:,}')
        lines = load_data_source(input_path)
        lines_by_source[source] = lines
        print(f' - total # of sentences: {len(lines):,}')


        if number_of_samples >= len(lines):
            raise "Error: number_of_samples should not greater than total number of samples"
        else:
            random.seed(args.seed)
            sampled_lines_and_indices = random.sample(list(enumerate(lines)), k=number_of_samples)
            sampled_lines_by_source[source] = list(map(lambda x: x[1], sampled_lines_and_indices))
            sampled_indices_by_source[source] = list(map(lambda x: x[0], sampled_lines_and_indices))
            print(f' - # of sampled sentences (target): {number_of_samples:,}')
            print('')
        
        del lines

    sampled_lines_by_source_with_split = {}
    for source, sampled_lines in sampled_lines_by_source.items():
        sampled_lines_by_source_with_split[source] = {}
        validation_lines, test_lines = train_test_split(sampled_lines,
                                                        test_size=0.5,
                                                        random_state=2021)
        
        sampled_lines_by_source_with_split[source]['validation'] = validation_lines
        sampled_lines_by_source_with_split[source]['test'] = test_lines


    print('INFO: Select training set')

    train_set_by_source = {}

    for source, indices in sampled_indices_by_source.items():
        print(f'source: {source}\n')
        train_set_by_source[source] = []
        lines = lines_by_source[source]

        print(f' - Total # of sentences: {len(lines):,}')    
        
        sampled_lines = sampled_lines_by_source[source]
        
        print(f' - Number of sentneces in training set: {len(lines) - len(sampled_lines):,}')    
        print(f' - Number of sentneces in validation and testset: {len(sampled_lines):,}')    
        sorted_indices = sorted(indices)
        for idx, line in tqdm(enumerate(lines), total=len(lines), position=0, leave=True):
                
            if len(sorted_indices) != 0 and idx == sorted_indices[0]:
                sorted_indices.pop(0)
                continue
            train_set_by_source[source].append(line)
        print('Done.\n')

  
    print('\nINFO: Data spliting done.\n')
    print('--'*20)
    print(f'\nINFO: Begin writing file to `{args.output_dir}`\n')
    split_names = ['train', 'validation', 'test']
    
    for split_name in split_names:
        base_output_dir =  os.path.join(args.output_dir, split_name)
        os.makedirs(base_output_dir, exist_ok=True)

        output_path = os.path.join(base_output_dir, f'{split_name}.txt')
        print(f'\nBegin writing {split_name} set to the file: `{output_path}`')
        with open(output_path, 'w', encoding='utf-8') as f:
            for source, sampled_lines in train_set_by_source.items():
                f.writelines(sampled_lines)

        if args.split_large_file:
            file_size = get_file_size(open(output_path, 'r'))
            def split_cb(f, s):
                print("Perform large file spliting: {0}, size: {1}".format(f, s))

            if file_size > args.max_file_size:
                fs.split(file=output_path, split_size=float(args.max_file_size) * ( 2 ** ( 10 * 3 )),
                        newline=True,
                        encoding='utf-8',
                        output_dir=base_output_dir, callback=split_cb)
                print(f'INFO: Large file spliting completed. \nRemove the following file {output_path}\n')
                os.remove(output_path)
        print('\n')
    print('INFO: Writing file done.')
    print(f'--'*20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_source_config_file_path', type=str, help='Path to the JSON file storeing path of data source')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--input_dir_prefix', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--number_of_val_sentneces',  type=int, default=500_000)

    parser.add_argument('--split_large_file', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--max_file_size', type=int, default=50, help='Maximum file size in GB')

    args = parser.parse_args()

    main(args)
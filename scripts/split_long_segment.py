# -*- coding: utf-8 -*-
import argparse
import re
from typing import List

import pythainlp
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize as en_sent_tokenize
from pythainlp.tokenize import word_tokenize, sent_tokenize as th_sent_tokenize


pattern_th = r'[ก-๙]'


def _group_splitted_segments(splitted_segments: List[str], max_group_seq_len) -> List[str]:
    splitted_segments_n_toks = [ len(word_tokenize(sent)) for sent in splitted_segments ]

    groupped_sents = []

    seq_len_counter = 0
    temp_groupped_sent = ''
    for i, sent in enumerate(splitted_segments):

        if seq_len_counter + splitted_segments_n_toks[i] >= max_group_seq_len:

            groupped_sents.append(temp_groupped_sent)
            seq_len_counter = 0
            temp_groupped_sent = sent
        else:
            temp_groupped_sent += sent
            seq_len_counter += splitted_segments_n_toks[i]

        if i == len(splitted_segments) - 1:
            groupped_sents.append(temp_groupped_sent)
            
    return groupped_sents

def _split_long_segment(segments: List[str],
                        max_seq_len: int,
                        max_group_seq_len:int ) -> List[str]:

    new_segments = []
    for _, segment in enumerate(segments):
        if len(word_tokenize(segment)) > max_seq_len:
            if re.search(pattern_th, segment) == None:
                # For English segment
                new_segments.extend(en_sent_tokenize(segment))
            else:
                # For Thao segment
                splitted_segments = th_sent_tokenize(segment)
                groupped_splitted_segments = _group_splitted_segments(splitted_segments,
                                                                      max_group_seq_len=max_group_seq_len)
                new_segments.extend(groupped_splitted_segments)
        else:
            new_segments.append(segment)
    
    return new_segments

def split_long_segment(lines: List[str], max_seq_len:int = 350,
                      max_group_seq_len:int = 300) -> List[str]:
    """Split line with sequence length gretaer than `max_seq_len`"""
            
    return _split_long_segment(lines, max_seq_len, max_group_seq_len)

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    max_seq_len = args.max_seq_len
    max_group_seq_len = args.max_group_seq_len
    
    print(f'\n[INFO]: pythialnlp version: {pythainlp.__version__}\n')

    print(f'Begin loading files from {args.input_path}')
    
    print(f'\nReading file from `{input_path}`.')
    lines = open(input_path, 'r').readlines()
    lines = list(map(lambda x: str(x).strip(), lines))
    lines = list(filter(lambda x: x != '\n', lines))
    print(f'\n  Total number of lines (before segment splitting): {len(lines)}')
    print(f'\nBegin spliting long segment.')
    print(f'  - max_seq_len = {max_seq_len}')
    print(f'  - max_group_seq_len = {max_group_seq_len}\n')

     
    segmented_lines = split_long_segment(lines,
                                         max_seq_len=max_seq_len,
                                        max_group_seq_len=max_group_seq_len)
                                    
    print(f'\n  Total number of lines (after segment splitting): {len(segmented_lines)}')
    print(f'Done.\n')

    print(f'Begin writing result to the file: `{output_path}`')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(segmented_lines))

    print(f'Done.\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help="Path to the file storing extracted text.")
    parser.add_argument('output_path', help="Output path to write the result file")
    parser.add_argument('--max_seq_len', type=int, default=350)
    parser.add_argument('--max_group_seq_len', type=int, default=300)
    
    args = parser.parse_args()
    
    main(args)

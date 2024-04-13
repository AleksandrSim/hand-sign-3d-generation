import os 
import json

import numpy as np

from src.process_data.utils import char_index_map


class CombineData:
    def __init__(self, letter_paths, bad_complication_json, spaces_json ):
        self.letter_paths = letter_paths
        self.bad_complication_json = bad_complication_json
        self.spaces_json = spaces_json
        self.final_dic = {}

    def iterate_letters(self):
        letter_folder = os.listdir(self.letter_paths)
        letter_folder = [i for i in letter_folder if i.endswith('json')]

        print(letter_folder)
        for letter in letter_folder:
            self.process_letter(letter)

    def process_letter(self, letter):
        with open(os.path.join(self.letter_paths, letter), 'r') as f:
            data = json.load(f)


            data = {i:k for i,k in sorted(data.items(), key = lambda x:int(x[0]))}

            prev_key = None
            for key in data.keys():

                if prev_key:
                    self.final_dic[data[prev_key] +'_' + data[key] ] = \
                                    [int(key) - int(prev_key), prev_key , letter ]
                    
                prev_key = key

    def add_stop_sequences(self):
        with open(self.spaces_json, 'r') as f:
            data = json.load(f)

            data = {i:k for i,k in sorted(data.items(), key = lambda x:int(x[0]))}
            keys = list(data.keys())

            for current_key in range(1, len(keys)-1, 2):
                prev_key = current_key -1
                self.final_dic['prob' + '_' + data[keys[prev_key]]]  = [int(keys[current_key]) - int(keys[prev_key]), keys[prev_key],
                                                                                       os.path.basename(self.spaces_json)]
    def add_problematic(self):
        with open(self.bad_complication_json, 'r') as f:
            data = json.load(f)

            data = {i:k for i,k in sorted(data.items(), key = lambda x:int(x[0]))}
            keys = list(data.keys())

            for current_key in range(1, len(keys)-1, 2):
                prev_key = current_key -1
                self.final_dic[data[keys[prev_key]] +'_' + data[keys[current_key]]] = [int(keys[current_key]) - int(keys[prev_key]), keys[prev_key],
                                                                                       os.path.basename(self.bad_complication_json)]

    def save_json(self):
        # Construct the full file path in the parent directory
        file_path = os.path.join(os.path.dirname(self.letter_paths), 'adjust_all_eng.json')
        # Open the file and write the JSON data
        with open(file_path, 'w') as f:
            json.dump(self.final_dic, f)
    

if __name__ == '__main__':
    combiner = CombineData('/Users/aleksandrsimonyan/Desktop/complete_sequence/english_full/json', 
                           '/Users/aleksandrsimonyan/Desktop/complete_sequence/bad_complications/Combinations_V1.json', 
                           '/Users/aleksandrsimonyan/Desktop/complete_sequence/letters_spaces/Letters_with_Stops.json')
    combiner.iterate_letters()
#    combiner.add_stop_sequences()
#    combiner.add_problematic()
    combiner.save_json()



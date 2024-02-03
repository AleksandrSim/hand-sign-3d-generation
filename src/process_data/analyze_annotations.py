import os
import json


class DataAnalyze:
    def __init__(self, path, write_sorted=False):
        self.path = path
        self.files = os.listdir(self.path)
        self.files = [i for i in self.files if i.endswith('.json')]
        self.all_data = {}
        self.write_sorted = write_sorted
        self.parent_dir = os.path.dirname(self.path)  # Correct way to get the parent directory
        os.makedirs(os.path.join(self.parent_dir, 'sorted_jsons'), exist_ok=True)

    def open_and_load_json(self, json_file):  # Renamed to avoid conflict with json module
        with open(os.path.join(self.path, json_file), 'r') as f:
            print(json_file)
            data = json.load(f)
        return data

    def iterate_files(self):
        for file in self.files:
            # Skip directories
            if os.path.isdir(os.path.join(self.path, file)):
                continue

            data = self.open_and_load_json(file)
            sorted_dict = {k: data[k] for k in sorted(data, key=int)}

            if self.write_sorted:
                # Correct file path for writing; ensure directory exists
                sorted_file_path = os.path.join(self.parent_dir, 'sorted_jsons', file)
                with open(sorted_file_path, 'w') as f:
                    json.dump(sorted_dict, f, indent=4)  # Added indent for better readability
            self.calculate_difference(sorted_dict, file)

        output_path = os.path.join(self.parent_dir, 'output.json')
        self.filter_final()

        with open(output_path, 'w') as f:
            json.dump(self.all_data, f)


    def calculate_difference(self, data, file):
        keys = list(data.keys())


        for idx in range(len(keys)-1):
            if data[keys[idx]] +'_'+ data[keys[idx+1]] not in self.all_data:
                self.all_data[data[keys[idx]] +'_'+ data[keys[idx+1]]] =  [int(keys[idx+1]) - int(keys[idx]), keys[idx], file ]

    def filter_final(self, thresh= 90):
        deleted_keys = []
        filtered_data = {}

        for key in self.all_data:
            if self.all_data[key][0]<thresh:
                deleted_keys.append(key)
            else:
                filtered_data[key] = self.all_data[key]

        self.all_data = filtered_data


if __name__ == '__main__':
    analyzer = DataAnalyze(path='/Users/aleksandrsimonyan/Desktop/complete_sequence/sorted_jsons', write_sorted=False)
    analyzer.iterate_files()
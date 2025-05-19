import h5py
import sys

file_path = sys.argv[1]

with h5py.File(file_path, 'r') as f:
    print("Datasets encontrados no arquivo:")
    def print_name(name):
        print(name)
    f.visit(print_name)

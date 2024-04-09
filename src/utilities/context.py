import os
from os import path
import sys

src_dir = path.abspath(path.join(path.dirname(__file__), os.pardir))
entry_dir = src_dir.replace("/src", "")
log_dir = src_dir + "/log"
collection_dir = src_dir + "/collection"
sys.path.append(entry_dir)

print(f'src_dir: {src_dir} | entry_dir: {entry_dir} \n\n')

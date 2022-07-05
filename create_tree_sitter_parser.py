from tree_sitter import Language
import os
import sys


lib_dir = sys.argv[1]
libs = [os.path.join(lib_dir, d) for d in os.listdir(lib_dir)]

Language.build_library(
    # Store the library in the `build` directory
    'parser/languages.so',
    libs,
)

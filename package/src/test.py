import pytest as py
from .get_data import *

def test_validation_yaml():
    with py.raises(FileNotFoundError):
        read_params(file_path="source/data/non_existing_file.yaml")

    with py.raises(yaml.scanner.ScannerError):
        # only show the first error
        read_params(file_path="source/data/sample_invalid.yaml")

    with py.raises(yaml.parser.ParserError):
        # only show the first error
        read_params(file_path="source/data/sample_invalid.json")
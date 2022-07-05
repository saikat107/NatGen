import os
import unittest

import tree_sitter
from src import DemoTransformation


class DemoTransformationTest(unittest.TestCase):
    def setUp(self) -> None:
        sitter_lib_path = "sitter-libs"
        libs = [os.path.join(sitter_lib_path, d) for d in os.listdir(sitter_lib_path)]
        tree_sitter.Language.build_library(
            'parser/languages.so',
            libs,
        )

    def test_parsing(self):
        code = """
        class A {
            public void foo(){
                int i=0;
            }
        }
        """
        transformer = DemoTransformation(
            parser="parser/languages.so",
            language="java"
        )
        root = transformer.parse_code(code)
        self.assertTrue(isinstance(root, tree_sitter.Node))

    def test_tokens(self):
        code = """
                class A {
                    public void foo(){
                        int i=0;
                    }
                }
                """
        expected_tokens = "class A { public void foo ( ) { int i = 0 ; } }".split()
        transformer = DemoTransformation(
            parser="parser/languages.so",
            language="java"
        )
        root = transformer.parse_code(code)
        tokens, _ = transformer.get_tokens_with_node_type(code.encode(), root)
        self.assertListEqual(tokens, expected_tokens)

import re
from typing import Union, Tuple
import os

import numpy as np

from src.data_preprocessors.language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor,
)
from src.data_preprocessors.language_processors.go_processor import GoProcessor
from src.data_preprocessors.language_processors.ruby_processor import RubyProcessor
from src.data_preprocessors.language_processors.utils import extract_statement_within_size, get_tokens, \
    get_tokens_insert_before, count_nodes
from src.data_preprocessors.transformations import TransformationBase

processor_function = {
    "java": JavaAndCPPProcessor,
    "c": JavaAndCPPProcessor,
    "cpp": JavaAndCPPProcessor,
    "c_sharp": CSharpProcessor,
    "python": PythonProcessor,
    "javascript": JavascriptProcessor,
    "go": GoProcessor,
    "php": PhpProcessor,
    "ruby": RubyProcessor,
}

tokenizer_function = {
    "java": get_tokens,
    "c": get_tokens,
    "cpp": get_tokens,
    "c_sharp": get_tokens,
    "python": PythonProcessor.get_tokens,
    "javascript": JavascriptProcessor.get_tokens,
    "go": get_tokens,
    "php": PhpProcessor.get_tokens,
    "ruby": get_tokens,
}

insertion_function = {
    "java": get_tokens_insert_before,
    "c": get_tokens_insert_before,
    "cpp": get_tokens_insert_before,
    "c_sharp": get_tokens_insert_before,
    "python": PythonProcessor.get_tokens_insert_before,
    "javascript": JavascriptProcessor.get_tokens_insert_before,
    "go": get_tokens_insert_before,
    "php": PhpProcessor.get_tokens_insert_before,
    "ruby": get_tokens_insert_before,
}


class DeadCodeInserter(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(DeadCodeInserter, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        self.insertion_function = insertion_function[self.language]

    def insert_random_dead_code(self, code_string, max_node_in_statement=-1):
        root = self.parse_code(code_string)
        original_node_count = count_nodes(root)
        if max_node_in_statement == -1:
            max_node_in_statement = int(original_node_count / 2)
        if self.language == "ruby":
            statement_markers = ["assignment", "until", "call", "if", "for", "while"]
        else:
            statement_markers = None
        statements = extract_statement_within_size(
            root, max_node_in_statement, statement_markers,
            code_string=code_string, tokenizer=self.tokenizer_function,
        )
        original_code = " ".join(self.tokenizer_function(code_string, root))
        try:
            while len(statements) > 0:
                random_stmt, insert_before = np.random.choice(statements, 2)
                statements.remove(random_stmt)
                dead_coed_body = " ".join(self.tokenizer_function(code_string, random_stmt)).strip()
                dead_code_function = np.random.choice(
                    [
                        self.processor.create_dead_for_loop,
                        self.processor.create_dead_while_loop,
                        self.processor.create_dead_if
                    ]
                )
                dead_code = dead_code_function(dead_coed_body)
                modified_code = " ".join(
                    self.insertion_function(
                        code_str=code_string, root=root, insertion_code=dead_code,
                        insert_before_node=insert_before
                    )
                )
                if modified_code != original_code:
                    modified_root = self.parse_code(" ".join(modified_code))
                    return modified_root, modified_code, True
        except:
            pass
        return root, original_code, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.insert_random_dead_code(code, -1)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    }
    """
    python_code = """
    def foo(n):
        res = 0
        for i in range(0, 19, 2):
            res += i
        i = 0
        while i in range(n):
            res += i
            i += 1
        return res
    """
    c_code = """
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    """
    cs_code = """
    int foo(int n){
            int res = 0, i = 0;
            while(i < n) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    """
    js_code = """function foo(n) {
        let res = '';
        for(let i = 0; i < 10; i++){
            res += i.toString();
            res += '<br>';
        } 
        while ( i < 10 ; ) { 
            res += 'bk'; 
        }
        return res;
    }
    """
    ruby_code = """
        for i in 0..5 do
           puts "Value of local variable is #{i}"
           if false then
                puts "False printed"
                while i == 10 do
                    print i;
                end
                i = u + 8
            end
        end
        """
    go_code = """
        func main() {
            sum := 0;
            i := 0;
            for ; i < 10;  {
                sum += i;
            }
            i++;
            fmt.Println(sum);
        }
        """
    php_code = """
    <?php 
    for ($x = 0; $x <= 10; $x++) {
        echo "The number is: $x <br>";
    }
    $x = 0 ; 
    while ( $x <= 10 ) { 
        echo "The number is:  $x  <br> "; 
        $x++; 
    } 
    ?> 
    """
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
        "cs": ("c_sharp", cs_code),
        "js": ("javascript", js_code),
        "python": ("python", python_code),
        "php": ("php", php_code),
        "ruby": ("ruby", ruby_code),
        "go": ("go", go_code),
    }
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../../'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    for lang in ["c", "cpp", "java", "python", "php", "ruby", "js", "go", "cs"]:
        lang, code = input_map[lang]
        dead_code_inserter = DeadCodeInserter(
            parser_path, lang
        )
        print(lang)
        code, meta = dead_code_inserter.transform_code(code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        print(meta)
        print("=" * 150)

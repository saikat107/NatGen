import copy
import os
import re
from typing import Union, Tuple

import numpy as np

from src.data_preprocessors.language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor
)
from src.data_preprocessors.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "cpp": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c_sharp": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "python": [PythonProcessor.for_to_while_random, PythonProcessor.while_to_for_random],
    "javascript": [JavascriptProcessor.for_to_while_random, JavascriptProcessor.while_to_for_random],
    "go": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "php": [PhpProcessor.for_to_while_random, PhpProcessor.while_to_for_random],
    "ruby": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
}


class ForWhileTransformer(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,
            "c": self.get_tokens_with_node_type,
            "cpp": self.get_tokens_with_node_type,
            "c_sharp": self.get_tokens_with_node_type,
            "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
            "php": PhpProcessor.get_tokens,
            "ruby": self.get_tokens_with_node_type,
            "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(i = 0; i < n; i++) {
                int j = 0;
                if (i == 0){
                    foo(7);
                    continue;
                }
                else if (res == 9) {
                    bar(8);
                    break;
                }
                else if (n == 0){
                    tar(9);
                    return 0;
                }
                else{
                    foo();
                    bar();
                    tar();
                }
            }
            return res;
        }
    }
    """
    python_code = """
    def is_prime(n):
        if n < 2:
            return False
        for k in range(2, n - 1, 7):
            if n % k == 0:
                foo(7)
                return False
            elif k == 0:
                bar(8)
                continue
            elif n == 9:
                tar(9)
                break
            else:
                foo()
                bar()
                tar()
        return True
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
        for(i = 0; i < n; i++) {
            int j = 0;
            if (i == 0){
                foo(7);
                continue;
            }
            else if (res == 9) {
                bar(8);
                break;
            }
            else if (n == 0){
                tar(9);
                return 0;
            }
            else{
                foo();
                bar();
                tar();
            }
        }
        return res;
    }
    """
    js_code = """function foo(n) {
        let res = '';
        for(let i = 0; i < 10; i++){
            if (i == 0){
                res += i.toString();
                continue;
            }
            else if (i == 1){
                res += '<br>';
                break;
            }
            else if (i == 2){
                foo(i);
                return 0;
            }
            else{
                bar();
                tar();
            }
        } 
        return res;
    }
    """
    ruby_code = """
        for i in 0..5
           puts "Value of local variable is #{i}"
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
    function foo(){
        for ($x = 0; $x <= 10; $x++) {
            if ($x == 0){
                foo();
                break;
            }
            elseif ($x == 1){
                bar();
                continue;
            }
            elseif ($x == 2){
                tar();
                return 0;
            }
            else{
                xar();
            }
        }
        return 4;
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
    for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby", "cs"]:
        lang, code = input_map[lang]
        for_while_transformer = ForWhileTransformer(parser_path, lang)
        print(lang, end="\t")
        code, meta = for_while_transformer.transform_code(code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        print(meta["success"])
        print("=" * 150)
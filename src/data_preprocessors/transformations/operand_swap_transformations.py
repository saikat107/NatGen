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
    PhpProcessor,
    GoProcessor,
    RubyProcessor
)
from src.data_preprocessors.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.operand_swap],
    "c": [JavaAndCPPProcessor.operand_swap],
    "cpp": [JavaAndCPPProcessor.operand_swap],
    "c_sharp": [CSharpProcessor.operand_swap],
    "python": [PythonProcessor.operand_swap],
    "javascript": [JavascriptProcessor.operand_swap],
    "go": [GoProcessor.operand_swap],
    "php": [PhpProcessor.operand_swap],
    "ruby": [RubyProcessor.operand_swap],
}


class OperandSwap(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(OperandSwap, self).__init__(parser_path=parser_path, language=language)
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
            modified_code, success = function(code, self)
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
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    python_code = """
        from typing import List

        def factorize(n: int) -> List[int]:
            import math
            fact = []
            i = 2
            while i <= int(math.sqrt(n) + 1):
                if n % i == 0:
                    fact.append(i)
                    n //= i
                else:
                    i += 1
            if n > 1:
                fact.append(n)
            return fact
        """
    c_code = """
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    cs_code = """
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    js_code = """function foo(n) {
            if (time < 10) {
              greeting = "Good morning";
            } 
            else {
              greeting = "Good evening";
            }
        }
        """
    ruby_code = """
        x = 1
        if x > 2
           puts "x is greater than 2"   
        else
           puts "I can't guess the number"
        end
        """
    go_code = """
        func main() {
           /* local variable definition */
           var a int = 100;

           /* check the boolean condition */
           if( a < 20 ) {
              /* if condition is true then print the following */
              fmt.Printf("a is less than 20\n" );
           } else {
              /* if condition is false then print the following */
              fmt.Printf("a is not less than 20\n" );
           }
           fmt.Printf("value of a is : %d\n", a);
        }
        """
    php_code = """
        <?php 
        $t = date("H");
        if ($t < "10") {
          echo "Have a good morning!";
        }  else {
          echo "Have a good night!";
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
    for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby",
                 "cs"]:  # ["c", "cpp", "java", "cs", "python",
        # "php", "go", "ruby"]:
        # lang = "php"
        lang, code = input_map[lang]
        operandswap = OperandSwap(
            parser_path, lang
        )
        print(lang)
        # print("-" * 150)
        # print(code)
        # print("-" * 150)
        code, meta = operandswap.transform_code(code)
        print(meta["success"])
        print("=" * 150)

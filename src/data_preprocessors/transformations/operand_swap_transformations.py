import copy
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

int time = 20;
if (time < 18 && time>20) {
  System.out.println("Good day.");
}
else if(time==20){
  System.out.println("Good noon")
}
 else {
  System.out.println("Good evening.");
}



    """
    python_code = """
    
aa = 33
bbb = 33
if bbb > aa:
  print("b is greater than a")
elif aa == bbb:
  print("a and b are equal")
    
    
    """
    c_code = """
 
 int time = 20;
if (time < 18 && time>20) {
  System.out.println("Good day.");
}
else if(time==20){
  System.out.println("Good noon")
}
 else {
  System.out.println("Good evening.");
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


x = 1 
unless x>=2
   puts "x is less than 2"
 else
   puts "x is greater than 2"
end





        """
    go_code = """

func main() {
   /* local variable definition */
   var a int = 100
 
   /* check the boolean condition */
   if( a == 10 ) {
      /* if condition is true then print the following */
      fmt.Printf("Value of a is 10\n" )
   } else if( a == 20 ) {
      /* if else if condition is true */
      fmt.Printf("Value of a is 20\n" )
   } else if( a == 30 ) {
      /* if else if condition is true  */
      fmt.Printf("Value of a is 30\n" )
   } else {
      /* if none of the conditions is true */
      fmt.Printf("None of the values is matching\n" )
   }
   fmt.Printf("Exact value of a is: %d\n", a )
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
    for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby",
                 "cs"]:  # ["c", "cpp", "java", "cs", "python",
        # "php", "go", "ruby"]:
        # lang = "php"
        lang, code = input_map[lang]
        operandswap = OperandSwap(
            "/parser/languages.so", lang
        )
        print(lang)
        # print("-" * 150)
        # print(code)
        # print("-" * 150)
        code, meta = operandswap.transform_code(code)
        print(meta["success"])
        print("=" * 150)

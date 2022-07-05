import copy
import os
import sys

from src.transformations import (
    BlockSwap, ConfusionRemover, DeadCodeInserter, ForWhileTransformer,
    OperandSwap, NoTransformation, SemanticPreservingTransformation, SyntacticNoisingTransformation
)

if __name__ == '__main__':
    java_code = """
            int foo(int n){
                int res = 0;
                String s = "Hello";
                foo("hello");
                for(int i = 0; i < n; i++) {
                    int j = 0;
                    while (j < i){
                        res += j; 
                    }
                }
                return res;
            }
        """
    python_code = """def foo(n):
        \"\"\"
        Documentation!
        \"\"\"
        res = 0
        for i in range(0, 19, 2):
            s = "hello"
            print("hello")
            res += i
            print("world")
        i = 0
        while i in range(n):
            res += i
            i += 1
        return res
        """
    c_code = """
            int foo(int n){
                int res = 0;
                String s = "Hello";
                foo("hello");
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
                    String s = "Hello";
                    foo("hello");
                    int j = 0;
                    while (j < i){
                        if (i < 0){
                            res += j;
                        }
                        else {
                            res -= j;
                        } 
                    }
                }
                return res;
            }
        """
    js_code = """function foo(n) {
            let res = '';
            let s = "hello"
            foo ("hello");
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
            for i in 0..5 then
               puts "Value of local variable is #{i}"
            end
            """
    go_code = """
            func main() {
                sum := 0;
                i := 0;
                s := "hello";
                foo("hello");
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
            foo ("hello");
        }
        $x = 0 ; 
        while ( $x <= 10 ) { 
            echo "The number is:  $x  <br> "; 
            $x++; 
        } 
        ?> 
        """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    transformers = {
        # This is a data structure of transformer with corresponding weights. Weights should be integers
        # And interpreted as number of corresponding transformer.
        BlockSwap: 1,
        ConfusionRemover: 1,
        DeadCodeInserter: 1,
        ForWhileTransformer: 1,
        OperandSwap: 1,
        SyntacticNoisingTransformation: 1
    }
    input_map = {
        "java": (
            java_code, NoTransformation(parser_path, "java"), SemanticPreservingTransformation(
                parser_path=parser_path, language="java", transform_functions=transformers
            )
        ),
        "c": (
            c_code, NoTransformation(parser_path, "c"), SemanticPreservingTransformation(
                parser_path=parser_path, language="c", transform_functions=transformers
            )
        ),
        "cpp": (
            c_code, NoTransformation(parser_path, "cpp"), SemanticPreservingTransformation(
                parser_path=parser_path, language="cpp", transform_functions=transformers
            )
        ),
        "c_sharp": (
            cs_code, NoTransformation(parser_path, "c_sharp"), SemanticPreservingTransformation(
                parser_path=parser_path, language="c_sharp", transform_functions=transformers
            )
        ),
        "javascript": (
            js_code, NoTransformation(parser_path, "javascript"), SemanticPreservingTransformation(
                parser_path=parser_path, language="javascript", transform_functions=transformers
            )
        ),
        "python": (
            python_code, NoTransformation(parser_path, "python"), SemanticPreservingTransformation(
                parser_path=parser_path, language="python", transform_functions=transformers
            )
        ),
        "php": (
            php_code, NoTransformation(parser_path, "php"), SemanticPreservingTransformation(
                parser_path=parser_path, language="php", transform_functions=transformers
            )
        ),
        "ruby": (
            ruby_code, NoTransformation(parser_path, "ruby"), SemanticPreservingTransformation(
                parser_path=parser_path, language="ruby", transform_functions=transformers
            )
        ),
        "go": (
            go_code, NoTransformation(parser_path, "go"), SemanticPreservingTransformation(
                parser_path=parser_path, language="go", transform_functions=transformers
            )
        ),
    }

    for lang in input_map.keys():
        code, no_transform, language_transformers = copy.copy(input_map[lang])
        tokenized_code, _ = no_transform.transform_code(code)
        transformed_code, used_transformer = language_transformers.transform_code(code)
        print("Language         : ", lang)
        print("Original Code    : ", tokenized_code)
        if not used_transformer:
            print("Cannot successfully transform!")
        else:
            print("Transformed Code : ", transformed_code)
            print("Transfomation    : ", used_transformer)
        print("=" * 150)
    pass

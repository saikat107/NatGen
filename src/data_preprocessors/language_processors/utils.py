from tree_sitter import Node


def get_tokens(code_str, root):
    if isinstance(code_str, str):
        code_str = code_str.encode()
    assert isinstance(root, Node)
    tokens = []
    if root.type == "comment":
        return tokens
    if "string" in str(root.type):
        return [code_str[root.start_byte:root.end_byte].decode()]
    children = root.children
    if len(children) == 0:
        tokens.append(code_str[root.start_byte:root.end_byte].decode().strip())
    for child in children:
        tokens += get_tokens(code_str, child)
    return tokens


def get_tokens_insert_before(code_str, root, insertion_code, insert_before_node):
    if isinstance(code_str, str):
        code_str = code_str.encode()
    assert isinstance(root, Node)
    tokens = []
    if root.type == "comment":
        return tokens
    if "string" in str(root.type):
        return [code_str[root.start_byte:root.end_byte].decode()]
    if root == insert_before_node:
        tokens += insertion_code.split()
    children = root.children
    if len(children) == 0:
        tokens.append(code_str[root.start_byte:root.end_byte].decode())
    for child in children:
        tokens += get_tokens_insert_before(code_str, child, insertion_code, insert_before_node)
    return tokens


def dfs_print(root, level=0):
    for _ in range(level):
        print("\t", end="")
    print(root)
    for child in root.children:
        dfs_print(child, level + 1)


def count_nodes(root):
    num_nodes = 1
    for child in root.children:
        if child is not None:
            num_nodes += count_nodes(child)
    return num_nodes


def extract_statement_within_size(root, max_node=10, endswith=None, code_string=None, tokenizer=None):
    if endswith is None:
        endswith = ["statement"]
    statements = []
    queue = [root]
    while len(queue) > 0:
        current_node = queue[0]
        queue = queue[1:]
        node_count = count_nodes(current_node)
        if code_string is not None and tokenizer is not None:
            tokens = tokenizer(code_string, current_node)
            current_code = " ".join(tokens).strip()
        else:
            current_code = "please provide code string and tokenizer to analyze code length"
        if any(str(current_node.type).endswith(e) for e in endswith) and\
                1 < node_count < max_node and len(current_code) > 0:
                statements.append(current_node)
        for child in current_node.children:
            queue.append(child)
    return statements

knowledge_base = [
    ("animal", "has", "skin"),
    ("bird", "is-a", "animal"),
    ("bird", "can", "fly"),
    ("penguin", "is-a", "bird"),
    ("penguin", "cannot", "fly"),
    ("ostrich", "is-a", "bird"),
    ("ostrich", "cannot", "fly"),
    ("ostrich", "can", "run fast")
]

def get_inheritance_chain(node, knowledge_base):
    chain = [node]
    current = node
    while True:
        parent = next((obj for obj, rel, sub in knowledge_base if sub == current and rel == "is-a"), None)
        if not parent:
            break
        chain.append(parent)
        current = parent
    return chain

def query(node, relation, knowledge_base):
    chain = get_inheritance_chain(node, knowledge_base)

    for level_node in chain:

        exception = next((obj for sub, rel, obj in knowledge_base if sub == level_node and rel == "cannot"), None)
        if exception and relation == "can":
            return f"cannot {exception}"

        local_value = next((obj for sub, rel, obj in knowledge_base if sub == level_node and rel == relation), None)
        if local_value:
            return local_value
    
    return "unknown"

print("===================================================")
print(f"Inheritance chain for penguin: {get_inheritance_chain('penguin', knowledge_base)}")
print("===================================================")
print(f"Inheritance chain for ostrich: {get_inheritance_chain('ostrich', knowledge_base)}")
print("===================================================")
print(f"Query: 'penguin can' -> {query('penguin', 'can', knowledge_base)}")
print("===================================================")
print(f"Query: 'ostrich can' -> {query('ostrich', 'can', knowledge_base)}")
print("===================================================")
print(f"Query: 'bird has' -> {query('bird', 'has', knowledge_base)}")
print("===================================================")
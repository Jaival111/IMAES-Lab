def negate(literal):
    """Return the opposite of a literal."""
    if literal.startswith("not "):
        return literal[4:]
    return "not " + literal

def is_negative(literal):
    """Check whether a literal is negative."""
    return literal.startswith("not ")

# ============================================================
# CORE ALGORITHM 1: MODUS PONENS
# ============================================================
def modus_ponens(facts, rules, justification):
    """Repeatedly apply Modus Ponens until no new fact can be derived."""
    changed = True
    while changed:
        changed = False
        for rule_id, antecedents, consequent in rules:
            if all(a in facts for a in antecedents):
                if consequent not in facts:
                    facts.add(consequent)
                    justification[consequent] = (rule_id, antecedents)
                    print(f"MP: {antecedents} --[{rule_id}]--> {consequent}")
                    changed = True
    return facts

# ============================================================
# CORE ALGORITHM 2: MODUS TOLLENS
# ============================================================
def modus_tollens(facts, rules, justification):
    """Apply Modus Tollens to derive negated antecedents."""
    changed = True
    while changed:
        changed = False
        for rule_id, antecedents, consequent in rules:
            if negate(consequent) not in facts:
                continue

            if len(antecedents) == 1:
                new_fact = negate(antecedents[0])
                if new_fact not in facts:
                    facts.add(new_fact)
                    justification[new_fact] = ("MT-" + rule_id, [negate(consequent)])
                    print(f"MT: {negate(consequent)} --[{rule_id}]--> {new_fact}")
                    changed = True
            else:
                true_antecedents = [a for a in antecedents if a in facts]
                if len(true_antecedents) == len(antecedents) - 1:
                    for a in antecedents:
                        if a not in facts:
                            new_fact = negate(a)
                            if new_fact not in facts:
                                facts.add(new_fact)
                                justification[new_fact] = (
                                    "MT-" + rule_id,
                                    [negate(consequent)] + true_antecedents
                                )
                                print(f"MT: {negate(consequent)} + {true_antecedents} --[{rule_id}]--> {new_fact}")
                                changed = True
                else:
                    pending = [negate(a) for a in antecedents if a not in facts]
                    print(f"MT pending conclusion: {' OR '.join(pending)}")
    return facts

# ============================================================
# UTILITIES
# ============================================================
def check_query(query, facts, rules):
    """Check whether a query is entailed to reject unsound patterns."""
    if query in facts:
        print(f"\nQuery: {query}\nResult: ENTAILED")
        return True
    print(f"\nQuery: {query}\nResult: NOT ENTAILED")
    return False

def print_justifications(facts, justification):
    print("\n--- JUSTIFICATION CHAINS ---")
    for fact in facts:
        if fact in justification:
            rule_id, premises = justification[fact]
            print(f"{fact} <- {rule_id} using {premises}")
        else:
            print(f"{fact} <- INPUT FACT")

def find_contradictions(facts):
    contradictions = []
    for fact in facts:
        opposite = negate(fact)
        if opposite in facts:
            pair = tuple(sorted([fact, opposite]))
            if pair not in contradictions:
                contradictions.append(pair)
    return contradictions

def print_contradictions(facts, justification):
    contradictions = find_contradictions(facts)
    if not contradictions:
        print("\nKnowledge base is CONSISTENT.")
        return
    print("\nKnowledge base is INCONSISTENT!")
    for a, b in contradictions:
        print(f"\nContradiction: {a} AND {b}")
        for item in (a, b):
            if item in justification:
                rule, premises = justification[item]
                print(f"  {item} <- {rule} using {premises}")
            else:
                print(f"  {item} <- INPUT FACT")

def consistent_after_removing(input_facts, rules, removed):
    facts = set(input_facts)
    facts.remove(removed)
    justification = {}
    modus_ponens(facts, rules, justification)
    modus_tollens(facts, rules, justification)
    return len(find_contradictions(facts)) == 0

def find_smallest_removal(input_facts, rules):
    for fact in input_facts:
        if consistent_after_removing(input_facts, rules, fact):
            return [fact]
    return []

# ============================================================
# MAIN PROGRAM EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 60, "\nCORE ALGORITHM\n", "=" * 60)
    rules = [("R1", ["rain"], "wet"), ("R2", ["wet"], "slippery"), ("R3", ["slippery"], "accident")]
    facts = {"rain"}
    justification = {}

    print("\n--- MODUS PONENS ---")
    modus_ponens(facts, rules, justification)
    
    print("\n--- MODUS TOLLENS ---")
    facts.add("not accident")
    modus_tollens(facts, rules, justification)
    print_justifications(facts, justification)

    print("\n\n" + "=" * 60, "\nUNSOUND INFERENCE PATTERNS\n", "=" * 60)
    check_query("rain", {"wet"}, rules)
    check_query("not wet", {"not rain"}, rules)

    print("\n\n" + "=" * 60, "\nEXERCISE 1\n", "=" * 60)
    ex1_rules = [("R4", ["A", "B", "C"], "D")]
    ex1_facts = {"not D"}
    ex1_just = {}
    modus_tollens(ex1_facts, ex1_rules, ex1_just)
    
    print("\nAdding A...")
    ex1_facts.add("A")
    modus_tollens(ex1_facts, ex1_rules, ex1_just)
    
    print("\nAdding B...")
    ex1_facts.add("B")
    modus_tollens(ex1_facts, ex1_rules, ex1_just)

    print("\n\n" + "=" * 60, "\nEXERCISE 2\n", "=" * 60)
    ex2_rules = [("R5", ["A"], "B"), ("R6", ["B"], "not C")]
    ex2_facts = {"A", "C"}
    ex2_just = {}
    
    modus_ponens(ex2_facts, ex2_rules, ex2_just)
    print_contradictions(ex2_facts, ex2_just)
    
    removal = find_smallest_removal(ex2_facts & {"A", "C"}, ex2_rules)
    if removal:
        print("\nSmallest input fact removal:", removal)
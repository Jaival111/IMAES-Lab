import time


class Rule:
    def __init__(self, rule_id, conditions, conclusion):
        self.rule_id = rule_id
        self.conditions = conditions
        self.conclusion = conclusion


def get_conflict_set(rules, wm):
    """Isolates the match phase to compute applicable rules."""
    conflict_set = []
    
    for rule in rules:
        # 1. Refractoriness: Conclusion must not already be in WM
        if rule.conclusion in wm:
            continue
            
        is_applicable = True
        
        # 2. Check all conditions against WM
        for condition in rule.conditions:
            # Handle negated conditions (Exercise 2)
            if condition.startswith("NOT "):
                fact = condition[4:]
                if fact in wm:
                    is_applicable = False
                    break
            # Handle positive conditions
            else:
                if condition not in wm:
                    is_applicable = False
                    break
                    
        if is_applicable:
            conflict_set.append(rule)
            
    return conflict_set


# EXERCISE 1 & 2
print("--- Exercises 1 & 2: Conflict Set Tracking ---")

rule_base = [
    Rule("R1", ["Battery_Dead"], "Car_Wont_Start"),
    Rule("R2", ["Car_Wont_Start", "NOT Lights_Working"], "Check_Alternator"),
    Rule("R3", ["Car_Wont_Start", "Lights_Working"], "Check_Starter")
]

working_memory = set()
print(f"Initial WM: {working_memory}")
print(f"Conflict Set: {[r.rule_id for r in get_conflict_set(rule_base, working_memory)]}\n")

facts_to_add = ["Battery_Dead", "Car_Wont_Start", "Lights_Working"]

for fact in facts_to_add:
    working_memory.add(fact)
    print(f"Added '{fact}' to WM. Current WM: {working_memory}")
    
    applicable_rules = get_conflict_set(rule_base, working_memory)
    print(f"Conflict Set: {[r.rule_id for r in applicable_rules]}\n")


# EXERCISE 3
print("--- Exercise 3: Performance Scaling ---")

# Auto-generate 1,000 rules 
large_rule_base = [Rule(f"Rule_{i}", [f"Fact_{i}"], f"Fact_{i+1}") for i in range(1000)]

# Pre-populate Working Memory with 500 facts
large_wm = {f"Fact_{i}" for i in range(500)}

# Measure execution time using time.perf_counter
start_time = time.perf_counter()
large_conflict_set = get_conflict_set(large_rule_base, large_wm)
end_time = time.perf_counter()

match_time = end_time - start_time
print(f"Rules generated: {len(large_rule_base)}")
print(f"Facts in WM: {len(large_wm)}")
print(f"Time to compute conflict set: {match_time:.6f} seconds")
print(f"Applicable rules found: {len(large_conflict_set)}")
"""AI401 Lab 9: audit elicited rules and induce rules with an ID3 tree."""

import itertools
import math
import random
import statistics


ATTRIBUTES = ("fever", "oxygen", "cough")
ATTRIBUTE_VALUES = {
    "fever": ("normal", "high"),
    "oxygen": ("normal", "low"),
    "cough": ("no", "yes"),
}
LABELS = ("routine", "urgent")


def all_cases():
    cases = []
    for values in itertools.product(*(ATTRIBUTE_VALUES[name] for name in ATTRIBUTES)):
        case = dict(zip(ATTRIBUTES, values))
        case["decision"] = deserved_decision(case)
        cases.append(case)
    return cases


def deserved_decision(case):
    if case["oxygen"] == "low" or (
        case["fever"] == "high" and case["cough"] == "yes"
    ):
        return "urgent"
    return "routine"


ELICITED_RULES = [
    {"id": "R1", "conditions": {"oxygen": "low"}, "decision": "urgent"},
    {
        "id": "R2",
        "conditions": {"fever": "high", "cough": "yes"},
        "decision": "urgent",
    },
    {"id": "R3", "conditions": {"fever": "high"}, "decision": "routine"},
    {
        "id": "R4",
        "conditions": {"cough": "yes", "oxygen": "normal"},
        "decision": "routine",
    },
]


def matching_rules(case, rules):
    return [
        rule
        for rule in rules
        if all(case[name] == value for name, value in rule["conditions"].items())
    ]


def audit_rules(cases, rules):
    uncovered = []
    conflicts = []
    correct_first_matches = 0
    for case in cases:
        fired = matching_rules(case, rules)
        if not fired:
            uncovered.append(case)
            continue
        conclusions = {rule["decision"] for rule in fired}
        if len(conclusions) > 1:
            conflicts.append((case, fired))
        if fired[0]["decision"] == case["decision"]:
            correct_first_matches += 1
    return uncovered, conflicts, correct_first_matches / len(cases)


def entropy(cases):
    if not cases:
        return 0.0
    counts = {}
    for case in cases:
        counts[case["decision"]] = counts.get(case["decision"], 0) + 1
    total = len(cases)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def information_gain(cases, attribute):
    base_entropy = entropy(cases)
    weighted_child_entropy = 0.0
    for value in ATTRIBUTE_VALUES[attribute]:
        subset = [case for case in cases if case[attribute] == value]
        if subset:
            weighted_child_entropy += len(subset) / len(cases) * entropy(subset)
    return base_entropy - weighted_child_entropy


def majority_label(cases):
    counts = {label: 0 for label in LABELS}
    for case in cases:
        counts[case["decision"]] += 1
    return max(LABELS, key=lambda label: counts[label])


def build_tree(cases, available_attributes=None, min_samples_split=1):
    if not cases:
        return {"decision": "routine"}
    labels = {case["decision"] for case in cases}
    if len(labels) == 1:
        return {"decision": next(iter(labels))}
    if not available_attributes or len(cases) < min_samples_split:
        return {"decision": majority_label(cases)}

    gains = {
        attribute: information_gain(cases, attribute)
        for attribute in available_attributes
    }
    attribute = max(available_attributes, key=lambda name: gains[name])
    remaining = [name for name in available_attributes if name != attribute]
    branches = {}
    for value in ATTRIBUTE_VALUES[attribute]:
        subset = [case for case in cases if case[attribute] == value]
        if subset:
            branches[value] = build_tree(subset, remaining, min_samples_split)
        else:
            branches[value] = {"decision": majority_label(cases)}
    return {
        "attribute": attribute,
        "branches": branches,
        "default": majority_label(cases),
    }


def predict(tree, case):
    while "decision" not in tree:
        attribute = tree["attribute"]
        tree = tree["branches"].get(case[attribute], {"decision": tree["default"]})
    return tree["decision"]


def tree_accuracy(tree, cases):
    return sum(predict(tree, case) == case["decision"] for case in cases) / len(cases)


def tree_depth(tree):
    if "decision" in tree:
        return 0
    return 1 + max(tree_depth(child) for child in tree["branches"].values())


def print_tree(tree, indent=""):
    if "decision" in tree:
        print(f"{indent}LEAF -> {tree['decision']}")
        return
    attribute = tree["attribute"]
    for value, child in tree["branches"].items():
        print(f"{indent}IF {attribute} = {value}:")
        print_tree(child, indent + "  ")


def tree_rules(tree, conditions=None):
    if conditions is None:
        conditions = []
    if "decision" in tree:
        return [(conditions, tree["decision"])]

    result = []
    attribute = tree["attribute"]
    for value, child in tree["branches"].items():
        result.extend(tree_rules(child, conditions + [(attribute, value)]))
    return result


def tree_agrees_with_elicited_rule(tree, rule, cases):
    matched = [case for case in cases if matching_rules(case, [rule])]
    return bool(matched) and all(predict(tree, case) == rule["decision"] for case in matched)


def describe_case(case):
    return ", ".join(f"{attribute}={case[attribute]}" for attribute in ATTRIBUTES)


def sample_case_base(cases, size, rng, corruption=0.0):
    sampled = []
    for _ in range(size):
        case = dict(rng.choice(cases))
        if rng.random() < corruption:
            case["decision"] = (
                "urgent" if case["decision"] == "routine" else "routine"
            )
        sampled.append(case)
    return sampled


def run_algorithm_1(cases):
    print("=" * 76)
    print("ALGORITHM 1 - AUDITING AN ELICITED RULE BASE")
    print("=" * 76)
    print("Attributes and values:", ATTRIBUTE_VALUES)
    print(f"Complete case space: {len(cases)} cases")
    print("Elicited rules:")
    for rule in ELICITED_RULES:
        tests = " AND ".join(
            f"{attribute}={value}" for attribute, value in rule["conditions"].items()
        )
        print(f"  {rule['id']}: IF {tests} THEN {rule['decision']}")

    uncovered, conflicts, accuracy = audit_rules(cases, ELICITED_RULES)
    print("\nUncovered cases:")
    if uncovered:
        for case in uncovered:
            print(f"  {describe_case(case)} | deserved={case['decision']}")
    else:
        print("  none")

    print("\nConflicting cases (firing rules have different conclusions):")
    if conflicts:
        for case, fired in conflicts:
            rule_results = ", ".join(
                f"{rule['id']}->{rule['decision']}" for rule in fired
            )
            print(f"  {describe_case(case)} | {rule_results}")
    else:
        print("  none")
    print(
        f"\nFirst-matching-rule accuracy over all cases: "
        f"{accuracy:.1%} ({round(accuracy * len(cases))}/{len(cases)})"
    )


def run_algorithm_2(cases):
    print("\n" + "=" * 76)
    print("ALGORITHM 2 - INDUCING RULES FROM SOLVED CASES")
    print("=" * 76)
    tree = build_tree(cases, list(ATTRIBUTES))
    print("Decision tree:")
    print_tree(tree)
    induced_rules = tree_rules(tree)
    print("\nRules read from root-to-leaf paths:")
    for index, (conditions, decision) in enumerate(induced_rules, start=1):
        tests = " AND ".join(f"{name}={value}" for name, value in conditions) or "TRUE"
        print(f"  T{index}: IF {tests} THEN {decision}")
    print(f"Number of induced rules: {len(induced_rules)}")
    print(f"Induced-tree accuracy over complete case space: {tree_accuracy(tree, cases):.1%}")
    agreements = [
        rule["id"]
        for rule in ELICITED_RULES
        if tree_agrees_with_elicited_rule(tree, rule, cases)
    ]
    print(
        f"Induced tree agrees with {len(agreements)}/{len(ELICITED_RULES)} elicited rules:",
        ", ".join(agreements) if agreements else "none",
    )
    return tree


def run_exercise_1(cases):
    print("\n" + "=" * 76)
    print("EXERCISE 1 - CORRUPTED LABELS AND MINIMUM SPLIT SIZE")
    print("=" * 76)
    print("20 sampled case bases per level; 16 examples per case base.")
    print("Accuracy is measured against the true label over the complete 8-case space.")
    print("corruption | unrestricted leaves | unrestricted accuracy | min-size-4 leaves | min-size-4 accuracy")
    print("-" * 100)

    for corruption in (0.0, 0.05, 0.10, 0.20):
        leaves_unrestricted = []
        accuracy_unrestricted = []
        leaves_limited = []
        accuracy_limited = []
        for seed in range(20):
            base = sample_case_base(
                cases,
                16,
                random.Random(9000 + seed),
                corruption=corruption,
            )
            full_tree = build_tree(base, list(ATTRIBUTES))
            limited_tree = build_tree(base, list(ATTRIBUTES), min_samples_split=4)
            leaves_unrestricted.append(len(tree_rules(full_tree)))
            accuracy_unrestricted.append(tree_accuracy(full_tree, cases))
            leaves_limited.append(len(tree_rules(limited_tree)))
            accuracy_limited.append(tree_accuracy(limited_tree, cases))
        print(
            f"{corruption:>10.0%} | "
            f"{statistics.mean(leaves_unrestricted):>19.2f} | "
            f"{statistics.mean(accuracy_unrestricted):>20.1%} | "
            f"{statistics.mean(leaves_limited):>17.2f} | "
            f"{statistics.mean(accuracy_limited):>19.1%}"
        )
    print(
        "The minimum-split rule stops fitting very small, potentially noisy subsets; "
        "whether this helps depends on how much label noise the sample contains."
    )


def run_exercise_2(cases, elicited_accuracy):
    print("\n" + "=" * 76)
    print("EXERCISE 2 - CASE-BASE SIZE SWEEP")
    print("=" * 76)
    print("30 draws with replacement at each size; rules are compared on the full case space.")
    print("cases | mean leaves | mean full-space accuracy | mean elicited-rule agreements")
    print("-" * 78)
    results = []

    for size in (8, 16, 24, 40, 80):
        leaves = []
        accuracies = []
        agreements = []
        for seed in range(30):
            base = sample_case_base(cases, size, random.Random(12000 + seed))
            tree = build_tree(base, list(ATTRIBUTES))
            leaves.append(len(tree_rules(tree)))
            accuracies.append(tree_accuracy(tree, cases))
            agreements.append(
                sum(
                    tree_agrees_with_elicited_rule(tree, rule, cases)
                    for rule in ELICITED_RULES
                )
            )
        mean_accuracy = statistics.mean(accuracies)
        results.append((size, mean_accuracy))
        print(
            f"{size:>5} | {statistics.mean(leaves):>11.2f} | "
            f"{mean_accuracy:>23.1%} | {statistics.mean(agreements):>28.2f}/4"
        )

    overtakes = [size for size, accuracy in results if accuracy > elicited_accuracy]
    print(
        "First case-base size whose mean induced accuracy exceeds the elicited accuracy:",
        overtakes[0] if overtakes else "none in this sweep",
    )
    print(
        "The agreement count is determined by the induced predictions over each elicited "
        "rule's firing region; more examples can stabilize predictions but cannot make "
        "a genuinely conflicting elicited rule agree everywhere."
    )


def main():
    cases = all_cases()
    _, _, elicited_accuracy = audit_rules(cases, ELICITED_RULES)
    run_algorithm_1(cases)
    run_algorithm_2(cases)
    run_exercise_1(cases)
    run_exercise_2(cases, elicited_accuracy)


if __name__ == "__main__":
    main()
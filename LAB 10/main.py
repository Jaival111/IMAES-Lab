"""AI401 Lab 10: fairness, privacy, and transparency audits of a decision tree."""

import math
import random
import statistics


PROTECTED_GROUPS = ("A", "B")
BASE_FEATURES = (
    "group",
    "experience",
    "test_score",
    "education",
    "region",
    "age_band",
)
QUASI_IDENTIFIERS = ("age_band", "education", "region")


def generate_population(size, seed, bias_rate=0.25):
    """Generate applicants with correlated features and historically biased decisions."""
    rng = random.Random(seed)
    applicants = []
    for index in range(size):
        group = "A" if rng.random() < 0.5 else "B"
        disadvantaged = group == "B"

        if disadvantaged:
            experience = "high" if rng.random() < 0.35 else "low"
            test_score = "high" if rng.random() < 0.40 else "low"
            education = rng.choices(
                ("basic", "vocational", "advanced"), weights=(0.50, 0.35, 0.15)
            )[0]
            region = rng.choices(
                ("north", "central", "south"), weights=(0.20, 0.25, 0.55)
            )[0]
            age_band = rng.choices(
                ("young", "middle", "senior"), weights=(0.45, 0.40, 0.15)
            )[0]
        else:
            experience = "high" if rng.random() < 0.60 else "low"
            test_score = "high" if rng.random() < 0.65 else "low"
            education = rng.choices(
                ("basic", "vocational", "advanced"), weights=(0.20, 0.35, 0.45)
            )[0]
            region = rng.choices(
                ("north", "central", "south"), weights=(0.45, 0.35, 0.20)
            )[0]
            age_band = rng.choices(
                ("young", "middle", "senior"), weights=(0.25, 0.50, 0.25)
            )[0]

        deserved = test_score == "high" and (
            experience == "high" or education == "advanced"
        )
        historical = deserved
        if deserved and disadvantaged and rng.random() < bias_rate:
            historical = False
        elif rng.random() < 0.06:
            historical = not historical

        applicants.append(
            {
                "applicant_id": f"AP{index:04d}",
                "group": group,
                "experience": experience,
                "test_score": test_score,
                "education": education,
                "region": region,
                "age_band": age_band,
                "deserved": "accept" if deserved else "reject",
                "historical": "accept" if historical else "reject",
            }
        )
    return applicants


def entropy(applicants, target):
    if not applicants:
        return 0.0
    counts = {}
    for applicant in applicants:
        label = applicant[target]
        counts[label] = counts.get(label, 0) + 1
    total = len(applicants)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def information_gain(applicants, attribute, target):
    base_entropy = entropy(applicants, target)
    groups = {}
    for applicant in applicants:
        groups.setdefault(applicant[attribute], []).append(applicant)
    remainder = sum(
        len(subset) / len(applicants) * entropy(subset, target)
        for subset in groups.values()
    )
    return base_entropy - remainder


def majority_label(applicants, target):
    counts = {}
    for applicant in applicants:
        counts[applicant[target]] = counts.get(applicant[target], 0) + 1
    return max(sorted(counts), key=lambda label: counts[label])


def build_tree(applicants, features, target):
    labels = {applicant[target] for applicant in applicants}
    if len(labels) == 1:
        return {"decision": next(iter(labels))}
    if not features:
        return {"decision": majority_label(applicants, target)}

    gains = {
        feature: information_gain(applicants, feature, target)
        for feature in features
    }
    attribute = max(features, key=lambda feature: gains[feature])
    remaining = [feature for feature in features if feature != attribute]
    branches = {}
    for value in sorted({applicant[attribute] for applicant in applicants}):
        subset = [applicant for applicant in applicants if applicant[attribute] == value]
        branches[value] = build_tree(subset, remaining, target)
    return {
        "attribute": attribute,
        "branches": branches,
        "default": majority_label(applicants, target),
    }


def predict(tree, applicant):
    while "decision" not in tree:
        attribute = tree["attribute"]
        tree = tree["branches"].get(
            applicant.get(attribute), {"decision": tree["default"]}
        )
    return tree["decision"]


def predict_with_trace(tree, applicant):
    trace = []
    while "decision" not in tree:
        attribute = tree["attribute"]
        value = applicant.get(attribute, "suppressed")
        trace.append(f"{attribute}={value}")
        if value in tree["branches"]:
            tree = tree["branches"][value]
        else:
            trace.append(f"no branch; default={tree['default']}")
            return tree["default"], trace
    return tree["decision"], trace


def accuracy(tree, applicants, target):
    return sum(predict(tree, applicant) == applicant[target] for applicant in applicants) / len(
        applicants
    )


def agreement_with_historical(tree, applicants):
    return accuracy(tree, applicants, "historical")


def transform_applicant(applicant, level):
    transformed = dict(applicant)
    if level in ("generalized", "suppressed"):
        transformed["age_band"] = {
            "young": "under-50",
            "middle": "under-50",
            "senior": "50-plus",
        }[applicant["age_band"]]
        transformed["education"] = {
            "basic": "non-degree",
            "vocational": "non-degree",
            "advanced": "degree",
        }[applicant["education"]]
        transformed["region"] = {
            "north": "north-central",
            "central": "north-central",
            "south": "south",
        }[applicant["region"]]
    if level == "suppressed":
        transformed.pop("region")
    return transformed


def quasi_identifier_statistics(applicants, attributes):
    equivalence_class_sizes = {}
    for applicant in applicants:
        key = tuple(applicant[attribute] for attribute in attributes)
        equivalence_class_sizes[key] = equivalence_class_sizes.get(key, 0) + 1

    sizes = list(equivalence_class_sizes.values())
    records_in_small_classes = sum(
        size for size in sizes if size < 5
    )
    singleton_records = sum(size for size in sizes if size == 1)
    return {
        "smallest_class": min(sizes) if sizes else 0,
        "unique_records": singleton_records,
        "small_classes_fraction": records_in_small_classes / len(applicants)
        if applicants
        else 0.0,
    }


def group_metrics(applicants, predictions, group):
    members = [
        (applicant, prediction)
        for applicant, prediction in zip(applicants, predictions)
        if applicant["group"] == group
    ]
    deserved_positive = sum(applicant["deserved"] == "accept" for applicant, _ in members)
    deserved_negative = len(members) - deserved_positive
    selected = sum(prediction == "accept" for _, prediction in members)
    true_positive = sum(
        prediction == "accept" and applicant["deserved"] == "accept"
        for applicant, prediction in members
    )
    false_positive = sum(
        prediction == "accept" and applicant["deserved"] == "reject"
        for applicant, prediction in members
    )
    return {
        "base_rate": deserved_positive / len(members) if members else 0.0,
        "selection_rate": selected / len(members) if members else 0.0,
        "tpr": true_positive / deserved_positive if deserved_positive else 0.0,
        "fpr": false_positive / deserved_negative if deserved_negative else 0.0,
        "precision": true_positive / selected if selected else 0.0,
    }


def fairness_gaps(applicants, predictions):
    metrics = {
        group: group_metrics(applicants, predictions, group)
        for group in PROTECTED_GROUPS
    }
    gaps = {
        metric: abs(metrics["A"][metric] - metrics["B"][metric])
        for metric in ("base_rate", "selection_rate", "tpr", "fpr", "precision")
    }
    return metrics, gaps


def mutual_information_with_group(applicants, attribute):
    return information_gain(applicants, attribute, "group")


def print_fairness_table(applicants, predictions):
    metrics, gaps = fairness_gaps(applicants, predictions)
    print("Group | base rate | selection | true positive | false positive | precision")
    print("------+-----------+-----------+---------------+----------------+----------")
    for group in PROTECTED_GROUPS:
        row = metrics[group]
        print(
            f"{group:>5} | {row['base_rate']:>9.1%} | {row['selection_rate']:>9.1%} | "
            f"{row['tpr']:>13.1%} | {row['fpr']:>14.1%} | {row['precision']:>9.1%}"
        )
    print(
        "A-B absolute gap | "
        + " | ".join(
            f"{gaps[name]:>8.1%}"
            for name in ("base_rate", "selection_rate", "tpr", "fpr", "precision")
        )
    )
    return metrics, gaps


def run_algorithm_1(train, test):
    print("=" * 84)
    print("ALGORITHM 1 - FAIRNESS AUDIT OF HISTORICAL DECISIONS")
    print("=" * 84)
    print(
        f"Training applicants: {len(train)} | test applicants: {len(test)} | "
        "historical rejection bias among qualified group-B applicants: 25%"
    )
    tree = build_tree(train, list(BASE_FEATURES), "historical")
    predictions = [predict(tree, applicant) for applicant in test]
    print("\nPer-group test metrics and absolute gaps:")
    print_fairness_table(test, predictions)
    print(f"\nAgreement with historical test decisions: {agreement_with_historical(tree, test):.1%}")
    print(f"Accuracy against deserved test outcomes: {accuracy(tree, test, 'deserved'):.1%}")
    return tree


def run_algorithm_2(train, test):
    print("\n" + "=" * 84)
    print("ALGORITHM 2 - PRIVACY AND TRANSPARENCY AUDIT")
    print("=" * 84)
    tree = build_tree(train, list(BASE_FEATURES), "historical")

    print("Quasi-identifier privacy statistics on the training records:")
    print("level        | QI attributes                    | min class | singleton records | records in classes <5")
    print("-------------+----------------------------------+-----------+-------------------+----------------------")
    privacy_results = []
    level_features = {
        "raw": list(QUASI_IDENTIFIERS),
        "generalized": list(QUASI_IDENTIFIERS),
        "suppressed": ["age_band", "education"],
    }
    for level in ("raw", "generalized", "suppressed"):
        transformed = [transform_applicant(applicant, level) for applicant in train]
        stats = quasi_identifier_statistics(transformed, level_features[level])
        privacy_results.append((level, stats))
        qi_text = ", ".join(level_features[level])
        print(
            f"{level:<12} | {qi_text:<32} | "
            f"{stats['smallest_class']:>9} | {stats['unique_records']:>17} | "
            f"{stats['small_classes_fraction']:>20.1%}"
        )

    traces = [predict_with_trace(tree, applicant)[1] for applicant in test]
    lengths = [len(trace) for trace in traces]
    print(
        f"\nExplanation length over {len(test)} test applicants: "
        f"mean={statistics.mean(lengths):.2f}, maximum={max(lengths)} conditions"
    )
    named_applicant = test[0]
    decision, trace = predict_with_trace(tree, named_applicant)
    print(f"\nExplanation for {named_applicant['applicant_id']}:")
    if trace:
        for step, condition in enumerate(trace, start=1):
            print(f"  {step}. {condition}")
    else:
        print("  The tree makes a decision at its root (no attribute tests).")
    print(f"  Decision: {decision}")
    return privacy_results


def run_exercise_1(train, test):
    print("\n" + "=" * 84)
    print("EXERCISE 1 - MITIGATING PROTECTED-ATTRIBUTE BIAS")
    print("=" * 84)
    non_protected_features = [name for name in BASE_FEATURES if name != "group"]
    most_correlated = max(
        non_protected_features,
        key=lambda name: mutual_information_with_group(train, name),
    )
    print(
        f"Most group-associated feature by training-set mutual information: "
        f"{most_correlated} "
        f"({mutual_information_with_group(train, most_correlated):.3f} bits)"
    )
    configurations = [
        ("Baseline (all features)", list(BASE_FEATURES), "historical"),
        (
            "Remove protected attribute",
            [name for name in BASE_FEATURES if name != "group"],
            "historical",
        ),
        (
            f"Remove group and {most_correlated}",
            [name for name in BASE_FEATURES if name not in ("group", most_correlated)],
            "historical",
        ),
        ("Train on deserved outcomes", list(BASE_FEATURES), "deserved"),
    ]
    print(
        "configuration | accuracy vs deserved | selection gap | TPR gap | precision gap"
    )
    print("-" * 91)
    for label, features, target in configurations:
        tree = build_tree(train, features, target)
        predictions = [predict(tree, applicant) for applicant in test]
        _, gaps = fairness_gaps(test, predictions)
        print(
            f"{label:<29} | {accuracy(tree, test, 'deserved'):>19.1%} | "
            f"{gaps['selection_rate']:>13.1%} | {gaps['tpr']:>7.1%} | "
            f"{gaps['precision']:>13.1%}"
        )
    print(
        "Removing the protected attribute does not remove correlated proxies. "
        "Training on deserved outcomes uses labels unavailable to a real system; "
        "it is an audit comparison, not an operational mitigation by itself."
    )


def run_exercise_2(train, test):
    print("\n" + "=" * 84)
    print("EXERCISE 2 - INDUCTION AFTER QUASI-IDENTIFIER GENERALIZATION")
    print("=" * 84)
    configurations = (
        ("raw", list(BASE_FEATURES), list(QUASI_IDENTIFIERS)),
        ("generalized", list(BASE_FEATURES), list(QUASI_IDENTIFIERS)),
        (
            "suppressed",
            [feature for feature in BASE_FEATURES if feature != "region"],
            ["age_band", "education"],
        ),
    )
    print(
        "level        | min class | unique records | historical agreement | "
        "accuracy vs deserved | mean explanation"
    )
    print(
        "-------------+-----------+----------------+----------------------+"
        "----------------------+-----------------"
    )
    for level, features, qid_features in configurations:
        transformed_train = [transform_applicant(applicant, level) for applicant in train]
        transformed_test = [transform_applicant(applicant, level) for applicant in test]
        tree = build_tree(transformed_train, features, "historical")
        privacy = quasi_identifier_statistics(transformed_train, qid_features)
        explanations = [
            len(predict_with_trace(tree, applicant)[1]) for applicant in transformed_test
        ]
        print(
            f"{level:<12} | {privacy['smallest_class']:>9} | "
            f"{privacy['unique_records']:>14} | "
            f"{agreement_with_historical(tree, transformed_test):>20.1%} | "
            f"{accuracy(tree, transformed_test, 'deserved'):>20.1%} | "
            f"{statistics.mean(explanations):>15.2f}"
        )
    print(
        "Generalization and suppression merge equivalence classes and may reduce "
        "re-identification risk, but they discard distinctions that can affect "
        "prediction and change the explanations the tree can provide."
    )


def main():
    train = generate_population(800, seed=41)
    test = generate_population(400, seed=2026)
    run_algorithm_1(train, test)
    run_algorithm_2(train, test)
    run_exercise_1(train, test)
    run_exercise_2(train, test)


if __name__ == "__main__":
    main()
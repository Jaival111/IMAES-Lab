from itertools import product


# ALGORITHM 1: SPECIFY THE NETWORK AND EVALUATE THE FULL JOINT
def validate_network(network):
    for variable, data in network.items():
        expected_rows = 2 ** len(data["parents"])
        if len(data["cpt"]) != expected_rows:
            raise ValueError(f"Invalid CPT for {variable}")
        if any(not 0 <= probability <= 1 for probability in data["cpt"].values()):
            raise ValueError(f"Invalid probability for {variable}")

    parameters = sum(len(data["cpt"]) for data in network.values())
    full_joint_parameters = 2 ** len(network) - 1
    return parameters, full_joint_parameters


def probability(variable, assignment, network):
    data = network[variable]
    parent_values = tuple(assignment[parent] for parent in data["parents"])
    p_true = data["cpt"][parent_values]
    return p_true if assignment[variable] else 1 - p_true


def joint_probability(assignment, network):
    result = 1.0
    for variable in network:
        result *= probability(variable, assignment, network)
    return result


def all_assignments(network):
    variables = list(network)
    for values in product([False, True], repeat=len(variables)):
        yield dict(zip(variables, values))


# ALGORITHM 2: POSTERIOR INFERENCE BY ENUMERATION
def posterior(query, evidence, network):
    unnormalised = {}
    for query_value in [False, True]:
        total = 0.0
        for assignment in all_assignments(network):
            if assignment[query] == query_value and all(
                assignment[key] == value for key, value in evidence.items()
            ):
                total += joint_probability(assignment, network)
        unnormalised[query_value] = total

    normalising_constant = sum(unnormalised.values())
    return {
        False: unnormalised[False] / normalising_constant,
        True: unnormalised[True] / normalising_constant,
    }, normalising_constant


# NETWORK AND CALLS FOR BOTH ALGORITHMS
network = {
    "Burglary": {"parents": [], "cpt": {(): 0.001}},
    "Earthquake": {"parents": [], "cpt": {(): 0.002}},
    "Alarm": {
        "parents": ["Burglary", "Earthquake"],
        "cpt": {(False, False): 0.001, (False, True): 0.29,
                (True, False): 0.94, (True, True): 0.95},
    },
    "JohnCalls": {"parents": ["Alarm"], "cpt": {(False,): 0.05, (True,): 0.90}},
    "MaryCalls": {"parents": ["Alarm"], "cpt": {(False,): 0.01, (True,): 0.70}},
}

parameters, full_joint_parameters = validate_network(network)
print("Algorithm 1")
print("Parameters:", parameters, "of", full_joint_parameters)
entry = {"Burglary": True, "Earthquake": False, "Alarm": True,
         "JohnCalls": True, "MaryCalls": False}
print("Joint entry:", joint_probability(entry, network))
print("Sum of full joint:", sum(joint_probability(a, network) for a in all_assignments(network)))

print("\nAlgorithm 2")
result, evidence_probability = posterior("Burglary", {"JohnCalls": True}, network)
print("P(Burglary | JohnCalls):", result, "P(evidence):", evidence_probability)


# EXERCISE 1: ADD THE SECOND CAUSE AS EVIDENCE
print("\nExercise 1")
without_earthquake, _ = posterior("Burglary", {"Alarm": True}, network)
with_earthquake, _ = posterior("Burglary", {"Alarm": True, "Earthquake": True}, network)
print("P(Burglary | Alarm):", without_earthquake)
print("P(Burglary | Alarm, Earthquake):", with_earthquake)
print("Direction:", "increases" if with_earthquake[True] > without_earthquake[True] else "decreases")


# EXERCISE 2: EXTEND THE NETWORK WITH A CHILD OF ALARM
extended_network = dict(network)
extended_network["PoliceCalls"] = {
    "parents": ["Alarm"],
    "cpt": {(False,): 0.02, (True,): 0.30},
}

print("\nExercise 2")
parameters, full_joint_parameters = validate_network(extended_network)
print("Parameters:", parameters, "of", full_joint_parameters)
for police_calls in [False, True]:
    result, _ = posterior("Burglary", {"JohnCalls": True, "PoliceCalls": police_calls}, extended_network)
    print(f"P(Burglary | JohnCalls, PoliceCalls={police_calls}):", result)

alarm_only, _ = posterior("Burglary", {"Alarm": True}, extended_network)
alarm_with_police, _ = posterior("Burglary", {"Alarm": True, "PoliceCalls": False}, extended_network)
independent, _ = posterior("Burglary", {"JohnCalls": True}, extended_network)
independent_with_police, _ = posterior("Burglary", {"JohnCalls": True, "PoliceCalls": False}, extended_network)
print("Independent check P(Burglary | Alarm), P(Burglary | Alarm, PoliceCalls=False):", alarm_only, alarm_with_police)
print("Dependent pair check P(Burglary | JohnCalls), P(Burglary | JohnCalls, PoliceCalls=False):", independent, independent_with_police)
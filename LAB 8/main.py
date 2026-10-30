"""AI401 Lab 8: Q-learning in a grid world and a coordination game."""

import random
import statistics


ROWS = 5
COLUMNS = 5
START = (0, 0)
GOAL = (4, 4)
PIT = (2, 2)
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_SYMBOLS = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}
ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
TERMINAL_STATES = {GOAL, PIT}


def move(state, action):
    """Apply a move, leaving the agent in place if it would leave the grid."""
    row, column = state
    row_delta, column_delta = ACTION_DELTAS[action]
    next_state = (row + row_delta, column + column_delta)
    if not (0 <= next_state[0] < ROWS and 0 <= next_state[1] < COLUMNS):
        return state
    return next_state


def transition_reward(next_state):
    if next_state == GOAL:
        return 10
    if next_state == PIT:
        return -10
    return -1


def greedy_actions(q_values, state):
    values = q_values[state]
    best_value = max(values.values())
    return [action for action in ACTIONS if values[action] == best_value]


def optimal_action_sets(discount):
    """Solve the small known grid only for evaluating the learned policy."""
    values = {(row, column): 0.0 for row in range(ROWS) for column in range(COLUMNS)}
    for _ in range(1000):
        largest_change = 0.0
        updated = dict(values)
        for state in values:
            if state in TERMINAL_STATES:
                continue
            updated[state] = max(
                transition_reward(move(state, action))
                + discount * values[move(state, action)]
                for action in ACTIONS
            )
            largest_change = max(largest_change, abs(updated[state] - values[state]))
        values = updated
        if largest_change < 1e-10:
            break

    optimal = {}
    for state in values:
        if state in TERMINAL_STATES:
            continue
        action_values = {
            action: transition_reward(move(state, action))
            + discount * values[move(state, action)]
            for action in ACTIONS
        }
        best_value = max(action_values.values())
        optimal[state] = {
            action
            for action, value in action_values.items()
            if abs(value - best_value) < 1e-8
        }
    return optimal


def policy_is_optimal(q_values, optimal):
    return all(set(greedy_actions(q_values, state)) <= actions for state, actions in optimal.items())


def policy_path(q_values):
    """Return one deterministic greedy route, or None if greedy choices loop."""
    state = START
    visited = {state}
    path = []
    tie_order = ("RIGHT", "DOWN", "LEFT", "UP")

    for _ in range(ROWS * COLUMNS * 2):
        if state == GOAL:
            return path
        if state in TERMINAL_STATES:
            return None

        candidates = set(greedy_actions(q_values, state))
        ordered = [action for action in tie_order if action in candidates]
        action = next(
            (candidate for candidate in ordered
             if move(state, candidate) != state and move(state, candidate) not in visited),
            None,
        )
        if action is None:
            return None

        state = move(state, action)
        path.append(action)
        visited.add(state)
    return None


def train_grid_world(
    episodes=1200,
    alpha=0.2,
    discount=0.9,
    epsilon=0.1,
    seed=42,
    initial_value=0.0,
    step_limit=100,
):
    rng = random.Random(seed)
    q_values = {
        (row, column): {action: initial_value for action in ACTIONS}
        for row in range(ROWS)
        for column in range(COLUMNS)
    }
    returns = []
    first_optimal_episode = None
    optimal = optimal_action_sets(discount)

    for episode in range(1, episodes + 1):
        state = START
        episode_return = 0

        for _ in range(step_limit):
            if rng.random() < epsilon:
                action = rng.choice(ACTIONS)
            else:
                best_actions = greedy_actions(q_values, state)
                action = rng.choice(best_actions)

            next_state = move(state, action)
            reward = transition_reward(next_state)
            bootstrap = 0.0 if next_state in TERMINAL_STATES else max(q_values[next_state].values())
            old_value = q_values[state][action]
            q_values[state][action] = old_value + alpha * (
                reward + discount * bootstrap - old_value
            )
            episode_return += reward
            state = next_state

            if state in TERMINAL_STATES:
                break

        returns.append(episode_return)
        if first_optimal_episode is None and policy_is_optimal(q_values, optimal):
            first_optimal_episode = episode

    return q_values, returns, first_optimal_episode


def print_policy(q_values):
    print("\nGreedy policy (all tied best actions are shown):")
    for row in range(ROWS):
        cells = []
        for column in range(COLUMNS):
            state = (row, column)
            if state == START:
                cell = "S"
            elif state == GOAL:
                cell = "G"
            elif state == PIT:
                cell = "P"
            else:
                cell = "".join(ACTION_SYMBOLS[action] for action in greedy_actions(q_values, state))
            cells.append(f"{cell:^5}")
        print("".join(cells))


def episode_summary(returns):
    midpoint = len(returns) // 2
    early = statistics.mean(returns[:midpoint])
    late = statistics.mean(returns[-50:])
    print(f"Mean return in first half ({midpoint} episodes): {early:.2f}")
    print(f"Mean return in last 50 episodes: {late:.2f}")


def print_single_agent_result(q_values, returns, first_optimal_episode):
    print_policy(q_values)
    path = policy_path(q_values)
    if path is None:
        print("Greedy path from start: no terminal route found")
    else:
        print("Greedy path from start:", " -> ".join(path))
        print("Steps from start to goal:", len(path))
    print("Shortest safe path length:", 8)
    print(
        "First episode with a globally optimal greedy policy:",
        first_optimal_episode if first_optimal_episode is not None else "not reached",
    )
    theoretical_start_actions = optimal_action_sets(0.9)[START]
    print(
        "Optimal start actions in the symmetric grid:",
        ", ".join(sorted(theoretical_start_actions)),
    )
    print("Highest-valued actions at the start:", ", ".join(greedy_actions(q_values, START)))
    episode_summary(returns)


GAME_ACTIONS = ("A", "SAFE", "B")


def game_payoff_matrix(penalty):
    return {
        ("A", "A"): 10,
        ("A", "SAFE"): 0,
        ("A", "B"): penalty,
        ("SAFE", "A"): 0,
        ("SAFE", "SAFE"): 2,
        ("SAFE", "B"): 0,
        ("B", "A"): penalty,
        ("B", "SAFE"): 0,
        ("B", "B"): 10,
    }


def choose_game_action(estimates, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(GAME_ACTIONS)
    best_value = max(estimates.values())
    return rng.choice([action for action in GAME_ACTIONS if estimates[action] == best_value])


def train_coordination_game(
    penalty=-50,
    rounds=5000,
    alpha=0.1,
    epsilon=0.1,
    seed=7,
    update="average",
):
    rng = random.Random(seed)
    payoff = game_payoff_matrix(penalty)
    estimates = [
        {action: 0.0 for action in GAME_ACTIONS},
        {action: 0.0 for action in GAME_ACTIONS},
    ]
    observed_actions = [set(), set()]
    recent_pairs = []

    for _ in range(rounds):
        actions = [
            choose_game_action(agent_estimates, epsilon, rng)
            for agent_estimates in estimates
        ]
        reward = payoff[tuple(actions)]
        for index, action in enumerate(actions):
            if update == "best":
                if action not in observed_actions[index]:
                    estimates[index][action] = reward
                    observed_actions[index].add(action)
                else:
                    estimates[index][action] = max(estimates[index][action], reward)
            else:
                old_value = estimates[index][action]
                estimates[index][action] = old_value + alpha * (reward - old_value)
        recent_pairs.append(tuple(actions))

    return estimates, recent_pairs[-100:]


def classify_pair(pair):
    first, second = pair
    if first == second and first in ("A", "B"):
        return "optimal joint action"
    if first == second == "SAFE":
        return "safe joint action"
    return "mismatch"


def print_payoff_matrix(penalty):
    payoffs = game_payoff_matrix(penalty)
    print(f"Common payoff matrix (penalty k={penalty}):")
    print("Agent 1 \\ Agent 2 |   A | SAFE |   B")
    print("-------------------+-----+------+-----")
    for first in GAME_ACTIONS:
        print(
            f"{first:^19}|"
            + "".join(f"{payoffs[first, second]:>6}" for second in GAME_ACTIONS)
        )


def run_algorithm_1():
    print("=" * 72)
    print("ALGORITHM 1 - SINGLE-AGENT Q-LEARNING")
    print("=" * 72)
    q_values, returns, first_optimal = train_grid_world()
    print("Grid: 5x5 | start: (0, 0) | goal: (4, 4) | pit: (2, 2)")
    print("Rewards: goal +10, pit -10, other moves -1; off-grid moves stay in place.")
    print("Episodes: 1200 | alpha: 0.2 | discount: 0.9 | epsilon: 0.1")
    print_single_agent_result(q_values, returns, first_optimal)


def run_algorithm_2():
    print("\n" + "=" * 72)
    print("ALGORITHM 2 - TWO INDEPENDENT Q-LEARNERS")
    print("=" * 72)
    penalty = -50
    print_payoff_matrix(penalty)
    estimates, recent_pairs = train_coordination_game(penalty=penalty)
    chosen_pair = tuple(
        max(agent_estimates, key=agent_estimates.get) for agent_estimates in estimates
    )
    print("Final estimates:")
    for index, agent_estimates in enumerate(estimates, start=1):
        print(f"  Agent {index}: {agent_estimates}")
    print("Chosen greedy actions:", chosen_pair)
    print("Joint outcome:", classify_pair(chosen_pair))
    print("Most common outcome in last 100 rounds:")
    counts = {pair: recent_pairs.count(pair) for pair in set(recent_pairs)}
    for pair, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {pair}: {count} rounds ({classify_pair(pair)})")
    print(
        "Equilibrium note: both matching optimal pairs and (SAFE, SAFE) are Nash equilibria; "
        "a mismatched optimal pair is not when its penalty is negative."
    )


def run_exercise_1():
    print("\n" + "=" * 72)
    print("EXERCISE 1 - EXPLORATION AND LEARNING-RATE SWEEP")
    print("=" * 72)
    print("Each setting uses 10 seeds and 700 episodes per run.")
    print("A run succeeds when every state's greedy actions are globally optimal.")
    print("epsilon | alpha | optimal runs/10 | median first-optimal episode | mean last-50 return")
    print("-" * 91)

    episodes = 700
    for epsilon in (0.0, 0.05, 0.1, 0.3):
        for alpha in (0.1, 0.5):
            first_episodes = []
            late_returns = []
            for seed in range(10):
                _, returns, first_optimal = train_grid_world(
                    episodes=episodes,
                    alpha=alpha,
                    epsilon=epsilon,
                    seed=100 + seed,
                )
                if first_optimal is not None:
                    first_episodes.append(first_optimal)
                late_returns.append(statistics.mean(returns[-50:]))
            median_episode = (
                f"{statistics.median(first_episodes):.1f}" if first_episodes else "not reached"
            )
            print(
                f"{epsilon:>7.2f} | {alpha:>5.1f} | "
                f"{len(first_episodes):>14}/10 | {median_episode:>28} | "
                f"{statistics.mean(late_returns):>19.2f}"
            )

    zero_exploration = []
    for seed in range(10):
        q_values, returns, first_optimal = train_grid_world(
            episodes=700,
            alpha=0.1,
            epsilon=0.0,
            seed=200 + seed,
            initial_value=-20.0,
        )
        zero_exploration.append((q_values, returns, first_optimal))
    successes = sum(first is not None for _, _, first in zero_exploration)
    print("\nZero-exploration check with all Q values initialized to -20:")
    print(f"Optimal greedy policies: {successes}/10")
    print(
        "Median first-optimal episode:",
        statistics.median(
            [first for _, _, first in zero_exploration if first is not None]
        ) if successes else "not reached",
    )
    print(
        "Mean return over last 50 episodes:",
        f"{statistics.mean(statistics.mean(returns[-50:]) for _, returns, _ in zero_exploration):.2f}",
    )
    print(
        "With zero exploration, equal initial values and deterministic tie handling can leave "
        "actions untried; negative initialization does not itself guarantee that the start "
        "state escapes that tie."
    )


def run_exercise_2():
    print("\n" + "=" * 72)
    print("EXERCISE 2 - COORDINATION-PENALTY SWEEP")
    print("=" * 72)
    penalties = (0, -20, -50, -100)
    runs = 50
    rounds = 3000
    print("Each cell is count/50 based on the final greedy action pair.")
    print("update | k | optimal joint | safe joint | mismatch")
    print("-" * 59)
    for update in ("average", "best"):
        for penalty in penalties:
            outcomes = {"optimal joint action": 0, "safe joint action": 0, "mismatch": 0}
            for seed in range(runs):
                estimates, _ = train_coordination_game(
                    penalty=penalty,
                    rounds=rounds,
                    seed=1000 + seed,
                    update=update,
                )
                pair = tuple(
                    max(agent_estimates, key=agent_estimates.get)
                    for agent_estimates in estimates
                )
                outcomes[classify_pair(pair)] += 1
            print(
                f"{update:>7} | {penalty:>4} | "
                f"{outcomes['optimal joint action']:>7}/50 | "
                f"{outcomes['safe joint action']:>9}/50 | "
                f"{outcomes['mismatch']:>8}/50"
            )
    print(
        "The best-payoff update keeps a lucky high payoff forever, while the averaging "
        "update lets later penalties reduce an action's estimate; this changes how stable "
        "exploration outcomes are."
    )


if __name__ == "__main__":
    run_algorithm_1()
    run_algorithm_2()
    run_exercise_1()
    run_exercise_2()
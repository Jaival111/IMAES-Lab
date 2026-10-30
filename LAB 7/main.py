# ai401 lab 7: negotiation strategies using fipa-acl

# each acl message stores the information needed to identify a conversation
def send_message(performative, sender, receiver, content, conversation_id, protocol):
    message = {
        "performative": performative,
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "conversation_id": conversation_id,
        "protocol": protocol
    }
    print(message)
    return message


# find the distance between two points on the grid
def distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


# -------------------- algorithm 1: contract net --------------------

agents = {
    "agent1": {"position": (0, 0), "reservation": 12},
    "agent2": {"position": (4, 0), "reservation": 12},
    "agent3": {"position": (0, 4), "reservation": 12},
    "agent4": {"position": (4, 4), "reservation": 12}
}

tasks = {
    "task1": (1, 1),
    "task2": (3, 1),
    "task3": (2, 3)
}

# calculate and display the cost of every agent for every task
cost_matrix = {}
print("\ncontract net: cost matrix")
for agent, details in agents.items():
    cost_matrix[agent] = {}
    for task, position in tasks.items():
        cost_matrix[agent][task] = distance(details["position"], position)
    print(agent, cost_matrix[agent])

print("\ncontract net: message transcript")
free_agents = list(agents.keys())
awards = []
total_paid = 0
message_count = 0

# each agent can win at most one task in this contract net run
for task_number, (task, position) in enumerate(tasks.items(), start=1):
    if not free_agents:
        print("no free agents remain for", task)
        break

    conversation_id = "contract-net-" + str(task_number)
    proposals = {}

    # the manager asks every free agent for a proposal
    for agent in free_agents:
        send_message("cfp", "manager", agent, task, conversation_id, "contract-net")
        message_count += 1

        cost = cost_matrix[agent][task]
        bid = cost + 2

        # an agent refuses if the bid is above its reservation price
        if bid > agents[agent]["reservation"]:
            send_message("refuse", agent, "manager", "reservation price exceeded", conversation_id, "contract-net")
        else:
            send_message("propose", agent, "manager", {"bid": bid, "cost": cost}, conversation_id, "contract-net")
            proposals[agent] = bid
        message_count += 1

    if not proposals:
        print("no proposals received for", task)
        continue

    # the manager accepts the lowest bid and rejects the other proposals
    winner = min(proposals, key=proposals.get)
    winning_bid = proposals[winner]

    for agent in proposals:
        if agent == winner:
            send_message("accept-proposal", "manager", agent, winning_bid, conversation_id, "contract-net")
        else:
            send_message("reject-proposal", "manager", agent, winning_bid, conversation_id, "contract-net")
        message_count += 1

    send_message("inform-done", winner, "manager", task + " completed", conversation_id, "contract-net")
    message_count += 1

    awards.append((task, winner, winning_bid))
    total_paid += winning_bid
    free_agents.remove(winner)

print("\ncontract net: awards")
for task, winner, bid in awards:
    print(task, "->", winner, "| payment =", bid)
print("total paid:", total_paid)
print("messages exchanged:", message_count)


# -------------------- algorithm 2: alternating offers --------------------

def offer(opening_price, reservation_price, round_number, deadline, strategy):
    # boulware concedes late, linear concedes steadily, and conceder concedes early
    if strategy == "boulware":
        exponent = 0.2
    elif strategy == "linear":
        exponent = 1.0
    else:
        exponent = 2.0

    progress = (round_number / deadline) ** (1 / exponent)
    return opening_price + (reservation_price - opening_price) * progress


def bargaining(buyer_strategy, seller_strategy, show_transcript=True):
    buyer_open = 50
    buyer_reservation = 100
    seller_open = 150
    seller_reservation = 90
    deadline = 5
    conversation_id = buyer_strategy + "-" + seller_strategy
    agreement = None
    agreement_round = None

    for round_number in range(1, deadline + 1):
        buyer_offer = offer(buyer_open, buyer_reservation, round_number, deadline, buyer_strategy)
        seller_offer = offer(seller_open, seller_reservation, round_number, deadline, seller_strategy)

        if show_transcript:
            send_message("propose", "buyer", "seller", round(buyer_offer, 2), conversation_id, "alternating-offers")
            send_message("propose", "seller", "buyer", round(seller_offer, 2), conversation_id, "alternating-offers")

        # the buyer accepts if the seller's price is within the buyer's current offer
        if seller_offer <= buyer_offer:
            agreement = seller_offer
            agreement_round = round_number
            if show_transcript:
                send_message("accept-proposal", "buyer", "seller", round(agreement, 2), conversation_id, "alternating-offers")
            break

        # the seller accepts if the buyer's offer meets or exceeds the seller's current price
        if buyer_offer >= seller_offer:
            agreement = buyer_offer
            agreement_round = round_number
            if show_transcript:
                send_message("accept-proposal", "seller", "buyer", round(agreement, 2), conversation_id, "alternating-offers")
            break

    if agreement is None:
        if show_transcript:
            send_message("cancel", "buyer", "seller", "deadline reached without agreement", conversation_id, "alternating-offers")
        return None, None, None, None

    # buyer utility is its reservation price minus the price paid
    buyer_utility = buyer_reservation - agreement
    # seller utility is the price received minus its reservation price
    seller_utility = agreement - seller_reservation
    return round(agreement, 2), agreement_round, round(buyer_utility, 2), round(seller_utility, 2)


print("\nalternating offers: one example transcript")
result = bargaining("linear", "conceder")
if result[0] is None:
    print("no agreement before the deadline")
else:
    print("agreed price:", result[0])
    print("round of agreement:", result[1])
    print("buyer utility:", result[2])
    print("seller utility:", result[3])


# -------------------- exercise 1: all nine strategy pairings --------------------

print("\nexercise 1: strategy comparison")
strategies = ["boulware", "linear", "conceder"]
results = []

for buyer_strategy in strategies:
    for seller_strategy in strategies:
        result = bargaining(buyer_strategy, seller_strategy, False)
        results.append((buyer_strategy, seller_strategy, result))
        print(buyer_strategy, "vs", seller_strategy, "->", result)

agreements = [item for item in results if item[2][0] is not None]
if agreements:
    earliest = min(agreements, key=lambda item: item[2][1])
    lopsided = max(agreements, key=lambda item: abs(item[2][2] - item[2][3]))
    print("earliest agreement:", earliest[0], "vs", earliest[1], "in round", earliest[2][1])
    print("most lopsided utility split:", lopsided[0], "vs", lopsided[1])
else:
    print("no strategy pairing reached an agreement")

boulware_result = bargaining("boulware", "boulware", False)
print("two boulware agents:", boulware_result)


# -------------------- exercise 2: contractor markup experiment --------------------

print("\nexercise 2: contractor markup experiment")
markup_agents = {
    "contractor1": (0, 0),
    "contractor2": (4, 0),
    "contractor3": (0, 4)
}
markup_tasks = {
    "task1": (1, 1),
    "task2": (3, 1),
    "task3": (2, 3)
}
markup_factors = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

for factor in markup_factors:
    won_tasks = []
    total_payment = 0
    contractor_profit = 0

    # all contractors can bid on every task in this experiment
    for task, task_position in markup_tasks.items():
        bids = {}
        costs = {}

        for contractor, contractor_position in markup_agents.items():
            cost = distance(contractor_position, task_position)
            costs[contractor] = cost

            # only contractor1 changes its bid using the markup factor
            if contractor == "contractor1":
                bids[contractor] = cost * factor
            else:
                bids[contractor] = cost

        winner = min(bids, key=bids.get)
        winning_bid = bids[winner]
        total_payment += winning_bid

        if winner == "contractor1":
            won_tasks.append(task)
            contractor_profit += winning_bid - costs[winner]

    print("markup:", factor,
          "| tasks won:", won_tasks,
          "| tasks served:", len(won_tasks),
          "| manager pays:", round(total_payment, 2),
          "| contractor1 profit:", round(contractor_profit, 2))
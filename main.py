import random

class Tileworld:
    def __init__(self, size=15, hole_prob=0.1, max_lifetime=20):
        self.size = size
        self.hole_prob = hole_prob
        self.max_lifetime = max_lifetime
        self.holes = {} # Dictionary mapping (x, y) to remaining lifetime
        self.holes_appeared = 0
        self.holes_filled = 0

    def tick(self):
        # Decrease lifetime of existing holes
        expired = []
        for pos in self.holes:
            self.holes[pos] -= 1
            if self.holes[pos] <= 0:
                expired.append(pos)
        
        for pos in expired:
            del self.holes[pos]

        # Randomly generate new holes
        if random.random() < self.hole_prob:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) not in self.holes:
                self.holes[(x, y)] = random.randint(10, self.max_lifetime)
                self.holes_appeared += 1

class BDIAgent:
    def __init__(self, world, gamma):
        self.world = world
        self.gamma = gamma # Reconsideration interval
        self.pos = (7, 7)  # Start in the middle of the 15x15 grid
        self.beliefs = {}
        self.intention = None
        self.plan = []
        self.s = 0 # Actions executed since last reconsideration

    def brf(self):
        # Update beliefs about the world
        self.beliefs = self.world.holes.copy()

    def options(self):
        # Desires are the currently known holes
        return list(self.beliefs.keys())

    def filter(self, desires):
        # Commit to the closest hole (Manhattan distance)
        if not desires:
            return None
        
        closest_hole = None
        min_dist = float('inf')
        for hole in desires:
            dist = abs(self.pos[0] - hole[0]) + abs(self.pos[1] - hole[1])
            if dist < min_dist:
                min_dist = dist
                closest_hole = hole
                
        return closest_hole

    def generate_plan(self):
        # Means-ends reasoning: path to intention
        if not self.intention:
            return []
        
        path = []
        curr_x, curr_y = self.pos
        target_x, target_y = self.intention
        
        while curr_x != target_x or curr_y != target_y:
            if curr_x < target_x:
                curr_x += 1
            elif curr_x > target_x:
                curr_x -= 1
            elif curr_y < target_y:
                curr_y += 1
            elif curr_y > target_y:
                curr_y -= 1
            path.append((curr_x, curr_y))
            
        return path

    def run_step(self, dynamism):
        # Advance world by dynamism ticks
        for _ in range(dynamism):
            self.world.tick()

        # Step 2: Revise beliefs
        self.brf()

        # Step 3: Initial commitment
        if self.intention is None or not self.plan:
            desires = self.options()
            self.intention = self.filter(desires)
            self.plan = self.generate_plan()
            if not desires:
                return # Take no action, let world advance

        # Step 4: Execute action
        if self.plan:
            self.pos = self.plan.pop(0)
            self.s += 1
            
            # Check if hole is filled
            if self.pos == self.intention and self.intention in self.world.holes:
                print(f"Event: ACHIEVED intention {self.intention}")
                del self.world.holes[self.intention]
                self.world.holes_filled += 1
                self.intention = None
                self.plan = []
                self.s = 0
                return

        # Step 5 & 6 (Algorithm 2): Intention reconsideration
        if self.s <= self.gamma:
            return # Stay committed
            
        # Reconsideration
        self.s = 0
        # Charge one time step for deliberation
        for _ in range(dynamism):
            self.world.tick()
            
        self.brf()
        
        # Check if intention is unachievable
        if self.intention not in self.beliefs:
            print(f"Event: DROPPED unachievable intention {self.intention}")
            self.intention = None
            self.plan = []
            return
            
        # Check for better options
        desires = self.options()
        preferred = self.filter(desires)
        if preferred != self.intention:
            print(f"Event: SWITCHED intention from {self.intention} to {preferred}")
            self.intention = preferred
            self.plan = self.generate_plan()

# Simulation Setup
if __name__ == "__main__":
    world = Tileworld()
    
    # Configure variables here for Exercise 1
    gamma = 2 
    dynamism = 2 # World ticks per agent action
    agent_steps = 600
    
    agent = BDIAgent(world, gamma)
    
    print(f"Starting simulation with Gamma={gamma}, Dynamism={dynamism}")
    for step in range(agent_steps):
        agent.run_step(dynamism)
        
    effectiveness = (world.holes_filled / world.holes_appeared) if world.holes_appeared > 0 else 0
    print("\n--- Final Results ---")
    print(f"Holes Appeared: {world.holes_appeared}")
    print(f"Holes Filled: {world.holes_filled}")
    print(f"Effectiveness: {effectiveness:.2%}")
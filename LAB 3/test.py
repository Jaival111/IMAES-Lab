import random
import matplotlib.pyplot as plt

class Tileworld:
    def __init__(self, size=15, hole_prob=0.1, max_lifetime=20):
        self.size = size
        self.hole_prob = hole_prob
        self.max_lifetime = max_lifetime
        self.holes = {} 
        self.holes_appeared = 0
        self.holes_filled = 0

    def tick(self):
        expired = []
        for pos in self.holes:
            self.holes[pos] -= 1
            if self.holes[pos] <= 0:
                expired.append(pos)
        
        for pos in expired:
            del self.holes[pos]

        if random.random() < self.hole_prob:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) not in self.holes:
                self.holes[(x, y)] = random.randint(10, self.max_lifetime)
                self.holes_appeared += 1

class BDIAgent:
    def __init__(self, world, gamma):
        self.world = world
        self.gamma = gamma 
        self.pos = (7, 7)  
        self.beliefs = {}
        self.intention = None
        self.plan = []
        self.s = 0 

    def brf(self):
        self.beliefs = self.world.holes.copy()

    def options(self):
        return list(self.beliefs.keys())

    def filter(self, desires):
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
        if not self.intention:
            return []
        path = []
        curr_x, curr_y = self.pos
        target_x, target_y = self.intention
        while curr_x != target_x or curr_y != target_y:
            if curr_x < target_x: curr_x += 1
            elif curr_x > target_x: curr_x -= 1
            elif curr_y < target_y: curr_y += 1
            elif curr_y > target_y: curr_y -= 1
            path.append((curr_x, curr_y))
        return path

    def run_step(self, dynamism):
        for _ in range(dynamism):
            self.world.tick()

        self.brf()

        if self.intention is None or not self.plan:
            desires = self.options()
            self.intention = self.filter(desires)
            self.plan = self.generate_plan()
            if not desires:
                return 

        if self.plan:
            self.pos = self.plan.pop(0)
            self.s += 1
            if self.pos == self.intention and self.intention in self.world.holes:
                del self.world.holes[self.intention]
                self.world.holes_filled += 1
                self.intention = None
                self.plan = []
                self.s = 0
                return

        if self.s <= self.gamma:
            return 
            
        self.s = 0
        for _ in range(dynamism):
            self.world.tick()
            
        self.brf()
        
        if self.intention not in self.beliefs:
            self.intention = None
            self.plan = []
            return
            
        desires = self.options()
        preferred = self.filter(desires)
        if preferred != self.intention:
            self.intention = preferred
            self.plan = self.generate_plan()

if __name__ == "__main__":
    gammas = [1, 2, 4, 8, 100] # 100 acts as 'bold'
    gamma_labels = ['1', '2', '4', '8', 'Bold']
    dynamisms = [1, 2, 4, 8]
    seeds = 25
    steps = 600

    results = {d: [] for d in dynamisms}

    print("Running simulations. This may take a few moments...\n")
    for d in dynamisms:
        for g in gammas:
            total_effectiveness = 0
            total_appeared = 0
            total_filled = 0
            
            for seed in range(seeds):
                random.seed(seed)
                world = Tileworld()
                agent = BDIAgent(world, g)
                
                for _ in range(steps):
                    agent.run_step(d)
                    
                eff = (world.holes_filled / world.holes_appeared) if world.holes_appeared > 0 else 0
                total_effectiveness += eff
                total_appeared += world.holes_appeared
                total_filled += world.holes_filled
                
            avg_eff = total_effectiveness / seeds
            avg_appeared = total_appeared / seeds
            avg_filled = total_filled / seeds
            
            results[d].append(avg_eff)
            print(f"Dynamism: {d}, Gamma: {g} -> Avg Holes Appeared: {avg_appeared:.1f}, Avg Holes Filled: {avg_filled:.1f}, Avg Effectiveness: {avg_eff:.2%}")
        print("-" * 80)

    # Generate and save the plot
    plt.figure(figsize=(10, 6))
    
    for d in dynamisms:
        plt.plot(gamma_labels, results[d], marker='o', label=f'World Speed = {d}')

    plt.title('Agent Effectiveness vs. Commitment Strategy (Gamma)')
    plt.xlabel('Gamma (Reconsideration Interval)')
    plt.ylabel('Average Effectiveness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    output_filename = "bdi_effectiveness_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved as '{output_filename}' in your current directory.")
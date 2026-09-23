import numpy as np


def compute_cost(
    ej_mhz: float, 
    ec_mhz: float, 
    ej_target: float = 20000.0, 
    ec_target: float = 320.0
) -> float:
    """
    Cost function with quadratic boundary penalties for:
      - 15000 MHz < Ej < 30000 MHz
      - 300 MHz < Ec < 500 MHz
      - Transmon regime protection (Ej/Ec >= 40)
    """
    # 1. Target Tracking Loss (Normalized MSE)
    loss_target = 0.5 * ((ej_mhz - ej_target) / ej_target)**2 + \
                  0.5 * ((ec_mhz - ec_target) / ec_target)**2

    # 2. Hard Boundary Penalties
    pen_ej = max(0.0, (15000.0 - ej_mhz) / 15000.0)**2 + max(0.0, (ej_mhz - 30000.0) / 30000.0)**2
    pen_ec = max(0.0, (300.0 - ec_mhz) / 300.0)**2 + max(0.0, (ec_mhz - 500.0) / 500.0)**2

    # 3. Charge Dispersion Barrier (Ej/Ec ratio)
    ratio = ej_mhz / max(1e-3, ec_mhz)
    pen_ratio = max(0.0, (40.0 - ratio) / 40.0)**2

    # Penalty weights
    lambda_ej = 1000.0
    lambda_ec = 1000.0
    lambda_ratio = 500.0

    return float(loss_target + (lambda_ej * pen_ej) + (lambda_ec * pen_ec) + (lambda_ratio * pen_ratio))


def select_parents(population: np.ndarray, costs: np.ndarray, tournament_size: int = 3) -> np.ndarray:
    """Tournament selection for continuous real-valued populations."""
    pop_size = len(population)
    if pop_size == 0 or len(costs) != pop_size:
        raise ValueError("population and costs must be non-empty and have matching lengths")
    if not 2 <= tournament_size <= pop_size:
        raise ValueError("tournament_size must be between 2 and population size")
    selected = []
    for _ in range(pop_size):
        aspirant_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
        winner_idx = aspirant_indices[np.argmin(costs[aspirant_indices])]
        selected.append(population[winner_idx])
    return np.array(selected)


def crossover_sbx(p1: np.ndarray, p2: np.ndarray, eta_c: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) for continuous geometric search spaces."""
    rand = np.random.rand(*p1.shape)
    gamma = np.where(
        rand <= 0.5,
        (2.0 * rand)**(1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - rand)))**(1.0 / (eta_c + 1.0))
    )
    child1 = 0.5 * ((1.0 + gamma) * p1 + (1.0 - gamma) * p2)
    child2 = 0.5 * ((1.0 - gamma) * p1 + (1.0 + gamma) * p2)
    return child1, child2


def mutate_polynomial(child: np.ndarray, bounds: np.ndarray, rate: float = 0.15, eta_m: float = 20.0) -> np.ndarray:
    """Polynomial mutation operator respecting hard fabrication parameter limits."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("mutation rate must be between 0 and 1")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each bound must have a greater upper than lower limit")
    low, high = bounds[:, 0], bounds[:, 1]
    mutated = np.clip(child.copy(), low, high)
    
    for i in range(len(child)):
        if np.random.rand() < rate:
            delta_1 = (child[i] - low[i]) / (high[i] - low[i])
            delta_2 = (high[i] - child[i]) / (high[i] - low[i])
            rand = np.random.rand()
            
            if rand <= 0.5:
                delta_q = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta_1)**(eta_m + 1.0))**(1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta_2)**(eta_m + 1.0))**(1.0 / (eta_m + 1.0))
                
            mutated[i] += delta_q * (high[i] - low[i])
            
    return np.clip(mutated, low, high)


def produce_next_generation(
    population: np.ndarray, 
    costs: np.ndarray, 
    bounds: np.ndarray, 
    n_elites: int = 2,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.15
) -> np.ndarray:
    """Executes elitism retention, parent selection, crossover, and mutation."""
    if len(population) < 2 or len(costs) != len(population):
        raise ValueError("population and costs must have matching lengths and at least two members")
    if not 0 <= n_elites < len(population):
        raise ValueError("n_elites must be less than population size")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover rate must be between 0 and 1")
    # 1. Elitism: Retain the best chromosomes directly
    elites = population[np.argsort(costs)[:n_elites]].copy()

    # 2. Selection
    parents = select_parents(population, costs)

    # 3. Offspring generation
    offspring = []
    needed_offspring = len(population) - n_elites
    
    for i in range(0, needed_offspring, 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % len(parents)]

        if np.random.rand() < crossover_rate:
            c1, c2 = crossover_sbx(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        c1 = np.clip(c1, bounds[:, 0], bounds[:, 1])
        c2 = np.clip(c2, bounds[:, 0], bounds[:, 1])
        c1 = mutate_polynomial(c1, bounds, rate=mutation_rate)
        c2 = mutate_polynomial(c2, bounds, rate=mutation_rate)

        offspring.append(c1)
        if len(offspring) < needed_offspring:
            offspring.append(c2)

    return np.vstack([elites, np.array(offspring)])

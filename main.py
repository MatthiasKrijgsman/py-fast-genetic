import numpy as np


def create_pool(*, size, gene_length):
    return np.zeros((size, gene_length))


def add_random_normal(a, mu, sigma):
    a += np.random.normal(mu, sigma, size=a.shape)
    return a


def get_fitness_vector(pool, func):
    return np.apply_along_axis(func, axis=1, arr=pool)


def get_sorted_mask(v, ascending=True):
    if not ascending:
        return np.argsort(v)[::-1]

    return np.argsort(v)


def get_child_indices(fitness_sorted_mask, elimination_rate):
    midpoint = len(fitness_sorted_mask) - round(elimination_rate * len(fitness_sorted_mask))
    return fitness_sorted_mask[midpoint:]


def get_parent_indices(fitness_sorted_mask, elimination_rate):
    midpoint = len(fitness_sorted_mask) - round(elimination_rate * len(fitness_sorted_mask))
    return fitness_sorted_mask[:midpoint]


def mutate(v, mutation_rate, mu, sigma):
    v += np.where(
        np.random.random(v.shape) < mutation_rate,
        np.random.normal(mu, sigma, size=v.shape),
        np.zeros_like(v)
    )
    return v


def crossover(p_1, p_2):
    return np.where(
        np.random.random(p_1.shape) < 0.5,
        p_1, p_2
    )


def next_generation(pool, eval_func, elim_rate, mut_rate, mut_mu=0, mut_sigma=1):
    fitness = get_fitness_vector(pool, eval_func)
    fitness_sorted_mask = get_sorted_mask(fitness, ascending=False)

    child_i = get_child_indices(fitness_sorted_mask, elim_rate)
    parent_i = get_parent_indices(fitness_sorted_mask, elim_rate)

    parents = np.random.choice(parent_i, size=child_i.shape[0] * 2).reshape(child_i.shape[0], 2)

    for i, p in enumerate(parents):
        pool[child_i[i]] = mutate(crossover(pool[p[0]], pool[p[1]]), mut_rate, mut_mu, mut_sigma)

    return pool, fitness, fitness_sorted_mask

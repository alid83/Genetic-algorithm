import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_frame = pd.read_excel('Dry_Bean_Dataset.xlsx')
data_frame = data_frame.dropna()
label = LabelEncoder()
data_frame['Class'] = label.fit_transform(data_frame['Class'])

X_data = data_frame.drop('Class', axis=1)
y_data = data_frame['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

pop_Size = 100
mutation_rate = 0.1
generations = 200


def fitness_func(chromosome, x):
    distances = np.linalg.norm(x[:, np.newaxis] - chromosome, axis=2)
    labels = np.argmin(distances, axis=1)
    if len(np.unique(labels)) > 1:
        fitness = silhouette_score(x, labels)
    else:
        fitness = 0

    return fitness


def mutation(child, mut):
    random_values = np.random.rand(len(child))
    mutation_mask = random_values < mut
    child[mutation_mask] = np.random.rand(np.sum(mutation_mask))
    return child


def crossover(parent_1, parent_2):
    crossover_point = np.random.randint(1, len(parent_1))
    child1 = np.hstack((parent_1[:crossover_point], parent_2[crossover_point:]))
    child2 = np.hstack((parent_2[:crossover_point], parent_1[crossover_point:]))
    return child1, child2


def selection(pop, fitness):
    selected_chromosomes = np.argsort(fitness)[-len(pop)//2:]
    return pop[selected_chromosomes]


population = np.random.rand(pop_Size, X_scaled.shape[1])


for generation in range(generations):
    fitness_values = np.array([fitness_func(chrom, X_scaled) for chrom in population])
    selected_pop = selection(population, fitness_values)
    next_pop = []

    while len(next_pop) < pop_Size:
        parents = np.random.choice(range(len(selected_pop)), size=2, replace=False)
        parent1, parent2 = selected_pop[parents[0]], selected_pop[parents[1]]
        c1, c2 = crossover(parent1, parent2)
        c1 = mutation(c1, mutation_rate)
        c2 = mutation(c2, mutation_rate)
        next_pop.extend([c1, c2])

    population = np.array(next_pop)

best_chromosome = population[np.argmax([fitness_func(chrom, X_scaled) for chrom in population])]
for i in range(len(best_chromosome)):
    print(f"{i} : {best_chromosome[i]}")

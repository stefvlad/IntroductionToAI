import collections

import numpy as np
import random
import math


def fitness_function(chromosome_my, chromosome_from_population):
    count = 0

    for i in range(len(chromosome_from_population)):
        if chromosome_my[i] != chromosome_from_population[i]:
            count += 1

    return 1 - count / len(chromosome_my)


def fitness_result(population, chromosome_my):
    fitness_product = []

    for i in population:
        num = fitness_function(chromosome_my, i)
        fitness_product.append(pow(num, 1))

    return fitness_product


def calculate_probability_of_occurrence(fitness_list):
    sum = 0.0

    for i in fitness_list:
        sum += i

    for i in range(len(fitness_list)):
        fitness_list[i] = math.floor(fitness_list[i] / sum * 100)

    return fitness_list


def generate_list_of_probabilities(probabilities_list, new_population):
    long_list = []

    for i in range(len(new_population)):
        for j in range(probabilities_list[i]):
            long_list.append(new_population[i])

    return long_list


def crossover(first_parent, second_parent):
    start_to_middle_len = random.randint(1, 5)

    middle_to_end_len = 6 - start_to_middle_len

    child = []

    for i in range(start_to_middle_len):
        child.append(first_parent[i])

    for i in range(middle_to_end_len):
        child.append(second_parent[i + start_to_middle_len])

    return child


def mutation(child):
    for i in range(len(child)):
        if random.random() < 0.18:
            if child[i] == 0:
                # print('0->1 at index: ' + str(i))
                child[i] = 1
                break
            else:
                # print('1->0 at index: ' + str(i))
                child[i] = 0
                break

    return child


def generate_new_population(long_list_loop):
    new_population = []
    for i in range(6):
        first_parent = random.choice(long_list_loop)
        second_parent = random.choice(long_list_loop)

        child = crossover(first_parent, second_parent)
        new_population.append(mutation(child))
    return new_population


def main():
    genes_pool = [0, 1]
    population_size = (6, 6)  # population_size, chromosome_size

    chromosome_my = [1, 0, 0, 1, 1, 0]

    new_population = np.random.choice(genes_pool, size=population_size)

    print("start population:")
    print(*new_population)

    # ----------init----------
    fitness_list = fitness_result(new_population, chromosome_my)  # very first fitness list

    print("very first fitness result:")
    print(fitness_list)

    probabilities_list = calculate_probability_of_occurrence(fitness_list)

    long_list = generate_list_of_probabilities(probabilities_list, new_population)

    print("very first new population:")
    first_new_population = generate_new_population(long_list)  # very first new population

    print(first_new_population)

    print("very first new population fitness result:")
    print(fitness_result(first_new_population, chromosome_my))
    # ----------init----------

    print()

    population_num = 1

    new_population_loop = first_new_population

    # while collections.Counter(fitness_result(new_population_loop, chromosome_my)) != collections.Counter(
    #        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
    while not (any(x == 1.0 for x in fitness_result(new_population_loop, chromosome_my))):
        print("population: " + str(population_num + 1))

        fitness_list_loop = fitness_result(new_population_loop, chromosome_my)

        probabilities_list_loop = calculate_probability_of_occurrence(fitness_list_loop)

        long_list_loop = generate_list_of_probabilities(probabilities_list_loop, new_population_loop)

        new_population_loop = generate_new_population(long_list_loop)
        print(new_population_loop)

        print(fitness_result(new_population_loop, chromosome_my))

        print()
        population_num += 1

    print("population num: " + str(population_num))


if __name__ == '__main__':
    main()

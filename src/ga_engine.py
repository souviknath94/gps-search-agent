"""
An implementation of genetic algorithm to find the shortest path between two nodes in a connected graph structure.
The implementation is specifically designed to suit the needs of finding the shortest path problems.

Author: Souvik Nath
Email: souvik.nath1@gmail.com
"""

import copy
import math
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

from src.models import Chromosome, FitnessFunction


class GeneticAlgorithm:

    def __init__(self,
                 population_size: int,
                 chromosome_specs: Chromosome,
                 fitness_func: FitnessFunction,
                 mutation_rate: float = None):

        self.population_size = population_size
        self.chromosome_specs = chromosome_specs
        self.fitness_func = fitness_func
        self.mutation_rate = mutation_rate

        self._all_valid_gene_pool = None
        self._starting_point_valid_node = None
        self._best_solution = None
        self._best_solutions = None
        self._iteration_history = None
        self.execution_time = None
        self.generation_metadata = None

    @property
    def history(self) -> pd.DataFrame:
        history = pd.DataFrame(self._iteration_history)
        return history

    @property
    def best_solutions(self) -> list:
        best_solutions = []
        for sol in self._best_solutions:
            path = self.chromosome_specs.chromosome_to_path(sol["sol"])
            data = dict(path=path, fitness=sol["fitness"])
            best_solutions.append(data)
        return best_solutions

    @property
    def best_solution(self) -> dict:
        """
        Property to return the best solution
        :return:
        """
        if self._best_solution is None:
            idx = np.argmax([sol["fitness"] for sol in self._best_solutions])
            self._best_solution = self._best_solutions[idx]
            return self._best_solution
        else:
            return self._best_solution

    @staticmethod
    def _store_generation_history(generation: int, generation_metadata: dict) -> dict:
        min_fitness = min(generation_metadata["fitness_scores"])
        max_fitness = max(generation_metadata["fitness_scores"])
        mean_fitness = sum(generation_metadata["fitness_scores"])/len(generation_metadata["fitness_scores"])
        history = dict(generation=generation, fitness_high=max_fitness,
                       fitness_avg=mean_fitness, fitness_low=min_fitness)
        return history

    def _roulette_wheel_selection(self, population: list) -> np.array:
        """
        This is the strategy of selecting two parents for purposes of mating to create children during crossover
        operation. Roulette wheel selection ensures this,
            Say if there is fitness function f(n) -> g(n) + h(n) + luck
            where,
                f(n) = path cost
                h(n) = heuristic cost
                luck := a component of luck introduced where the probability of selection is directly proportional to
                        the contribution of a parents fitness to total fitness of the population
        :return: the selected parent
        """
        fitness_scores = [self.fitness_func(chromosome=chromosome) for chromosome in population]
        total_fitness = sum(fitness_scores)
        prob_of_selection = [fitness_score / total_fitness for fitness_score in fitness_scores]
        parent_idx = np.random.choice(range(len(population)), size=1, p=prob_of_selection)[0]
        parent = population[parent_idx]
        return parent

    def init_population(self, pop_size: int) -> list:
        """
        Initialize a population randomly by controlling the structure of the chromosome:
            after selecting the first starting point, we randomly select neighbors of the previous point and add it
            until a randomly selected size of the chromosome isn't reached

        :param pop_size: size of the population. Preferred values are of the form: 2^n where n belongs to Real Numbers
        :return: list of chromosome. Number of chromosome == population size
        """
        self._all_valid_gene_pool = self.chromosome_specs.gene_space
        if self._starting_point_valid_node is None:
            self._starting_point_valid_node = self.chromosome_specs.nodes_to_int[self.fitness_func.start]

        pop = []
        print("Initializing population...")
        while len(pop) < pop_size:
            chromosome_size = np.random.randint(3, len(self._all_valid_gene_pool))
            chromosome = [self._starting_point_valid_node]
            start_node = self._starting_point_valid_node
            while len(chromosome) < chromosome_size:
                neighbors = self.chromosome_specs.edge_list[start_node]
                random_neighbor = np.random.choice(neighbors, size=1)[0]
                chromosome.append(random_neighbor)
                start_node = random_neighbor

            chromosome = np.array(chromosome)

            if any([(chromosome == member).all() if len(chromosome) == len(member) else False for member in pop]):
                print("Random chromosome already in population. Skipping ...")
                continue
            else:
                pop.append(chromosome)

        return pop

    def _update_edge_list(self, edge_list: dict, node: int) -> dict:
        """
        Update the edge list by removing the previous node from the edge list

        :param edge_list: this is the list of valid edges from each individual node
        :param node: The node to be removed from the neighbors in the edege list
        :return: update edge list
        """
        for edge_, neighbors in edge_list.items():
            if node in neighbors:
                edge_list[edge_] = [i for i in neighbors if i != node]
        return edge_list

    def _find_next_best_edge(self, child: np.array, edge_list, node: int) -> int:
        """
        Given a node, one needs to find the next best node with the least number of active connections.
        This process is crucial during the edge recombination strategy.

        If the current node has no neighbors, please select any node randomly

        :param child: the offspring being created during the crossover process
        :param edge_list: the list of valid paths from every city
        :param node: the current node to be removed
        :return: the next best node to visit given the current node
        """
        if edge_list.get(node, None) is None:
            random_best_choice = np.random.choice([i for i in
                                                   self.chromosome_specs.gene_space if i not in child],
                                                  size=1, replace=False)[0]
            return random_best_choice

        current_neighbors = edge_list[node]
        count_neighbor = 9999
        least_connected_neighbor = None
        for neighbor in current_neighbors:
            if len(edge_list[neighbor]) < count_neighbor:
                count_neighbor = len(edge_list[neighbor])
                least_connected_neighbor = neighbor
            elif len(edge_list[neighbor]) == count_neighbor:
                least_connected_neighbor = np.random.choice([neighbor, least_connected_neighbor], size=1)[0]

        return least_connected_neighbor

    def _edge_recombination_crossover(self, parent1: np.array, parent2: np.array) -> np.array:
        """
        The edge recombination strategy is used in solving shortest path problems using local search algorithms like
        the GA. This algorithm works by finding as many node-connections as possible to generate children. It
        optimizes by selecting the node-connections with the least connected neighbor. The algorithm strives by
        introducing the fewest paths as possible.

        However, the edge recombination strategy works only for fixed sized chromosomes.

        :param parent1: The first parent(an array of nodes representing the path to be taken)
        :param parent2: The second parent(an array of nodes representing the path to be taken)
        :return: child: the offspring created from the parent
        """
        edge_list = copy.copy(self.chromosome_specs.edge_list)
        parents = [parent1, parent2]

        child_len_ = round(np.average([len(parent1), len(parent2)]), 0)
        # child = np.array([self._starting_point_valid_node])
        # remove_node = self._starting_point_valid_node
        random_loc_ = np.random.randint(low=1, high=child_len_-1, size=1)[0]
        rnd_parent_loc_ = np.random.choice([0, 1], size=1)[0]
        rnd_parent_ = parents[rnd_parent_loc_].copy()

        if random_loc_ >= len(rnd_parent_):
            parents.pop(rnd_parent_loc_)
            rnd_parent_ = parents[0]

        child = rnd_parent_[0:random_loc_]
        remove_node = child[-1]

        while len(child) < child_len_:
            edge_list = self._update_edge_list(edge_list=edge_list, node=remove_node)
            next_best_choice = self._find_next_best_edge(child=child, edge_list=edge_list, node=remove_node)
            child = np.append(child, next_best_choice)
            remove_node = next_best_choice

        return child

    @staticmethod
    def _scramble_mutation(chromosome: np.array) -> np.array:
        """
        A subset of the chromosome, in this case the path would be shuffled randomly within a chromosome.

        :param chromosome: the offspring which depicts the path taken to reach the goal
        :return: mutated chromosome
        """
        mutated_chromosome = chromosome.copy()
        start, end = sorted(np.random.randint(2, len(chromosome)-1, size=2))
        subset = mutated_chromosome[start:end].copy()
        np.random.shuffle(subset)  # Shuffles the subset in place
        mutated_chromosome[start:end] = subset
        return mutated_chromosome

    def crossover(self, parent_pair: tuple[np.array, np.array]):
        """
        Crossover operations is performed to create children from a pair of parents. Since this is a problem of the
        nature of TSP(Travelling Salesman Problem) with modifications on: existence of a start and end point, finding
        the shortest path between two nodes in a graph. The crossover strategy to be used will be, Edge Recombination
        Strategy. This technique has been explained in the `_edge_recombination_crossover` method.

        :param parent_pair: the pair of parents from which the child would be created through crossover
        :return: offspring: a chromosome of the child representing a path
        """
        p1, p2 = parent_pair
        # offspring1, offspring2 = self._partially_mapped_crossover(parent1=p1, parent2=p2)
        offspring = self._edge_recombination_crossover(parent1=p1, parent2=p2)
        return offspring

    def mutate(self, generation: list) -> list:
        """
        Perform mutation on x% of the population in any given generation.
        Mutation strategy being followed here is the scramble mutation strategy where a subset of the path
        as represented in the chromosome is shuffled.

        :param generation: list of chromosomes; the current generation of children
        :return: a population of offsprings with some of them with mutated genes
        """
        if (self.mutation_rate == 0) or (self.mutation_rate is None):
            print("Mutation rate is `0`. Skipping mutation")
            return generation
        else:
            num_chromo_to_mut = int(self.mutation_rate * len(generation))
            print(f"Number of chromosomes to mutate: {num_chromo_to_mut}. Number of children: {len(generation)}")
            to_mutate = np.random.choice(range(len(generation)), size=num_chromo_to_mut, replace=False)
            for loc in to_mutate:
                chromo = generation[loc]
                mut_chromo = self._scramble_mutation(chromosome=chromo)
                generation[loc] = mut_chromo
        return generation

    def store_generation_metadata(self,
                                  generation: int,
                                  children: list) -> dict:
        """
        Store generation metadata

        :param generation: the current generation counter
        :param children: the offsprings in the current generation
        :return: a dictionary of generation metadata
        """
        generation_metadata = dict(
            generation_count=generation,
            chromosome_size=[len(child) for child in children],
            fitness_scores=[self.fitness_func(chromosome=child) for child in children],
            chromosomes=[child for child in children]
        )
        return generation_metadata

    def run(self, generations: int) -> dict:
        start = datetime.now()

        population = self.init_population(pop_size=self.population_size)
        prev_children = []
        self.generation_metadata = []
        if self._iteration_history is None:
            self._iteration_history = []

        # store only 25% of the population size's best solutions
        best_solution_metadata = [None] * max(int(0.25 * self.population_size), 10)

        for generation in range(generations):
            parents = []
            children = []

            if generation == 0:
                parents = copy.copy(population)
            else:
                parents = copy.copy(prev_children)

            if generation % 10 == 0:
                print(f"Running for generation: {generation}")

            # selecting a pair of parents for mating
            print(f"Selecting parents for mating")
            pairs = []
            while len(pairs) < len(parents):
                parent1 = self._roulette_wheel_selection(population=parents)
                parent2 = self._roulette_wheel_selection(population=parents)
                pair_ = (parent1, parent2)
                pairs.append(pair_)

            i = 0
            flag_break = False
            offsprings = []
            while len(offsprings) < self.population_size and not flag_break:
                i += 1
                for pair in pairs:
                    p1, p2 = pair
                    offspring = self.crossover(parent_pair=(p1, p2))
                    if not any(np.array_equal(offspring, arr) for arr in offsprings):
                        print(f"Generation: {generation}. Adding Child: {offspring.tolist()} to current generation")
                        offsprings += [offspring]
                    else:
                        if abs(len(offsprings) - i) > 100:
                            print(f"No new offspring found for 10 rounds of crossover. Breaking the loop")
                            flag_break = True

            # selecting best chromosomes from parents and children combined
            # this increases the chances of selecting the fittest individual
            print("Selecting the best chromosomes from parents and children combined")
            i = 0
            flag_break = False
            while len(children) < self.population_size and not flag_break:
                i += 1
                child = self._roulette_wheel_selection(population=parents + offsprings)
                if not any(np.array_equal(child, arr) for arr in children):
                    children += [child]
                else:
                    if abs(len(children) - i) > 10:
                        flag_break = True


            # Start mutation
            print(f"Initializing Mutation... Mutation Rate - {self.mutation_rate}")
            children = self.mutate(generation=children)

            # store performance metadata
            metadata = self.store_generation_metadata(generation=generation, children=children)
            self.generation_metadata.append(metadata)

            # populate the best_solution_metadata
            if all(i is None for i in best_solution_metadata):
                print("Populating the best solutions list")
                chromosomes_w_scores = list(zip(metadata["chromosomes"], metadata["fitness_scores"]))
                chromosomes_w_scores_sorted = sorted(chromosomes_w_scores, key=lambda x: x[1], reverse=True)
                top_n_chromosomes_with_scores = chromosomes_w_scores_sorted[:len(best_solution_metadata)]
                for chromosome, fitness in top_n_chromosomes_with_scores:
                    best_solution_metadata.append({"sol": chromosome, "fitness": fitness})

            # Find the current best solution in the new generation
            best_sol_index = np.argmax(metadata["fitness_scores"])
            current_best_sol = metadata["chromosomes"][best_sol_index]
            current_best_fitness = metadata["fitness_scores"][best_sol_index]

            min_fitness_in_best_solutions = min(sol["fitness"] for sol in best_solution_metadata if sol is not None)

            if current_best_fitness > min_fitness_in_best_solutions:
                solution_exists = any(
                    np.array_equal(sol["sol"], current_best_sol) for sol in best_solution_metadata if sol is not None)
                if not solution_exists:
                    min_fitness_index = np.argmin([sol["fitness"] for sol in best_solution_metadata if sol is not None])
                    best_solution_metadata[min_fitness_index] = {"sol": current_best_sol,
                                                                 "fitness": current_best_fitness}

            # store current generation fitness scores
            generation_performance = self._store_generation_history(generation=generation,
                                                                    generation_metadata=metadata)
            self._iteration_history.append(generation_performance)

            # select from parents + children
            prev_children = children

        # update best solutions
        self._best_solutions = [i for i in best_solution_metadata if i is not None]
        best_sol = self.best_solution

        end = datetime.now()
        self.execution_time = f"{(end-start).microseconds / 1000000}"
        return best_sol


if __name__ == "__main__":
    pass

import numpy as np
import pandas as pd

from src.utils import haversine


class Chromosome:
    """
    This is the chromosome structure to represent any given path in the connected graph.
    Each path is represented as a list of nodes that can traversed from one point to the other.
    Each node is represented as an integer.

    There are two methods:
        - chromosome_to_path: This converts the array of ints depicting a path to a pandas DataFrame where each row
                              corresponds to an edge between the two points along with the edge weight
        - path_to_chromosome: This converts a pandas DataFrame between two points along with the edge weight to an array
                              of integers representing a path from one point to another
    """
    def __init__(self, map_df: pd.DataFrame):
        self.map = map_df

        all_cities = list(set(self.map["source"].tolist() + self.map["destination"].tolist()))

        self.distance_map = self.map.set_index(["source", "destination"]).to_dict()["distance"]
        self.edges = list(self.distance_map.keys())
        self.nodes = all_cities
        self.nodes_to_int = {edge: i + 1 for i, edge in enumerate(self.nodes)}
        self.int_to_nodes = {i: edge for edge, i in self.nodes_to_int.items()}
        self.gene_space = list(self.int_to_nodes)

        city_path_list = map_df.groupby("source").agg({"destination": list}).to_dict()["destination"]
        self.edge_list = dict()
        for edge, neighbors in city_path_list.items():
            self.edge_list[self.nodes_to_int[edge]] = [self.nodes_to_int[edge] for edge in neighbors]

    def path_to_chromosome(self, path: pd.DataFrame) -> np.array:
        """
        helper function to convert a path to a numpy array. Each array will be a collection of integers where
        each integer is a node in the graph.

        :param path: a pandas dataframe of the paths taken and their metadata
        :return: chromosome
        """
        chromosome_ = [self.nodes_to_int[path.iloc[0]['source']]]

        for _, row in path.iterrows():
            destination_int = self.nodes_to_int[row['destination']]
            if destination_int != chromosome_[-1]:  # Check for consecutive repetitions
                chromosome_.append(destination_int)

        return np.array(chromosome_)

    def chromosome_to_path(self, chromosome: np.array) -> pd.DataFrame:
        """
        Convert a numpy array of integers representing the a path in the map to a DataFrame for readability purposes.
        Each row in the dataframe has an edge and the edge weight.

        :param chromosome: array of int representing edges in the map
        :return: path: A pandas dataframe of the edges and the distance between the nodes in each edge
        """
        path = []
        for i, city in enumerate(chromosome):
            if i+1 >= len(chromosome):
                break
            next_city = chromosome[i + 1]
            path_ = (self.int_to_nodes[city], self.int_to_nodes[next_city])
            path.append(path_)

        path_df = pd.DataFrame(path, columns=["source", "destination"])
        # this -100 is a penalty for having an invalid edge connection
        path_df["distance"] = path_df[["source",
                                       "destination"]].apply(lambda x: self.distance_map.get((x["source"],
                                                                                              x["destination"]),
                                                                                             np.nan), axis=1)
        return path_df
    

class FitnessFunction:

    def __init__(self, chromosome_specs: Chromosome, co_ordinates: dict, goal: str, start: str):
        """
        The fitness function object to calculate the final fitness score.
        fitness_function = reward - (path cost + heuristic cost + penalty)

        The fitness function has the following components:
            - reward: If the goal is reached than a reward is introduced to the fitness function calculation
                      This reward is 0 if the goal is not reached else it's a very high number(9999 in this case)
            - penalty: If the path is invalid(has loops and disconnected edges) a penalty is added to the fitness
                       function calculation
            - path cost: This is the cost of moving from one node to another node, considering the node is valid
            - heuristic cost: This is the haversine distance between two points on the surface of the earth. In this
                              case it is computed by taking the haversine distance between the last point on the path
                              or chromosome and the goal.

            If the path cost and heuristic cost are low and the goal has been reached than the fitness function will be
            a high value. In an ideal case, the genetic algorithm should be chasing for the shortest path between the
            start and end points, going through a valid path.

        :param chromosome_specs: The chromosome class
        :param co_ordinates: The lat, long co-ordinates of every node in the graph
        :param goal: the final node to reach the goal
        :param start: the node to find the shortest path to the goal
        """
        self.chromosome_specs = chromosome_specs
        self.co_ordinates = co_ordinates
        self.goal = goal
        self.start = start
        self.goal_coord = self._get_coordinates(city=self.goal)

    def __call__(self, chromosome: np.array):
        return self.calculate_fitness(chromosome_arr=chromosome)

    def _get_coordinates(self, city: str) -> tuple[float, float]:
        """
        Fetch the latitude and longitude of a given path
        :param city:
        :return: tuple of latitude and longitude
        """
        lat = self.co_ordinates["latitude"].get(city, None)
        long = self.co_ordinates["longitude"].get(city, None)
        return lat, long

    @staticmethod
    def _check_for_loops(chromosome: np.array) -> bool:
        """
        Check if a given chromosome (path) contains any loops.

        A loop is defined as any node being revisited non-consecutively in the path,
        indicating a circular path.

        :param chromosome: numpy array of int representing nodes in the path
        :return: True if the path contains loops, False otherwise
        """
        visited = set()
        for i, node in enumerate(chromosome):
            if node in visited:
                # To specifically catch non-consecutive loops, check if the current node appears again in the future,
                # ignoring direct consecutive duplicates which are not loops but rather backtracking.
                if i < len(chromosome) - 1 and chromosome[i + 1] == node:
                    continue
                return True  # A loop is detected
            visited.add(node)
        return False

    def is_valid_path(self, path: pd.DataFrame) -> bool:
        """
        Check if a path is valid. It can so happen that during crossover or mutation certain children chromosome
        are made which are loops or not valid paths
        :param path: a pandas dataframe of a given path
        :return:
        """
        path_list = [(c1, c2) for c1, c2 in zip(path["source"].tolist(), path["destination"].tolist())]
        checks = []

        chromosome = self.chromosome_specs.path_to_chromosome(path=path)
        if self._check_for_loops(chromosome):
            return False

        prev_neighbor = None
        for i, edge in enumerate(path_list):
            if edge not in self.chromosome_specs.edges:
                checks.append(False)

            c1, c2 = edge

            if i == 0:
                pass
            else:
                named_edge_list = [self.chromosome_specs.int_to_nodes[i] for i in
                                   self.chromosome_specs.edge_list[self.chromosome_specs.nodes_to_int[prev_neighbor]]
                                   ]

                if c1 != prev_neighbor:
                    checks.append(False)

                elif c2 in named_edge_list:
                    checks.append(True)

                else:
                    checks.append(False)
            prev_neighbor = c2

        return all(checks)

    def _calc_heuristic_cost(self, path: pd.DataFrame) -> float:
        """
        Calculate haversine distance from the last city in the path to the goal
        :param path: a pandas dataframe of a given path
        :return: the haversine distance between the last node in the path and the goal
        """
        last_location = path.iloc[-1]["destination"]
        coord1 = self._get_coordinates(city=last_location)
        return haversine(coord1=coord1, coord2=self.goal_coord)

    @staticmethod
    def _calc_path_cost(path: pd.DataFrame) -> float:
        """
        Calculate the total distance travelled in the path taken
        :param path: a pandas dataframe of a given path
        :return: total distance travelled
        """
        return path["distance"].sum()

    def goal_test(self, path: pd.DataFrame) -> bool:
        """
        Check if goal has been achieved
        :param path: a pandas dataframe of a given path
        :return: bool if goal is achieved
        """
        last_node = path.iloc[-1]["destination"]
        if last_node == self.goal:
            return True
        else:
            return False

    def calculate_fitness(self, chromosome_arr: np.array) -> float:
        """
        Calculate final fitness value
        :param chromosome_arr: an array of integers where each integer represents a node in the graph and the order
                                represents a path
        :return: total fitness score
        """
        path = self.chromosome_specs.chromosome_to_path(chromosome=chromosome_arr)

        if not self.is_valid_path(path):
            # return a very large negative value if the path is not valid such that it has
            # 1. loops
            # 2. invalid connections
            # 3. incorrect starting point
            # this penalty will be added to the path cost + heuristic cost
            penalty = 9999
        else:
            penalty = 0

        actual_cost = self._calc_path_cost(path=path)
        heuristic_cost = self._calc_heuristic_cost(path=path)

        if self.goal_test(path=path):
            # if goal has been achieved subtract the total cost from the reward
            # this would ensure that even if the path to the goal has been identified
            # it will motivate the algorithm to find an more efficient path as the total fitness would in that case
            # be higher
            reward = 9999
        else:
            reward = 0

        fitness = reward - (actual_cost + heuristic_cost + penalty)
        # since we will use this fitness function for parent selection during cross over using roulette wheel
        # selection. The fitness function scores should all be positive. Thus converting negative fitness scores to
        # a very small positive value between 0 and 1 by inverting the absolute fitness score would be helpful as it
        # would assign very small values to negative fitness functions leading them to occupy only a small piece of
        # the pie
        fitness = 1/abs(fitness) if fitness < 0 else fitness

        return fitness

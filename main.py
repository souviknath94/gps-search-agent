import pandas as pd
from src.utils import DataUtils
from src.models import Chromosome, FitnessFunction
from src.ga_engine import GeneticAlgorithm


def search_best_path_using_ga(start, goal, map_df, meta_data, num_generations):
    # Initialize utilities and data
    data_utils = DataUtils(path=map_df, metadata=meta_data)
    data_utils.run()

    # Define the GA components
    chromosome = Chromosome(map_df=data_utils.path)
    fitness_func = FitnessFunction(chromosome_specs=chromosome, 
                                   co_ordinates=data_utils.lat_long, 
                                   start=start, goal=goal)
    
    # Initialize and run the engine
    ga = GeneticAlgorithm(population_size=64, 
                          chromosome_specs=chromosome, 
                          fitness_func=fitness_func, 
                          mutation_rate=0.05)
    
    best_sol = ga.run(generations=num_generations)
    best_path = chromosome.chromosome_to_path(chromosome=best_sol["sol"])
    
    return ga, best_path

if __name__ == "__main__":
    # Load data and execute
    map_df = pd.read_csv("data/path_map.csv")
    meta_data = pd.read_csv("data/latlong.csv")

    ga, best_path = search_best_path_using_ga(start="Panaji",
                                              goal="Chennai",
                                              map_df=map_df,
                                              meta_data=meta_data,
                                              num_generations=10)
    print(best_path)

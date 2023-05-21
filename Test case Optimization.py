import random
import numpy as np
import pandas as pd

def fitness_function(test_case, dataset):
    # Evaluate the fitness of the test case based on the dataset
    # Return a fitness value (lower is better)

    # Extract the values from the test case
    values = [dataset.iloc[test_case_index, test_case_index+1] for test_case_index in range(len(test_case))]

    # Calculate the fitness value using the equation F = (J - 75) + (K - 65) + (X - 55) + (Y - 45) + (Z - 35)
    fitness_value = sum([value - threshold for value, threshold in zip(values, [75, 65, 55, 45, 35])])

    # Return the fitness value
    return fitness_value


# Artificial Bee Colony (ABC) algorithm
def abc_algorithm(population_size, test_case_length, max_iterations, dataset):
    # Initialize population
    population = []
    for _ in range(population_size):
        test_case = [random.randint(0, len(dataset.columns)-2) for _ in range(test_case_length)]
        population.append(test_case)

    # Main loop
    iteration = 0
    while iteration < max_iterations:
        # Employed bees phase
        fitness_values = [fitness_function(test_case, dataset) for test_case in population]
        probabilities = [1.0 / (fitness_value + 1) for fitness_value in fitness_values]
        probabilities_sum = sum(probabilities)

        if probabilities_sum == 0:
            # Assign equal probabilities if sum is zero
            selection_probabilities = [1.0 / population_size] * population_size
        else:
            # Calculate selection probabilities
            selection_probabilities = [prob / probabilities_sum for prob in probabilities]

        # Select employed bees
        selected_population = []
        for _ in range(population_size):
            selected_index = np.random.choice(range(population_size), p=selection_probabilities)
            selected_population.append(population[selected_index])

        # Onlooker bees phase
        onlooker_population = []
        for i in range(population_size):
            selected_test_case = selected_population[i]
            selected_fitness = fitness_values[i]
            selected_probability = probabilities[i]
            selected_probability_sum = probabilities_sum - selected_probability

            # Calculate onlooker bee probability
            if selected_probability_sum == 0:
                onlooker_probability = 1.0 / population_size
            else:
                onlooker_probability = selected_probability / selected_probability_sum

            # Generate a new test case using onlooker bee strategy
            new_test_case = selected_test_case.copy()
            for j in range(test_case_length):
                if random.random() < onlooker_probability:
                    new_test_case[j] = random.randint(0, len(dataset.columns)-2)  # Assign a random attribute index

            # Evaluate the fitness of the new test case
            new_fitness = fitness_function(new_test_case, dataset)

            # Update onlooker population
            if new_fitness < selected_fitness:
                onlooker_population.append(new_test_case)
            else:
                onlooker_population.append(selected_test_case)

        # Scout bees phase
        scout_bees = []
        for test_case in onlooker_population:
            if random.random() < 0.01:  # Probability of abandoning the solution
                scout_bees.append([random.randint(0, len(dataset.columns)-2) for _ in range(test_case_length)])
            else:
                scout_bees.append(test_case)

        # Update population
        population = scout_bees

        iteration += 1

    # Return the best test case found
    best_test_case = min(population, key=lambda x: fitness_function(x, dataset))
    return best_test_case

# Provide your dataset here
dataset = pd.read_csv('dataset.csv')

# Calculate the population size, test case length, and max iterations
population_size = len(dataset)
test_case_length = len(dataset.columns) - 2  # Exclude the Test Case id and Fitness columns
max_iterations = 100

print("Population Size:", population_size)
print("Test Case Length:", test_case_length)
print("Max Iterations:", max_iterations)
print("Dataset:\n", dataset)

best_test_case = abc_algorithm(population_size, test_case_length, max_iterations, dataset)

# Get the column names or indices corresponding to the best test case
column_names = dataset.columns[1:-1]  # Exclude the first column (Test Case id) and last column (Fitness)
best_test_case_values = [dataset.iloc[best_test_case_index, best_test_case_index+1] for best_test_case_index in best_test_case]
best_test_case_with_columns = list(zip(column_names, best_test_case_values))

print("Best Test Case:")
for column, value in best_test_case_with_columns:
    print(f"{column}: {value}")

print("Fitness:", fitness_function(best_test_case, dataset))
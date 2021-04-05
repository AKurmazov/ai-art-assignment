import os, random, copy, math
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


BASE_IMAGE_SIZE = 256
MAX_LINE_LENGTH = 5.0
POPULATION_SIZE = 8
GENES_SIZE = 800
MUTATION_CHANCE = 0.05
CHANGE_VISABILITY_CHANCE = 0.001

goal_image = cv2.resize(cv2.imread('goal_image.png'), (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE))
edges_image = cv2.Canny(goal_image, 100, 200, apertureSize=3, L2gradient=True)


def compute_similarity_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(gray, edges_image, full=True)
    return score
        

def draw_images(*images):
    for image in images:
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def produce_image(chromosome):
    image = np.zeros((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), np.uint8)
    for gene in chromosome:
        if not gene[4]:
            cv2.line(image, *gene[:2], (255, 255, 255), thickness=1)
    return image


def get_random_coordinates():
    return (random.randint(1, BASE_IMAGE_SIZE), random.randint(1, BASE_IMAGE_SIZE))


def generate_random_gene():
        p1 = get_random_coordinates()
        
        angle = random.uniform(0.0, 1.0) * 2 * math.pi
        length = np.random.exponential(MAX_LINE_LENGTH)

        x_offset = math.floor(math.cos(angle) * length)
        y_offset = math.floor(math.sin(angle) * length)

        p2 = (p1[0] + x_offset, p1[1] + y_offset)

        return (p1, p2, angle, length, False)


def generate_random_chromosome():
    chromosome = []
    for i in range(GENES_SIZE):
        chromosome.append(generate_random_gene())

    image = produce_image(chromosome)
    score = compute_similarity_score(image)

    return [chromosome, score]


def initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(generate_random_chromosome())
    return population


def mating(population):
    def crossover(parent1, parent2):
        child = copy.deepcopy(parent1)
        crossover_point = random.randint(0, GENES_SIZE)
        child[crossover_point:] = copy.deepcopy(parent2[crossover_point:])
        return child

    new_population = []
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                new_population.append(population[i])
            else:
                child = crossover(population[i][0], population[j][0])
                score = compute_similarity_score(produce_image(child))
                new_population.append([child, score])

    return new_population


def mutation(population):
    def mutate(gene):
        new_gene = copy.copy(gene)
        if random.uniform(0.0, 1.0) <= CHANGE_VISABILITY_CHANCE:
            new_gene = (gene[0], gene[1], gene[2], gene[3], not gene[4])
        elif random.uniform(0.0, 1.0) <= MUTATION_CHANCE:
            x = gene[0][0] + (1 if random.random() < 0.5 else -1)
            y = gene[0][1] + (1 if random.random() < 0.5 else -1)

            angle = random.uniform(0.0, 1.0) * 2 * math.pi
            length = gene[3] + (1 if random.random() < 0.5 else -1) * np.random.exponential(5.0)

            x_offset = math.floor(math.cos(angle) * length)
            y_offset = math.floor(math.sin(angle) * length)

            new_gene = ((x, y), (x + x_offset, y + y_offset), angle, length, gene[4])

        return tuple(new_gene)

    for chromosome in population:
        for i in range(len(chromosome[0])):
            if random.uniform(0.0, 1.0) <= MUTATION_CHANCE:
                chromosome[0][i] = mutate(chromosome[0][i])
        chromosome[1] = compute_similarity_score(produce_image(chromosome[0]))
    
    return population


# TODO: In mutation function, add a very low probability of removing a gene DONE
# TODO: Make the line length to be exponentially/geomtrically distributed DONE
# TODO: Make the crossover point be randomly chosen DONE
# TODO: Change gene's parameters to p1, p2, angle, length, visibility DONE

def main():
    cv2.imwrite("edges_image.png", edges_image)

    population = initial_population()
    best_score = 0
    iteration = 0
    while best_score < 0.97:

        population = sorted(population, key=lambda item: item[1], reverse=True)[:POPULATION_SIZE]
        assert len(population) == POPULATION_SIZE

        best_score = population[0][1]
        best_chromosome = population[0][0]

        image = produce_image(best_chromosome)
        cv2.imwrite("out_image.png", image)
        print("Iteration", iteration, "| Best SSIM:", best_score * 100, "%")

        if iteration % 500 == 0:
            cv2.imwrite(f"{iteration}_{math.ceil(best_score*100)}.png", image)
        iteration += 1

        population = mating(population)
        population = mutation(population)
        
    draw_images(goal_image, edges_image)


if __name__ == "__main__":
    main()
import os, random, copy, math
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


BASE_IMAGE_SIZE = 512
MAX_LINE_LENGTH = 15.0
MIN_LINE_LENGTH = 7.0
MAX_LINE_THICKNESS = 1  # max(1, BASE_IMAGE_SIZE // 256)
POPULATION_SIZE = 12
GENES_SIZE = 400
MUTATION_CHANCE = 0.005

goal_image = cv2.resize(cv2.imread('goal_image.png', 0), (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE))
ret, thr = cv2.threshold(goal_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
edges_image = cv2.Canny(goal_image, ret, 0.5 * ret, apertureSize=3, L2gradient=True)

contours, _ = cv2.findContours(edges_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
MINX = MINY = BASE_IMAGE_SIZE
MAXX = MAXY = 0 
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if x < MINX:
        MINX = x
    if x + w > MAXX:
        MAXX = x + w
    if y < MINY:
        MINY = y
    if y + h > MAXY:
        MAXY = y + h


def random_sign():
    return (1 if random.random() < 0.5 else -1)


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
        if not gene[5]:
            cv2.line(image, *gene[:2], (255, 255, 255), thickness=gene[4])
    return image


def get_random_coordinates():
    return (random.randint(MINX, MAXX), random.randint(MINY, MAXY))


def generate_random_gene():
    p1 = get_random_coordinates()
    
    angle = random.uniform(0.0, 1.0) * 2 * math.pi
    length = MIN_LINE_LENGTH + np.random.exponential(MAX_LINE_LENGTH)
    thickness = random.randint(1, MAX_LINE_THICKNESS)

    x_offset = math.floor(math.cos(angle) * length)
    y_offset = math.floor(math.sin(angle) * length)

    p2 = (p1[0] + x_offset, p1[1] + y_offset)

    return (p1, p2, angle, length, thickness, False)


def generate_random_chromosome():
    chromosome = []
    for i in range(GENES_SIZE):
        chromosome.append(generate_random_gene())
    image = produce_image(chromosome)
    return [chromosome, None]


def initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(generate_random_chromosome())
    return population


def mutation_choice(extra=1):
    assert extra >= 0.0 and extra <= 1.0
    return random.uniform(0.0, 1.0) <= MUTATION_CHANCE * extra


def mating(population):
    def crossover(parent1, parent2):
        child = []
        crossover_point = GENES_SIZE // 2  # random.randint(0, GENES_SIZE)
        for i in range(GENES_SIZE):
            if i < crossover_point:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    new_population = []
    for i in range(len(population)):
        for j in range(i, len(population)):
            if i == j:
                new_population.append(population[i])
            else:
                child1 = crossover(population[i][0], population[j][0])
                child2 = crossover(population[j][0], population[i][0])
                new_population.extend([[child1, None], [child2, None]])
    return new_population


def mutation(population):
    def mutate(gene):
        x = gene[0][0] + random_sign() * random.randint(1, 5) if mutation_choice() else gene[0][0]
        y = gene[0][1] + random_sign() * random.randint(1, 5) if mutation_choice() else gene[0][1]

        angle = gene[2] + random_sign() * random.uniform(0.0, 0.2) * math.pi if mutation_choice() else gene[2]
        length = max(MIN_LINE_LENGTH, gene[3] + random_sign() * np.random.exponential(5.0)) if mutation_choice() else gene[3]
        thickness = random.randint(1, MAX_LINE_THICKNESS)
        visability = not gene[5] if mutation_choice(extra=0.01) else gene[5]

        x_offset = math.floor(math.cos(angle) * length)
        y_offset = math.floor(math.sin(angle) * length)

        new_gene = ((x, y), (x + x_offset, y + y_offset), angle, length, thickness, visability)
        return tuple(new_gene)

    for chromosome in population:
        for i in range(len(chromosome[0])):
            chromosome[0][i] = mutate(chromosome[0][i])
    return population


# TODO: In mutation function, add a very low probability of removing a gene DONE
# TODO: Make the line length to be exponentially/geometrically distributed DONE
# TODO: Make the crossover point be randomly chosen DONE
# TODO: Change gene's parameters to p1, p2, angle, length, visibility DONE
# TODO: Change visability to be set by color (0, 0, 0) <-> (255, 255, 255)

import time

def main():
    from multiprocessing import Pool
    pool = Pool()

    cv2.imwrite("edges_image1.png", edges_image)

    population = initial_population()
    best_score = 0
    iteration = 0
    while best_score < 0.97:

        # Compute fitness
        start = time.time()

        results = [None] * len(population)
        for i in range(len(population)):
            _image = produce_image(population[i][0])
            results[i] = pool.apply_async(compute_similarity_score, [_image])
    
        assert len(results) == len(population)
        for i in range(len(results)):
            population[i][1] = results[i].get(timeout=10)

        end = time.time()
        print(end - start, 'seconds')

        # Selection
        population = sorted(population, key=lambda item: item[1], reverse=True)[:POPULATION_SIZE]
        assert len(population) == POPULATION_SIZE

        best_score = population[0][1]
        best_chromosome = population[0][0]

        image = produce_image(best_chromosome)
        cv2.imwrite("out_image1.png", image)
        print("Iteration", iteration, "| Best SSIM:", best_score * 100, "%")

        if iteration % 500 == 0:
            cv2.imwrite(f"t_{iteration}_{math.ceil(best_score*100)}.png", image)
        iteration += 1

        # Evolution
        population = mating(population)
        population = mutation(population)
        
    draw_images(goal_image, edges_image)


if __name__ == "__main__":
    main()
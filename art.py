import os, random, copy, math
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


# Algorithm's parameters
BASE_IMAGE_SIZE = 512
MAX_LINE_LENGTH = 13.0
MIN_LINE_LENGTH = 5.0
MAX_LINE_THICKNESS = 1
POPULATION_SIZE = 6
GENES_SIZE = 1000
MUTATION_CHANCE = 0.005

# Read and resize the source image
goal_image = cv2.resize(cv2.imread('contest1.jpg'), (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE))

# Blur the image via noice filtering
kernel = np.ones((5, 5), np.float32) / 25
goal_image = cv2.filter2D(goal_image, -1, kernel)

# Get the goal image via Canny edge detection
edges_image = cv2.Canny(goal_image, 80, 110, apertureSize=3, L2gradient=False)

# Determine the contours of the objects inside the goal image
contours, _ = cv2.findContours(edges_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
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
    """ Randomly generates sign (i.e., -1 or 1) with equal probability """

    return (1 if random.random() < 0.5 else -1)


def compute_similarity_score(image):
    """ Computes the similarity score of the given image with the goal image """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(gray, edges_image, full=True)
    return score
        

def draw_images(*images):
    """ Draws the given images """

    for image in images:
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def produce_image(chromosome):
    """ Produces an image from the given chromosome """

    image = np.zeros((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), np.uint8) # Take the black image as a background
    for gene in chromosome:
        if not gene[5]:
            cv2.line(image, *gene[:2], (255, 255, 255), thickness=gene[4]) # Draw the lines onto the image
    return image


def get_random_coordinates():
    """ Generates a random pair of coordinates within the contours """

    return (random.randint(MINX, MAXX), random.randint(MINY, MAXY))


def generate_random_gene():
    """ Generates a random gene """

    p1 = get_random_coordinates()

    # Set visibility to False if position is far from good
    hidden = True
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if p1[0] >= x and p1[0] <= x + w and p1[1] >= y and p1[1] <= y + h:
            hidden = False
            break

    angle = random.uniform(0.0, 1.0) * 2 * math.pi
    length = MIN_LINE_LENGTH + np.random.exponential(MAX_LINE_LENGTH)
    thickness = random.randint(1, MAX_LINE_THICKNESS)

    # Calculate the coordinates of where the line ends
    x_offset = math.floor(math.cos(angle) * length)
    y_offset = math.floor(math.sin(angle) * length)
    p2 = (p1[0] + x_offset, p1[1] + y_offset)

    return (p1, p2, angle, length, thickness, hidden)


def generate_random_chromosome():
    """ Generates random chromosome """

    chromosome = []
    for i in range(GENES_SIZE):
        chromosome.append(generate_random_gene())
    return [chromosome, None]


def initial_population():
    """ Generates initial population """"

    population = []
    for i in range(POPULATION_SIZE):
        population.append(generate_random_chromosome())
    return population


def mutation_choice(extra=1):
    """ Decides whether gene's parameter should mutate """

    assert extra >= 0.0 and extra <= 1.0
    return random.uniform(0.0, 1.0) <= MUTATION_CHANCE * extra


def mating(population):
    """ Performs crossover over the population """

    def crossover(parent1, parent2):
        """ Performs crossover over two parents via single point""" 

        child = []
        crossover_point = random.randint(0, GENES_SIZE)
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
    """ Perfroms mutation over the population """

    def mutate(gene):
        """ Performs mutation over parameters of the given gene """"

        x = gene[0][0] + random_sign() * random.randint(1, 5) if mutation_choice() else gene[0][0]
        y = gene[0][1] + random_sign() * random.randint(1, 5) if mutation_choice() else gene[0][1]

        angle = gene[2] + random_sign() * random.uniform(0.0, 0.2) * math.pi if mutation_choice() else gene[2]
        length = max(MIN_LINE_LENGTH, gene[3] + random_sign() * np.random.exponential(5.0)) if mutation_choice() else gene[3]
        thickness = random.randint(1, MAX_LINE_THICKNESS)
        hidden = not gene[5] if mutation_choice(extra=0.01) else gene[5]

        x_offset = math.floor(math.cos(angle) * length)
        y_offset = math.floor(math.sin(angle) * length)

        new_gene = ((x, y), (x + x_offset, y + y_offset), angle, length, thickness, hidden)
        return tuple(new_gene)

    for chromosome in population:
        for i in range(len(chromosome[0])):
            chromosome[0][i] = mutate(chromosome[0][i])
    return population


# TODO: In mutation function, add a very low probability of removing a gene DONE
# TODO: Make the line length to be exponentially/geometrically distributed DONE
# TODO: Make the crossover point be randomly chosen DONE
# TODO: Change gene's parameters to p1, p2, angle, length, visibility DONE

def main():
    from multiprocessing import Pool
    pool = Pool() # Initialize pool

    cv2.imwrite("edges_image.png", edges_image)

    # Set initial conditions
    population = initial_population()
    best_score = 0
    iteration = 0

    # Run the main loop
    while best_score < 0.97:

        # Compute fitness using the concept of multiprocessing
        results = [None] * len(population)
        for i in range(len(population)):
            _image = produce_image(population[i][0])
            results[i] = pool.apply_async(compute_similarity_score, [_image])
    
        assert len(results) == len(population)
        for i in range(len(results)):
            population[i][1] = results[i].get(timeout=10)

        # Selection
        population = sorted(population, key=lambda item: item[1], reverse=True)[:POPULATION_SIZE]
        assert len(population) == POPULATION_SIZE

        # Choose the best chromosome of the current population
        best_score = population[0][1]
        best_chromosome = population[0][0]

        # Save the best chromosome
        image = produce_image(best_chromosome)
        cv2.imwrite("out_image.png", image)
        print("Iteration", iteration, "| SSIM:", best_score * 100, "%")

        if iteration % 500 == 0:
            cv2.imwrite(f"t_{iteration}_{math.ceil(best_score*100)}.png", image)
        iteration += 1

        # Crossoover and mutation
        population = mating(population)
        population = mutation(population)
        
    draw_images(goal_image, edges_image)


if __name__ == "__main__":
    main()
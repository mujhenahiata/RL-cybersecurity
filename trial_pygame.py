import pygame
import numpy as np
import random
import time

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
NODE_SIZE = 50
EDGE_WIDTH = 2
FPS = 30
ATTACK_REWARD = -1
DEFEND_REWARD = 1
NUM_NODES = 8
ACTION_DELAY = 1  # Seconds between actions

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Attacker vs Defender Network Simulation")

# Load sprites
node_img = pygame.image.load("node.png").convert_alpha()
attacker_img = pygame.image.load("attacker.png").convert_alpha()
defender_img = pygame.image.load("defender.png").convert_alpha()

# Scale images
node_img = pygame.transform.scale(node_img, (NODE_SIZE, NODE_SIZE))
attacker_img = pygame.transform.scale(attacker_img, (50, 50))
defender_img = pygame.transform.scale(defender_img, (50, 50))

# Generate nodes
nodes = [
    (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))
    for _ in range(NUM_NODES)
]
edges = [(i, j) for i in range(NUM_NODES) for j in range(i + 1, NUM_NODES) if random.random() > 0.5]

# Initialize Q-learning variables
q_table = np.zeros((NUM_NODES, NUM_NODES))
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

# Game variables
attacker_pos = random.randint(0, NUM_NODES - 1)
defender_pos = random.randint(0, NUM_NODES - 1)
attacked_nodes = set()
saved_nodes = set()
move_count = 0
start_time = time.time()

# Functions
def draw_network(message=""):
    """Draw the network of nodes, edges, and current game state."""
    screen.fill(WHITE)

    # Draw edges
    for edge in edges:
        pygame.draw.line(screen, BLACK, nodes[edge[0]], nodes[edge[1]], EDGE_WIDTH)

    # Draw nodes
    for i, node in enumerate(nodes):
        if i in attacked_nodes:
            color = RED  # Attacked
        elif i in saved_nodes:
            color = GREEN  # Saved
        else:
            color = BLUE  # Neutral
        pygame.draw.circle(screen, color, node, NODE_SIZE // 2, 3)
        screen.blit(node_img, (node[0] - NODE_SIZE // 2, node[1] - NODE_SIZE // 2))

    # Draw attacker and defender
    attacker_node = nodes[attacker_pos]
    defender_node = nodes[defender_pos]
    screen.blit(attacker_img, (attacker_node[0] - 25, attacker_node[1] - 25))
    screen.blit(defender_img, (defender_node[0] - 25, defender_node[1] - 25))

    # Display message
    font = pygame.font.Font(None, 36)
    text = font.render(message, True, BLACK)
    screen.blit(text, (20, 20))

    pygame.display.flip()

def q_learning_action(state):
    """Decide the defender's action using Q-learning."""
    if random.random() < epsilon:  # Exploration
        return random.choice(range(NUM_NODES))
    else:  # Exploitation
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    """Update the Q-table based on the action taken."""
    best_next_action = np.max(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * best_next_action - q_table[state, action])

def reset_game():
    """Reset the game variables after a successful capture."""
    global attacker_pos, defender_pos, attacked_nodes, saved_nodes, move_count, start_time
    attacker_pos = random.randint(0, NUM_NODES - 1)
    defender_pos = random.randint(0, NUM_NODES - 1)
    attacked_nodes = set()
    saved_nodes = set()
    move_count = 0
    start_time = time.time()

# Main loop
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Attacker moves randomly
    attacker_prev = attacker_pos
    attacker_pos = random.choice(range(NUM_NODES))
    attacked_nodes.add(attacker_pos)

    # Draw network and attacker move
    draw_network()
    time.sleep(ACTION_DELAY)

    # Defender decides an action
    defender_prev = defender_pos
    defender_action = q_learning_action(defender_pos)
    defender_pos = defender_action
    move_count += 1

    # Update Q-table
    reward = DEFEND_REWARD if defender_pos == attacker_pos else ATTACK_REWARD
    update_q_table(defender_prev, defender_action, reward, attacker_pos)

    # Update saved nodes
    if defender_pos == attacker_pos:
        elapsed_time = time.time() - start_time
        message = f"Attacker caught! Time: {elapsed_time:.2f}s, Moves: {move_count}"
        draw_network(message)
        time.sleep(2)
        reset_game()
        continue
    elif attacker_pos in saved_nodes:
        saved_nodes.add(attacker_pos)

    # Draw network and defender move
    draw_network()
    time.sleep(ACTION_DELAY)

pygame.quit()

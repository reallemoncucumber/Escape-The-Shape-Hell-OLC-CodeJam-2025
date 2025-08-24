# /// script
# dependencies = [
# "pygame-ce",
# "cffi", 
# "pymunk",
# ]
# ///

import pygame
import math
import random
import pymunk
import pickle
import sys
from typing import List, Dict, Any, Tuple

# Initialize Pygame and Pymunk for data generation
pygame.init()
space = pymunk.Space()
space.damping = 0
space.collision_bias = 0.01

# Constants (copied from main.py)
SCREEN_WIDTH = 1920 
SCREEN_HEIGHT = 1080    
FPS = 60
WORLD_WIDTH = SCREEN_WIDTH * 2
WORLD_HEIGHT = SCREEN_HEIGHT * 2
BOUNCE_DAMPING = 1
SIMULATION_DURATION = 300  # 5 minutes in seconds
TOTAL_FRAMES = SIMULATION_DURATION * FPS  # 18,000 frames

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 107, 107)
CYAN = (78, 205, 196)
BLUE = (69, 183, 209)
GREEN = (150, 206, 180)
YELLOW = (255, 234, 167)
PURPLE = (155, 89, 182)
ORANGE = (255, 165, 0)
DARK_BLUE = (30, 60, 114)
LIGHT_GREEN = (46, 204, 113)
PINK = (231, 76, 60)
MOTHER_COLOR = WHITE

class PrecomputeShape:
    """Simplified shape class for precomputation - focuses only on physics"""
    
    def __init__(self, vertices=None, center=None, radius=None, color=WHITE, shape_id=0, is_mother=False):
        self.vertices = vertices
        self.center = center
        self.radius = radius
        self.color = color
        self.is_circle = center is not None and radius is not None
        self.shape_id = shape_id
        self.is_mother = is_mother
        self.mother_pulse_time = 0
        
        # Shape personality system
        if is_mother:
            self.mood = 'happy'
        else:
            self.mood = 'happy' if random.random() < 0.7 else 'angry'
        
        if self.is_mother:
            self.color = MOTHER_COLOR
        
        # Physics properties
        angle = random.uniform(0, 2 * math.pi)
        speed = 64
        normal_x = math.cos(angle)
        normal_y = math.sin(angle)
        self.velocity = [normal_x * speed, normal_y * speed]
        
        if self.is_circle:
            self.mass = math.pi * radius * radius * 0.01
        else:
            self.mass = self.calculate_polygon_area() * 0.01
        
        # Create Pymunk physics objects
        self.pm_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        
        if self.is_circle:
            self.pm_body.position = self.center
            self.pm_shape = pymunk.Circle(self.pm_body, self.radius)
        else:
            center_x = sum(v[0] for v in vertices) / len(vertices)
            center_y = sum(v[1] for v in vertices) / len(vertices)
            self.pm_body.position = (center_x, center_y)
            pm_vertices = [(v[0] - center_x, v[1] - center_y) for v in vertices]
            self.pm_shape = pymunk.Poly(self.pm_body, pm_vertices)
        
        self.pm_shape.mass = self.mass
        self.pm_shape.friction = 0.0
        self.pm_shape.elasticity = BOUNCE_DAMPING
        self.pm_shape.shape_id = self.shape_id
        
        space.add(self.pm_body, self.pm_shape)
        self.collision_bounds = self.calculate_bounds()
    
    def calculate_polygon_area(self):
        if not self.vertices or len(self.vertices) < 3:
            return 100
        area = 0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2
    
    def calculate_bounds(self):
        if self.is_circle:
            return [
                self.center[0] - self.radius,
                self.center[1] - self.radius,
                self.center[0] + self.radius,
                self.center[1] + self.radius
            ]
        else:
            min_x = min(v[0] for v in self.vertices)
            max_x = max(v[0] for v in self.vertices)
            min_y = min(v[1] for v in self.vertices)
            max_y = max(v[1] for v in self.vertices)
            return [min_x, min_y, max_x, max_y]
    
    def update_physics(self, dt, current_time):
        """Update shape position and handle collisions"""
        if self.is_mother:
            self.mother_pulse_time += dt * 4
        
        # Safety check for velocity
        current_speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if current_speed < 1:
            angle = random.uniform(0, 2 * math.pi)
            speed = 64
            self.velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        # Update position
        if self.is_circle:
            self.center = (
                self.center[0] + self.velocity[0] * dt,
                self.center[1] + self.velocity[1] * dt
            )
            self.pm_body.position = self.center
        else:
            dx = self.velocity[0] * dt
            dy = self.velocity[1] * dt
            self.vertices = [(v[0] + dx, v[1] + dy) for v in self.vertices]
            center_x = sum(v[0] for v in self.vertices) / len(self.vertices)
            center_y = sum(v[1] for v in self.vertices) / len(self.vertices)
            self.pm_body.position = (center_x, center_y)
        
        self.collision_bounds = self.calculate_bounds()
        self.check_world_bounds()
    
    def check_world_bounds(self):
        """Handle world boundary collisions"""
        bounds = self.collision_bounds
        
        if bounds[0] < 0:
            self.velocity[0] = abs(self.velocity[0]) * BOUNCE_DAMPING
            if self.is_circle:
                overlap = 0 - bounds[0]
                self.center = (self.center[0] + overlap, self.center[1])
            else:
                overlap = 0 - bounds[0]
                self.vertices = [(v[0] + overlap, v[1]) for v in self.vertices]
        
        elif bounds[2] > WORLD_WIDTH:
            self.velocity[0] = -abs(self.velocity[0]) * BOUNCE_DAMPING
            if self.is_circle:
                overlap = bounds[2] - WORLD_WIDTH
                self.center = (self.center[0] - overlap, self.center[1])
            else:
                overlap = bounds[2] - WORLD_WIDTH
                self.vertices = [(v[0] - overlap, v[1]) for v in self.vertices]
        
        if bounds[1] < 0:
            self.velocity[1] = abs(self.velocity[1]) * BOUNCE_DAMPING
            if self.is_circle:
                overlap = 0 - bounds[1]
                self.center = (self.center[0], self.center[1] + overlap)
            else:
                overlap = 0 - bounds[1]
                self.vertices = [(v[0], v[1] + overlap) for v in self.vertices]
        
        elif bounds[3] > WORLD_HEIGHT:
            self.velocity[1] = -abs(self.velocity[1]) * BOUNCE_DAMPING
            if self.is_circle:
                overlap = bounds[3] - WORLD_HEIGHT
                self.center = (self.center[0], self.center[1] - overlap)
            else:
                overlap = bounds[3] - WORLD_HEIGHT
                self.vertices = [(v[0], v[1] - overlap) for v in self.vertices]
        
        self.collision_bounds = self.calculate_bounds()
    
    def get_serializable_state(self):
        """Return state that can be serialized"""
        return {
            'shape_id': self.shape_id,
            'is_circle': self.is_circle,
            'center': self.center if self.is_circle else None,
            'radius': self.radius if self.is_circle else None,
            'vertices': self.vertices if not self.is_circle else None,
            'velocity': self.velocity,
            'collision_bounds': self.collision_bounds,
            'color': self.color,
            'mood': self.mood,
            'is_mother': self.is_mother,
            'mother_pulse_time': self.mother_pulse_time if self.is_mother else 0
        }

class BackgroundShapePrecompute:
    """Background shape for precomputation"""
    
    def __init__(self):
        self.lifetime = random.randint(60, 180)
        self.alpha = 0
        self.fade_in = True
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.size = random.randint(30, 100)
        self.is_circle = random.choice([True, False])
        
        if not self.is_circle:
            num_vertices = random.randint(3, 6)
            self.vertices = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                x = self.x + self.size * math.cos(angle)
                y = self.y + self.size * math.sin(angle)
                self.vertices.append((x, y))
    
    def update(self):
        if self.fade_in:
            self.alpha = min(30, self.alpha + 1)
            if self.alpha >= 30:
                self.fade_in = False
        else:
            self.alpha = max(0, self.alpha - 1)
        
        self.lifetime -= 1
        return self.lifetime > 0
    
    def get_serializable_state(self):
        return {
            'x': self.x,
            'y': self.y,
            'size': self.size,
            'alpha': self.alpha,
            'is_circle': self.is_circle,
            'vertices': self.vertices if hasattr(self, 'vertices') else None
        }

def generate_regular_polygon(sides, center_x, center_y, radius):
    """Generate regular polygon vertices"""
    vertices = []
    for i in range(sides):
        angle = (i * 2 * math.pi) / sides - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices

def is_position_valid(x, y, radius, existing_shapes):
    """Check if position doesn't overlap with existing shapes"""
    for shape in existing_shapes:
        if shape.is_circle:
            dx = x - shape.center[0]
            dy = y - shape.center[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < radius + shape.radius + 50:
                return False
        else:
            shape_center_x = sum(v[0] for v in shape.vertices) / len(shape.vertices)
            shape_center_y = sum(v[1] for v in shape.vertices) / len(shape.vertices)
            dx = x - shape_center_x
            dy = y - shape_center_y
            distance = math.sqrt(dx*dx + dy*dy)
            max_dist = max(
                math.sqrt((v[0] - shape_center_x)**2 + (v[1] - shape_center_y)**2) 
                for v in shape.vertices
            )
            if distance < radius + max_dist + 50:
                return False
    return True

def generate_shape_at_valid_position(existing_shapes, shape_id):
    """Generate a random shape at valid position"""
    colors = [RED, CYAN, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, LIGHT_GREEN]
    
    max_attempts = 100
    for attempt in range(max_attempts):
        margin = 100
        x = random.randint(margin, WORLD_WIDTH - margin)
        y = random.randint(margin, WORLD_HEIGHT - margin)
        
        shape_type = random.choice(['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'circle'])
        radius = random.randint(40, 80)
        
        if is_position_valid(x, y, radius, existing_shapes):
            color = random.choice(colors)
            
            if shape_type == 'circle':
                return PrecomputeShape(center=(x, y), radius=radius, color=color, shape_id=shape_id)
            else:
                sides_map = {
                    'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 
                    'heptagon': 7, 'octagon': 8
                }
                sides = sides_map[shape_type]
                vertices = generate_regular_polygon(sides, x, y, radius)
                return PrecomputeShape(vertices=vertices, color=color, shape_id=shape_id)
    
    # Fallback position if no valid spot found
    x = random.randint(100, WORLD_WIDTH - 100)
    y = random.randint(100, WORLD_HEIGHT - 100)
    return PrecomputeShape(center=(x, y), radius=50, color=random.choice(colors), shape_id=shape_id)

def on_collision(arbiter, space, data):
    """Handle shape collisions during precomputation"""
    shape_a, shape_b = arbiter.shapes
    shape_a_id, shape_b_id = shape_a.shape_id, shape_b.shape_id
    
    # Find corresponding game shapes
    game_shape_a = next(s for s in shapes if s.shape_id == shape_a_id)
    game_shape_b = next(s for s in shapes if s.shape_id == shape_b_id)
    
    normal = arbiter.normal
    
    rel_velocity = [
        game_shape_b.velocity[0] - game_shape_a.velocity[0],
        game_shape_b.velocity[1] - game_shape_a.velocity[1]
    ]
    normal_velocity = rel_velocity[0] * normal[0] + rel_velocity[1] * normal[1]
    
    if normal_velocity < 0:
        restitution = BOUNCE_DAMPING
        j = -(1 + restitution) * normal_velocity
        j /= (1/game_shape_a.mass + 1/game_shape_b.mass)
        
        game_shape_a.velocity[0] -= j * normal[0] / game_shape_a.mass
        game_shape_a.velocity[1] -= j * normal[1] / game_shape_a.mass
        game_shape_b.velocity[0] += j * normal[0] / game_shape_b.mass
        game_shape_b.velocity[1] += j * normal[1] / game_shape_b.mass
    
    return True

def generate_movement_data():
    """Main function to generate and save precomputed movement data"""
    global shapes, background_shapes
    
    print("Starting movement data generation...")
    print(f"Simulating {SIMULATION_DURATION} seconds ({TOTAL_FRAMES} frames) at {FPS} FPS")
    
    # Initialize shapes
    shapes = []
    num_shapes = 104
    
    print(f"Generating {num_shapes} shapes...")
    for i in range(num_shapes):
        shape = generate_shape_at_valid_position(shapes, i)
        shapes.append(shape)
        if (i + 1) % 20 == 0:
            print(f"Generated {i+1}/{num_shapes} shapes...")
    
    # Create mother shape
    margin = 100
    for _ in range(100):  # Try to find good position for mother
        x = random.randint(margin, WORLD_WIDTH - margin)
        y = random.randint(margin, WORLD_HEIGHT - margin)
        if is_position_valid(x, y, 80, shapes):
            mother_shape = PrecomputeShape(center=(x, y), radius=80, color=WHITE, 
                                         shape_id=len(shapes), is_mother=True)
            mother_shape.mood = 'happy'
            shapes.append(mother_shape)
            break
    
    print(f"Total shapes: {len(shapes)} (including mother)")
    
    # Setup collision handler
    handler = space.add_default_collision_handler()
    handler.pre_solve = on_collision
    
    # Initialize background shapes
    background_shapes = []
    background_time = 0
    
    # Generate frame data
    frame_data = []
    dt = 1.0 / FPS
    
    print("Computing frame data...")
    
    for frame in range(TOTAL_FRAMES):
        current_time = frame * dt
        
        # Update all shapes
        for shape in shapes:
            shape.update_physics(dt, current_time)
        
        # Step physics simulation
        space.step(dt)
        
        # Update background shapes
        background_time += 0.002
        background_shapes = [bg for bg in background_shapes if bg.update()]
        
        if random.random() < 0.02:
            background_shapes.append(BackgroundShapePrecompute())
        
        # Store frame data
        frame_info = {
            'timestamp': current_time,
            'frame_number': frame,
            'shapes': [shape.get_serializable_state() for shape in shapes],
            'background_shapes': [bg.get_serializable_state() for bg in background_shapes],
            'background_time': background_time
        }
        
        frame_data.append(frame_info)
        
        # Progress indication
        if (frame + 1) % 1000 == 0:
            progress = (frame + 1) / TOTAL_FRAMES * 100
            print(f"Progress: {progress:.1f}% ({frame + 1}/{TOTAL_FRAMES} frames)")
    
    # Save data to file
    print("Saving data to movement_data.bin...")
    
    metadata = {
        'total_frames': TOTAL_FRAMES,
        'fps': FPS,
        'duration': SIMULATION_DURATION,
        'world_width': WORLD_WIDTH,
        'world_height': WORLD_HEIGHT,
        'num_shapes': len(shapes),
        'mother_shape_id': len(shapes) - 1  # Mother is last shape
    }
    
    output_data = {
        'metadata': metadata,
        'frames': frame_data
    }
    
    with open('movement_data.bin', 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = len(pickle.dumps(output_data)) / (1024 * 1024)
    print(f"Data generation complete!")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total frames generated: {len(frame_data)}")
    print(f"Saved to: movement_data.bin")

if __name__ == "__main__":
    # Set random seed for reproducible results (optional)
    random.seed(42)
    
    try:
        generate_movement_data()
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)
import pygame
import math
import sys
import random
import pymunk

# Initialize Pygame and Pymunk
pygame.init()
pygame.mixer.init()  # Initialize the sound system
space = pymunk.Space()
space.damping = 0  # No automatic damping/friction
space.collision_bias = 0.01  # Helps prevent shapes from sinking into each other

# Load and start background music
try:
    pygame.mixer.music.load('assets/soundtrack.ogg')
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely
    pygame.mixer.music.set_volume(0.5)  # Set to 50% volume
except Exception as e:
    print(f"Could not load music: {e}")

# Constants
SCREEN_WIDTH = 1920 
SCREEN_HEIGHT = 1080    
FPS = 60
HARPOON_MAX_DISTANCE = 200
HARPOON_SPEED = 8
PULL_SPEED = 6
CAMERA_SMOOTH = 0.1

# Expanded world bounds - 2x screen size for exploration
WORLD_WIDTH = SCREEN_WIDTH * 2
WORLD_HEIGHT = SCREEN_HEIGHT * 2
WORLD_BOUNDS = (0, 0, WORLD_WIDTH, WORLD_HEIGHT)

# Physics constants
BOUNCE_DAMPING = 1  # Energy loss on wall bounces
DAMAGE_RATE = 25  # Health damage per second on angry shapes  
HEAL_RATE = 15   # Health regeneration per second on happy shapes
# No friction - shapes maintain velocity forever!

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

# Mother shape special color - bright gold
MOTHER_COLOR = (255, 215, 0)

class Camera:
    def __init__(self, screen_width, screen_height):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calculate base zoom with smaller visibility area for higher zoom
        visibility_multiplier = 1.8  # Reduced from 2.5 for closer camera
        required_width = HARPOON_MAX_DISTANCE * visibility_multiplier * 2
        required_height = HARPOON_MAX_DISTANCE * visibility_multiplier * 2
        
        zoom_x = screen_width / required_width
        zoom_y = screen_height / required_height
        base_zoom = min(zoom_x, zoom_y)
        
        # Apply zoom multiplier for even closer view
        zoom_multiplier = 1.4
        self.zoom = base_zoom * zoom_multiplier
        
        print(f"Base zoom: {base_zoom:.2f}, Final zoom: {self.zoom:.2f}")
        
        # Higher minimum and maximum zoom bounds
        self.zoom = max(1.5, min(3.5, self.zoom))
    
    def update(self, target_x, target_y):
        """Update camera to follow target smoothly"""
        self.target_x = target_x - self.screen_width / (2 * self.zoom)
        self.target_y = target_y - self.screen_height / (2 * self.zoom)
        
        # Smooth camera movement
        self.x += (self.target_x - self.x) * CAMERA_SMOOTH
        self.y += (self.target_y - self.y) * CAMERA_SMOOTH
    
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates"""
        world_x, world_y = world_pos
        screen_x = (world_x - self.x) * self.zoom
        screen_y = (world_y - self.y) * self.zoom
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        screen_x, screen_y = screen_pos
        world_x = screen_x / self.zoom + self.x
        world_y = screen_y / self.zoom + self.y
        return (world_x, world_y)
    
    def scale_size(self, size):
        """Scale a size value by camera zoom"""
        return size * self.zoom

class Harpoon:
    def __init__(self):
        self.active = False
        self.launching = False
        self.retracting = False
        self.pulling_character = False
        self.start_pos = (0, 0)
        self.current_pos = (0, 0)
        self.target_pos = (0, 0)
        self.hit_pos = None
        self.hit_shape = None
        self.max_distance = HARPOON_MAX_DISTANCE
    
    def launch(self, start_pos, target_pos):
        self.active = True
        self.launching = True
        self.retracting = False
        self.pulling_character = False
        self.start_pos = start_pos
        self.current_pos = start_pos
        self.target_pos = target_pos
        self.hit_pos = None
        self.hit_shape = None
        
        # Limit target to max distance
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > self.max_distance:
            factor = self.max_distance / distance
            self.target_pos = (
                start_pos[0] + dx * factor,
                start_pos[1] + dy * factor
            )
    
    def update(self):
        if not self.active:
            return
        
        if self.launching:
            # Move harpoon towards target
            dx = self.target_pos[0] - self.current_pos[0]
            dy = self.target_pos[1] - self.current_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < HARPOON_SPEED:
                self.current_pos = self.target_pos
                if not self.hit_shape:
                    self.start_retracting()
            else:
                self.current_pos = (
                    self.current_pos[0] + (dx/distance) * HARPOON_SPEED,
                    self.current_pos[1] + (dy/distance) * HARPOON_SPEED
                )
        
        elif self.retracting:
            # Retract harpoon back to start
            dx = self.start_pos[0] - self.current_pos[0]
            dy = self.start_pos[1] - self.current_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < HARPOON_SPEED:
                self.active = False
            else:
                self.current_pos = (
                    self.current_pos[0] + (dx/distance) * HARPOON_SPEED,
                    self.current_pos[1] + (dy/distance) * HARPOON_SPEED
                )
    
    def start_retracting(self):
        self.launching = False
        self.retracting = True
        self.pulling_character = False
    
    def start_pulling_character(self, hit_pos, hit_shape):
        self.hit_pos = hit_pos
        self.hit_shape = hit_shape
        self.launching = False
        self.retracting = False
        self.pulling_character = True
    
    def draw(self, screen, camera):
        if not self.active:
            return
        
        # Convert positions to screen coordinates
        screen_start = camera.world_to_screen(self.start_pos)
        
        # Draw harpoon line
        if self.pulling_character and self.hit_pos:
            screen_hit = camera.world_to_screen(self.hit_pos)
            pygame.draw.line(screen, ORANGE, screen_start, screen_hit, max(1, int(camera.scale_size(3))))
            # Draw harpoon hook at hit position
            pygame.draw.circle(screen, RED, (int(screen_hit[0]), int(screen_hit[1])), int(camera.scale_size(5)))
        else:
            screen_current = camera.world_to_screen(self.current_pos)
            pygame.draw.line(screen, WHITE, screen_start, screen_current, max(1, int(camera.scale_size(2))))
            # Draw harpoon tip
            pygame.draw.circle(screen, RED, (int(screen_current[0]), int(screen_current[1])), int(camera.scale_size(3)))

class Character:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 12
        self.angle = 0  # Position along perimeter (0-1)
        self.speed = 0.003
        self.current_shape_id = 0
        self.being_pulled = False
        self.pull_target = None
        self.pull_target_shape = None
        # Store relative position on shape for moving shapes
        self.shape_relative_angle = 0
        
        # Health system
        self.max_health = 100
        self.current_health = 100
    
    def start_being_pulled(self, target_pos, target_shape_id):
        self.being_pulled = True
        self.pull_target = target_pos
        self.pull_target_shape = target_shape_id
    
    def update_pull(self, shapes):
        if not self.being_pulled or not self.pull_target:
            return
        
        # Move towards pull target (which might be moving!)
        dx = self.pull_target[0] - self.x
        dy = self.pull_target[1] - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < PULL_SPEED:
            # Reached target - snap to new shape
            self.x = self.pull_target[0]
            self.y = self.pull_target[1]
            self.current_shape_id = self.pull_target_shape
            
            # Calculate angle on new shape
            self.angle = self.calculate_angle_on_shape(shapes[self.current_shape_id])
            self.shape_relative_angle = self.angle
            
            self.being_pulled = False
            self.pull_target = None
            self.pull_target_shape = None
            
            # Return True to indicate pull completed
            return True
        else:
            # Move towards target
            self.x += (dx/distance) * PULL_SPEED
            self.y += (dy/distance) * PULL_SPEED
            return False
    
    def update_position_on_moving_shape(self, shapes):
        """Update character position when attached to a moving shape"""
        if self.being_pulled:
            return
        
        current_shape = shapes[self.current_shape_id]
        # Get new position based on current angle parameter
        pos = current_shape.get_position_on_perimeter(self.angle)
        self.x, self.y = pos
    
    def calculate_angle_on_shape(self, shape):
        """Calculate the angle parameter for current position on shape"""
        if shape.is_circle:
            dx = self.x - shape.center[0]
            dy = self.y - shape.center[1]
            angle = math.atan2(dy, dx)
            # Convert from [-π, π] to [0, 2π] then to [0, 1]
            if angle < 0:
                angle += 2 * math.pi
            return angle / (2 * math.pi)
        else:
            # For polygons, find closest edge and position along perimeter
            min_dist = float('inf')
            best_angle = 0
            
            total_perimeter = shape.get_total_perimeter()
            current_distance = 0
            
            for i in range(len(shape.vertices)):
                start = shape.vertices[i]
                end = shape.vertices[(i + 1) % len(shape.vertices)]
                
                # Find closest point on this edge
                edge_dx = end[0] - start[0]
                edge_dy = end[1] - start[1]
                edge_length = math.sqrt(edge_dx*edge_dx + edge_dy*edge_dy)
                
                if edge_length > 0:
                    # Project character position onto edge
                    t = max(0, min(1, ((self.x - start[0]) * edge_dx + (self.y - start[1]) * edge_dy) / (edge_length * edge_length)))
                    closest_x = start[0] + t * edge_dx
                    closest_y = start[1] + t * edge_dy
                    
                    dist = math.sqrt((self.x - closest_x)**2 + (self.y - closest_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_angle = (current_distance + t * edge_length) / total_perimeter
                
                current_distance += edge_length
            
            return best_angle
    
    def draw(self, screen, camera):
        # Convert position to screen coordinates
        screen_pos = camera.world_to_screen((self.x, self.y))
        screen_x, screen_y = screen_pos
        screen_radius = camera.scale_size(self.radius)
        
        # Character body with glow effect
        for i in range(3):
            alpha = 50 - i * 15
            glow_radius = screen_radius + camera.scale_size(i * 2)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*WHITE, alpha), (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surf, (screen_x - glow_radius, screen_y - glow_radius))
        
        # Main body
        pygame.draw.circle(screen, WHITE, (int(screen_x), int(screen_y)), int(screen_radius))
        pygame.draw.circle(screen, BLACK, (int(screen_x), int(screen_y)), int(screen_radius), max(1, int(camera.scale_size(2))))
        
        # Friendly face for the baby character
        
        # Eyes
        eye_offset = camera.scale_size(4)
        eye_radius = camera.scale_size(2)
        left_eye = (int(screen_x - eye_offset), int(screen_y - camera.scale_size(3)))
        right_eye = (int(screen_x + eye_offset), int(screen_y - camera.scale_size(3)))
        pygame.draw.circle(screen, BLACK, left_eye, int(eye_radius))
        pygame.draw.circle(screen, BLACK, right_eye, int(eye_radius))
        
        # Happy smile
        smile_size = camera.scale_size(8)
        smile_rect = pygame.Rect(screen_x - smile_size/2, screen_y + camera.scale_size(1), smile_size, camera.scale_size(5))
        pygame.draw.arc(screen, BLACK, smile_rect, 0, math.pi, max(1, int(camera.scale_size(2))))
        
        # Cheek blush for extra cuteness
        cheek_radius = camera.scale_size(2)
        cheek_color = (255, 200, 200)  # Light pink
        left_cheek = (int(screen_x - camera.scale_size(7)), int(screen_y + camera.scale_size(1)))
        right_cheek = (int(screen_x + camera.scale_size(7)), int(screen_y + camera.scale_size(1)))
        pygame.draw.circle(screen, cheek_color, left_cheek, int(cheek_radius))
        pygame.draw.circle(screen, cheek_color, right_cheek, int(cheek_radius))

class Shape:
    def __init__(self, vertices=None, center=None, radius=None, color=WHITE, shape_id=0, is_mother=False):
        self.vertices = vertices
        self.center = center
        self.radius = radius
        self.color = color
        self.is_circle = center is not None and radius is not None
        self.shape_id = shape_id
        self.is_mother = is_mother
        self.mother_pulse_time = 0  # For pulsating effect
        
        # Shape personality system
        if is_mother:
            self.mood = 'happy'  # Mother is always happy
        else:
            # 70% happy, 30% angry for game balance
            self.mood = 'happy' if random.random() < 0.7 else 'angry'
        
        if self.is_mother:
            self.color = MOTHER_COLOR  # Override color for mother
        
        # Physics properties - increased speed to 64 for more challenge
        angle = random.uniform(0, 2 * math.pi)  # Random direction
        speed = 64  # Doubled from 32
        # Create normal vector and multiply by speed
        normal_x = math.cos(angle)
        normal_y = math.sin(angle)
        self.velocity = [normal_x * speed, normal_y * speed]
        
        if self.is_circle:
            self.mass = math.pi * radius * radius * 0.01  # Area-based mass
        else:
            # Calculate polygon area for mass
            self.mass = self.calculate_polygon_area() * 0.01
        
        # Create Pymunk physics objects
        self.pm_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        
        if self.is_circle:
            self.pm_body.position = self.center
            self.pm_shape = pymunk.Circle(self.pm_body, self.radius)
        else:
            # Convert vertices to Pymunk-style (relative to body position)
            center_x = sum(v[0] for v in vertices) / len(vertices)
            center_y = sum(v[1] for v in vertices) / len(vertices)
            self.pm_body.position = (center_x, center_y)
            
            # Make vertices relative to body position
            pm_vertices = [(v[0] - center_x, v[1] - center_y) for v in vertices]
            self.pm_shape = pymunk.Poly(self.pm_body, pm_vertices)
        
        # Set physics properties
        self.pm_shape.mass = self.mass
        self.pm_shape.friction = 0.0
        self.pm_shape.elasticity = BOUNCE_DAMPING
        
        # Store shape_id as user data for collision handling
        self.pm_shape.shape_id = self.shape_id
        
        # Add to space
        space.add(self.pm_body, self.pm_shape)
        
        # Collision bounds for screen wrapping
        self.collision_bounds = self.calculate_bounds()
    
    def calculate_polygon_area(self):
        """Calculate area of polygon using shoelace formula"""
        if not self.vertices or len(self.vertices) < 3:
            return 100  # Default mass
        
        area = 0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2
    
    def calculate_bounds(self):
        """Calculate axis-aligned bounding box for broad-phase collision detection"""
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
        """Update shape position based on velocity - using Pymunk for collisions"""
        # Update mother pulse animation
        if self.is_mother:
            self.mother_pulse_time += dt * 4  # Pulse frequency
        
        # Safety check: if velocity is somehow zero, give it a new random direction
        current_speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if current_speed < 1:  # Very slow or stationary
            angle = random.uniform(0, 2 * math.pi)
            speed = 64  # Updated speed
            self.velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        # Update position manually
        if self.is_circle:
            self.center = (
                self.center[0] + self.velocity[0] * dt,
                self.center[1] + self.velocity[1] * dt
            )
            # Update Pymunk body position
            self.pm_body.position = self.center
        else:
            dx = self.velocity[0] * dt
            dy = self.velocity[1] * dt
            self.vertices = [(v[0] + dx, v[1] + dy) for v in self.vertices]
            # Update Pymunk body position (using center of mass)
            center_x = sum(v[0] for v in self.vertices) / len(self.vertices)
            center_y = sum(v[1] for v in self.vertices) / len(self.vertices)
            self.pm_body.position = (center_x, center_y)
        
        # Update collision bounds for screen wrapping
        self.collision_bounds = self.calculate_bounds()
        
        # Bounce off expanded world boundaries
        self.check_world_bounds()
    
    def check_world_bounds(self):
        """Check and handle collisions with expanded world boundaries - proper bouncing"""
        bounds = self.collision_bounds
        
        # Check left/right world bounds
        if bounds[0] < 0:  # Hit left edge
            self.velocity[0] = abs(self.velocity[0]) * BOUNCE_DAMPING
            # Minimal position correction to prevent sticking
            if self.is_circle:
                overlap = 0 - bounds[0]
                self.center = (self.center[0] + overlap, self.center[1])
            else:
                overlap = 0 - bounds[0]
                self.vertices = [(v[0] + overlap, v[1]) for v in self.vertices]
        
        elif bounds[2] > WORLD_WIDTH:  # Hit right edge
            self.velocity[0] = -abs(self.velocity[0]) * BOUNCE_DAMPING
            # Minimal position correction to prevent sticking
            if self.is_circle:
                overlap = bounds[2] - WORLD_WIDTH
                self.center = (self.center[0] - overlap, self.center[1])
            else:
                overlap = bounds[2] - WORLD_WIDTH
                self.vertices = [(v[0] - overlap, v[1]) for v in self.vertices]
        
        # Check top/bottom world bounds
        if bounds[1] < 0:  # Hit top edge
            self.velocity[1] = abs(self.velocity[1]) * BOUNCE_DAMPING
            # Minimal position correction to prevent sticking
            if self.is_circle:
                overlap = 0 - bounds[1]
                self.center = (self.center[0], self.center[1] + overlap)
            else:
                overlap = 0 - bounds[1]
                self.vertices = [(v[0], v[1] + overlap) for v in self.vertices]
        
        elif bounds[3] > WORLD_HEIGHT:  # Hit bottom edge
            self.velocity[1] = -abs(self.velocity[1]) * BOUNCE_DAMPING
            # Minimal position correction to prevent sticking
            if self.is_circle:
                overlap = bounds[3] - WORLD_HEIGHT
                self.center = (self.center[0], self.center[1] - overlap)
            else:
                overlap = bounds[3] - WORLD_HEIGHT
                self.vertices = [(v[0], v[1] - overlap) for v in self.vertices]
        
        # Update bounds after boundary correction
        self.collision_bounds = self.calculate_bounds()
    
    def get_center(self):
        """Get center point of polygon"""
        if self.is_circle:
            return self.center
        
        center_x = sum(v[0] for v in self.vertices) / len(self.vertices)
        center_y = sum(v[1] for v in self.vertices) / len(self.vertices)
        return (center_x, center_y)
    
    def draw(self, screen, camera, is_active=False):
        color = self.color
        width = max(1, int(camera.scale_size(5 if is_active else 3)))
        
        # Special mother shape effects
        if self.is_mother:
            pulse_factor = 0.8 + 0.4 * math.sin(self.mother_pulse_time)
            
            # Pulsating outer glow
            glow_size = max(1, int(camera.scale_size(8 * pulse_factor)))
            if self.is_circle:
                screen_center = camera.world_to_screen(self.center)
                glow_radius = camera.scale_size(self.radius + 10 * pulse_factor)
                
                # Create glow surface with alpha
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                alpha = int(100 * pulse_factor)
                pygame.draw.circle(glow_surf, (*MOTHER_COLOR, alpha), 
                                 (glow_radius, glow_radius), glow_radius)
                screen.blit(glow_surf, (screen_center[0] - glow_radius, screen_center[1] - glow_radius))
            else:
                # Polygon glow effect
                center = self.get_center()
                for v in self.vertices:
                    screen_pos = camera.world_to_screen(v)
                    glow_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
                    alpha = int(150 * pulse_factor)
                    pygame.draw.circle(glow_surf, (*MOTHER_COLOR, alpha), (10, 10), 10)
                    screen.blit(glow_surf, (screen_pos[0] - 10, screen_pos[1] - 10))
            
            # Animate the main shape width
            width = max(1, int(camera.scale_size(3 + 4 * pulse_factor)))
        
        # Add glow effect for active shape
        if is_active:
            glow_color = tuple(min(255, c + 50) for c in self.color)
            glow_width = max(1, int(camera.scale_size(2)))
            if self.is_circle:
                screen_center = camera.world_to_screen(self.center)
                screen_radius = camera.scale_size(self.radius + 3)
                pygame.draw.circle(screen, glow_color, 
                                 (int(screen_center[0]), int(screen_center[1])), 
                                 int(screen_radius), glow_width)
            else:
                center = self.get_center()
                glow_vertices = []
                for v in self.vertices:
                    dx, dy = v[0] - center[0], v[1] - center[1]
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx, dy = dx/length * 3, dy/length * 3
                    glow_world_pos = (v[0] + dx, v[1] + dy)
                    glow_vertices.append(camera.world_to_screen(glow_world_pos))
                if len(glow_vertices) > 2:
                    pygame.draw.polygon(screen, glow_color, glow_vertices, glow_width)
        
        # Main shape
        if self.is_circle:
            screen_center = camera.world_to_screen(self.center)
            screen_radius = camera.scale_size(self.radius)
            pygame.draw.circle(screen, color, 
                             (int(screen_center[0]), int(screen_center[1])), 
                             int(screen_radius), width)
        else:
            screen_vertices = [camera.world_to_screen(v) for v in self.vertices]
            if len(screen_vertices) > 2:
                pygame.draw.polygon(screen, color, screen_vertices, width)
        
        # Draw face based on mood
        self.draw_face(screen, camera)
    
    def draw_face(self, screen, camera):
        """Draw a face on the shape based on its mood"""
        center = self.get_center()
        screen_center = camera.world_to_screen(center)
        
        # Calculate face size based on shape size
        if self.is_circle:
            face_scale = min(self.radius / 60, 1.5)  # Scale based on circle radius
        else:
            # For polygons, estimate size from vertices
            min_x = min(v[0] for v in self.vertices)
            max_x = max(v[0] for v in self.vertices)
            min_y = min(v[1] for v in self.vertices)
            max_y = max(v[1] for v in self.vertices)
            avg_size = ((max_x - min_x) + (max_y - min_y)) / 4
            face_scale = min(avg_size / 60, 1.5)
        
        face_scale = max(0.3, face_scale)  # Minimum readable size
        screen_face_scale = camera.scale_size(face_scale)
        
        # Eye positions and sizes
        eye_offset_x = camera.scale_size(12 * face_scale)
        eye_offset_y = camera.scale_size(8 * face_scale)
        eye_radius = max(1, int(camera.scale_size(3 * face_scale)))
        
        left_eye_pos = (int(screen_center[0] - eye_offset_x), int(screen_center[1] - eye_offset_y))
        right_eye_pos = (int(screen_center[0] + eye_offset_x), int(screen_center[1] - eye_offset_y))
        
        if self.mood == 'happy':
            # Draw happy face: round eyes and smile
            pygame.draw.circle(screen, BLACK, left_eye_pos, eye_radius)
            pygame.draw.circle(screen, BLACK, right_eye_pos, eye_radius)
            
            # Happy smile (upward arc - bottom half of circle)
            smile_rect = pygame.Rect(
                screen_center[0] - camera.scale_size(15 * face_scale),
                screen_center[1] + camera.scale_size(5 * face_scale),
                camera.scale_size(30 * face_scale),
                camera.scale_size(15 * face_scale)
            )
            pygame.draw.arc(screen, BLACK, smile_rect, math.pi, 2 * math.pi, max(1, int(camera.scale_size(2 * face_scale))))
            
        elif self.mood == 'angry':
            # Draw angry face: angled eyebrows and frown
            
            # Angry eyebrows (angled lines)
            brow_length = camera.scale_size(8 * face_scale)
            brow_width = max(1, int(camera.scale_size(2 * face_scale)))
            
            # Left eyebrow (angled down towards center)
            left_brow_start = (int(left_eye_pos[0] - brow_length//2), int(left_eye_pos[1] - camera.scale_size(6 * face_scale)))
            left_brow_end = (int(left_eye_pos[0] + brow_length//2), int(left_eye_pos[1] - camera.scale_size(3 * face_scale)))
            pygame.draw.line(screen, RED, left_brow_start, left_brow_end, brow_width)
            
            # Right eyebrow (angled down towards center)
            right_brow_start = (int(right_eye_pos[0] - brow_length//2), int(right_eye_pos[1] - camera.scale_size(3 * face_scale)))
            right_brow_end = (int(right_eye_pos[0] + brow_length//2), int(right_eye_pos[1] - camera.scale_size(6 * face_scale)))
            pygame.draw.line(screen, RED, right_brow_start, right_brow_end, brow_width)
            
            # Angry eyes (small red circles)
            pygame.draw.circle(screen, RED, left_eye_pos, eye_radius)
            pygame.draw.circle(screen, RED, right_eye_pos, eye_radius)
            
            # Angry frown (downward arc)
            frown_rect = pygame.Rect(
                screen_center[0] - camera.scale_size(12 * face_scale),
                screen_center[1] + camera.scale_size(8 * face_scale),
                camera.scale_size(24 * face_scale),
                camera.scale_size(12 * face_scale)
            )
            pygame.draw.arc(screen, RED, frown_rect, math.pi, 2 * math.pi, max(1, int(camera.scale_size(2 * face_scale))))
    
    def get_total_perimeter(self):
        if self.is_circle:
            return 2 * math.pi * self.radius
        
        perimeter = 0
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]
            perimeter += math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        return perimeter
    
    def get_position_on_perimeter(self, angle):
        if self.is_circle:
            a = angle * 2 * math.pi
            x = self.center[0] + self.radius * math.cos(a)
            y = self.center[1] + self.radius * math.sin(a)
            return (x, y)
        
        total_perimeter = self.get_total_perimeter()
        target_distance = angle * total_perimeter
        
        current_distance = 0
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]
            edge_length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            
            if current_distance + edge_length >= target_distance:
                t = (target_distance - current_distance) / edge_length if edge_length > 0 else 0
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                return (x, y)
            
            current_distance += edge_length
        
        return self.vertices[0] if self.vertices else (0, 0)
    
    def check_line_intersection(self, start_pos, end_pos):
        """Check if a line intersects with this shape"""
        if self.is_circle:
            return self.line_circle_intersection(start_pos, end_pos)
        else:
            return self.line_polygon_intersection(start_pos, end_pos)
    
    def line_circle_intersection(self, start_pos, end_pos):
        """Check line intersection with circle"""
        cx, cy = self.center
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Line direction vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from start to circle center
        fx = x1 - cx
        fy = y1 - cy
        
        # Quadratic equation coefficients
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - self.radius*self.radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None
        
        discriminant = math.sqrt(discriminant)
        
        # Two possible intersection points
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        # Check if intersections are on the line segment
        for t in [t1, t2]:
            if 0 <= t <= 1:
                hit_x = x1 + t * dx
                hit_y = y1 + t * dy
                return (hit_x, hit_y)
        
        return None
    
    def line_polygon_intersection(self, start_pos, end_pos):
        """Check line intersection with polygon edges"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        closest_hit = None
        min_distance = float('inf')
        
        for i in range(len(self.vertices)):
            edge_start = self.vertices[i]
            edge_end = self.vertices[(i + 1) % len(self.vertices)]
            
            hit = self.line_line_intersection(start_pos, end_pos, edge_start, edge_end)
            if hit:
                distance = math.sqrt((hit[0] - x1)**2 + (hit[1] - y1)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_hit = hit
        
        return closest_hit
    
    def line_line_intersection(self, p1, p2, p3, p4):
        """Calculate intersection between two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            hit_x = x1 + t*(x2-x1)
            hit_y = y1 + t*(y2-y1)
            return (hit_x, hit_y)
        
        return None

def generate_regular_polygon(sides, center_x, center_y, radius):
    vertices = []
    for i in range(sides):
        angle = (i * 2 * math.pi) / sides - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices

class BackgroundShape:
    def __init__(self):
        self.lifetime = random.randint(60, 180)  # 1-3 seconds at 60 FPS
        self.alpha = 0
        self.fade_in = True
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.size = random.randint(30, 100)
        
        # Randomly choose between circle and polygon
        self.is_circle = random.choice([True, False])
        if not self.is_circle:
            # Create random polygon vertices (3 to 6 sides)
            num_vertices = random.randint(3, 6)
            self.vertices = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                x = self.x + self.size * math.cos(angle)
                y = self.y + self.size * math.sin(angle)
                self.vertices.append((x, y))

    def update(self):
        if self.fade_in:
            self.alpha = min(30, self.alpha + 1)  # Max alpha of 30 (very transparent)
            if self.alpha >= 30:
                self.fade_in = False
        else:
            self.alpha = max(0, self.alpha - 1)
        
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, screen):
        shape_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        if self.is_circle:
            pygame.draw.circle(shape_surface, (255, 255, 255, self.alpha), (self.x, self.y), self.size)
        else:
            pygame.draw.polygon(shape_surface, (255, 255, 255, self.alpha), self.vertices)
        screen.blit(shape_surface, (0, 0))

# List to keep track of background shapes
background_shapes = []
background_time = 0

def create_gradient_background(screen):
    global background_time, background_shapes
    
    # Update background time for color cycling (slower)
    background_time += 0.002  # Reduced from 0.01 for slower cycling
    
    # Create cycling colors
    for y in range(SCREEN_HEIGHT):
        # Create bright cycling colors using sine waves with different phases
        # Minimum value increased to 180 to ensure only bright colors
        r = int(180 + 75 * math.sin(background_time + y * 0.001))
        g = int(180 + 75 * math.sin(background_time * 1.3 + y * 0.001))
        b = int(180 + 75 * math.sin(background_time * 0.7 + y * 0.001))
        color = (r, g, b)
        pygame.draw.line(screen, color, (0, y), (SCREEN_WIDTH, y))
    
    # Manage background shapes
    # Remove expired shapes
    background_shapes = [shape for shape in background_shapes if shape.update()]
    
    # Add new shapes occasionally
    if random.random() < 0.02:  # 2% chance each frame
        background_shapes.append(BackgroundShape())
    
    # Draw all background shapes
    for shape in background_shapes:
        shape.draw(screen)

def draw_crosshair(screen, pos, camera):
    """Draw aiming crosshair at mouse position"""
    x, y = pos  # pos is already in screen coordinates
    size = camera.scale_size(15)
    line_width = max(1, int(camera.scale_size(2)))
    pygame.draw.line(screen, WHITE, (x-size, y), (x+size, y), line_width)
    pygame.draw.line(screen, WHITE, (x, y-size), (x, y+size), line_width)
    pygame.draw.circle(screen, WHITE, (int(x), int(y)), int(size//2), max(1, int(camera.scale_size(1))))

def is_position_valid(x, y, radius, existing_shapes):
    """Check if a position is valid (doesn't overlap with existing shapes)"""
    for shape in existing_shapes:
        if shape.is_circle:
            dx = x - shape.center[0]
            dy = y - shape.center[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < radius + shape.radius + 50:  # 50 pixel buffer
                return False
        else:
            shape_center = shape.get_center()
            dx = x - shape_center[0]
            dy = y - shape_center[1]
            distance = math.sqrt(dx*dx + dy*dy)
            # Estimate shape radius from vertices
            max_dist = 0
            for vertex in shape.vertices:
                vx, vy = vertex
                v_dist = math.sqrt((vx - shape_center[0])**2 + (vy - shape_center[1])**2)
                max_dist = max(max_dist, v_dist)
            if distance < radius + max_dist + 50:  # 50 pixel buffer
                return False
    return True

def generate_shape_at_valid_position(existing_shapes, shape_id):
    """Generate a random shape at a valid (non-overlapping) position"""
    colors = [RED, CYAN, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, LIGHT_GREEN]
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # Random position in expanded world
        margin = 100
        x = random.randint(margin, WORLD_WIDTH - margin)
        y = random.randint(margin, WORLD_HEIGHT - margin)
        
        # Random shape type and size
        shape_type = random.choice(['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'circle'])
        radius = random.randint(40, 80)
        
        if is_position_valid(x, y, radius, existing_shapes):
            color = random.choice(colors)
            
            if shape_type == 'circle':
                return Shape(center=(x, y), radius=radius, color=color, shape_id=shape_id)
            else:
                sides_map = {
                    'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 
                    'heptagon': 7, 'octagon': 8
                }
                sides = sides_map[shape_type]
                vertices = generate_regular_polygon(sides, x, y, radius)
                return Shape(vertices=vertices, color=color, shape_id=shape_id)
    
    # If we can't find a valid position, place it randomly anyway
    x = random.randint(100, WORLD_WIDTH - 100)
    y = random.randint(100, WORLD_HEIGHT - 100)
    return Shape(center=(x, y), radius=50, color=random.choice(colors), shape_id=shape_id)

class DirectionalArrow:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.arrow_size = 30
        self.margin = 50
    
    def is_mother_visible(self, mother_shape, camera):
        """Check if the mother shape is currently visible on screen"""
        mother_center = mother_shape.get_center()
        screen_pos = camera.world_to_screen(mother_center)
        
        # Check if center is within screen bounds (with some margin)
        margin = 100
        return (-margin <= screen_pos[0] <= self.screen_width + margin and 
                -margin <= screen_pos[1] <= self.screen_height + margin)
    
    def get_arrow_position_and_rotation(self, character, mother_shape, camera):
        """Calculate arrow position and rotation to point towards mother"""
        char_pos = (character.x, character.y)
        mother_center = mother_shape.get_center()
        
        # Vector from character to mother
        dx = mother_center[0] - char_pos[0]
        dy = mother_center[1] - char_pos[1]
        
        # Normalize the direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance == 0:
            return None, 0
        
        dir_x = dx / distance
        dir_y = dy / distance
        
        # Calculate angle for rotation
        angle = math.atan2(dir_y, dir_x)
        
        # Find where this direction intersects screen edges
        screen_center_x = self.screen_width // 2
        screen_center_y = self.screen_height // 2
        
        # Scale the direction to reach screen edge
        edge_scale = max(
            abs(screen_center_x / dir_x) if dir_x != 0 else float('inf'),
            abs(screen_center_y / dir_y) if dir_y != 0 else float('inf')
        )
        
        # Position at screen edge minus margin
        edge_scale = max(0, edge_scale - self.margin)
        arrow_x = screen_center_x + dir_x * edge_scale
        arrow_y = screen_center_y + dir_y * edge_scale
        
        # Clamp to screen bounds
        arrow_x = max(self.margin, min(self.screen_width - self.margin, arrow_x))
        arrow_y = max(self.margin, min(self.screen_height - self.margin, arrow_y))
        
        return (arrow_x, arrow_y), math.degrees(angle)
    
    def draw(self, screen, character, mother_shape, camera):
        """Draw the directional arrow pointing to mother"""
        if self.is_mother_visible(mother_shape, camera):
            return  # Don't draw arrow if mother is visible
        
        arrow_pos, angle = self.get_arrow_position_and_rotation(character, mother_shape, camera)
        if not arrow_pos:
            return
        
        # Create arrow points (pointing right initially)
        arrow_points = [
            (self.arrow_size, 0),      # Tip
            (0, -self.arrow_size//2),  # Top back
            (self.arrow_size//3, 0),   # Back middle
            (0, self.arrow_size//2),   # Bottom back
        ]
        
        # Rotate arrow points
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        
        rotated_points = []
        for px, py in arrow_points:
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            rotated_points.append((arrow_pos[0] + rx, arrow_pos[1] + ry))
        
        # Draw arrow with glow effect
        glow_color = (255, 215, 0, 100)  # Gold with transparency
        main_color = MOTHER_COLOR
        
        # Draw glow (larger)
        glow_points = []
        for px, py in arrow_points:
            # Scale up for glow
            px *= 1.3
            py *= 1.3
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            glow_points.append((arrow_pos[0] + rx, arrow_pos[1] + ry))
        
        glow_surf = pygame.Surface((self.arrow_size * 3, self.arrow_size * 3), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surf, glow_color, 
                           [(p[0] - arrow_pos[0] + self.arrow_size * 1.5, 
                             p[1] - arrow_pos[1] + self.arrow_size * 1.5) for p in glow_points])
        screen.blit(glow_surf, (arrow_pos[0] - self.arrow_size * 1.5, arrow_pos[1] - self.arrow_size * 1.5))
        
        # Draw main arrow
        pygame.draw.polygon(screen, main_color, rotated_points)
        pygame.draw.polygon(screen, WHITE, rotated_points, 2)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game state
        self.game_won = False
        self.game_over = False  # New game over state
        self.win_time = 0
        self.shapes_frozen = False
        
        # Health system and visual feedback
        self.screen_flash_timer = 0  # For damage flash effect
        self.last_damage_time = 0  # To prevent multiple flashes per frame
        
        # Create camera
        self.camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Create directional arrow
        self.arrow = DirectionalArrow(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Generate 100+ shapes spread across expanded world
        print("Generating 100+ shapes across expanded world...")
        self.shapes = []
        
        # Generate shapes with non-overlapping positions
        num_shapes = 104  # Doubled for more exploration challenge
        for i in range(num_shapes):
            shape = generate_shape_at_valid_position(self.shapes, i)
            self.shapes.append(shape)
            if i % 20 == 0:  # Print every 20 shapes for 104 total
                print(f"Generated {i+1}/{num_shapes} shapes...")
        
        # First, select a random starting shape for the character
        char_start_shape = random.randint(0, len(self.shapes) - 1)
        start_shape = self.shapes[char_start_shape]
        start_pos = start_shape.get_position_on_perimeter(0.0)
        
        # Now select mother shape ensuring it's far enough away
        min_distance = 800  # Minimum distance in pixels between start and mother
        valid_mother_shapes = []
        
        for i, shape in enumerate(self.shapes):
            if i == char_start_shape:
                continue  # Can't be the same shape
            
            # Calculate distance between starting position and this potential mother
            mother_center = shape.get_center()
            distance = math.sqrt((start_pos[0] - mother_center[0])**2 + 
                               (start_pos[1] - mother_center[1])**2)
            
            if distance >= min_distance:
                valid_mother_shapes.append(i)
        
        # If we don't have enough valid distant shapes, reduce the requirement
        if len(valid_mother_shapes) == 0:
            min_distance = 400  # Fallback to smaller distance
            for i, shape in enumerate(self.shapes):
                if i == char_start_shape:
                    continue
                mother_center = shape.get_center()
                distance = math.sqrt((start_pos[0] - mother_center[0])**2 + 
                                   (start_pos[1] - mother_center[1])**2)
                if distance >= min_distance:
                    valid_mother_shapes.append(i)
        
        # Select mother from valid distant shapes
        if valid_mother_shapes:
            self.mother_shape_id = random.choice(valid_mother_shapes)
        else:
            # Last resort: pick any shape that's not the starting shape
            self.mother_shape_id = (char_start_shape + len(self.shapes) // 2) % len(self.shapes)
        
        self.shapes[self.mother_shape_id].is_mother = True
        self.shapes[self.mother_shape_id].color = MOTHER_COLOR
        
        # Calculate final distance for reporting
        mother_center = self.shapes[self.mother_shape_id].get_center()
        final_distance = math.sqrt((start_pos[0] - mother_center[0])**2 + 
                                 (start_pos[1] - mother_center[1])**2)
        
        print(f"Mother shape selected: Shape {self.mother_shape_id} (Type: {'Circle' if self.shapes[self.mother_shape_id].is_circle else 'Polygon'})")
        print(f"Distance from start to mother: {final_distance:.0f} pixels")
        
        # Count and report shape mood distribution
        happy_count = sum(1 for shape in self.shapes if shape.mood == 'happy')
        angry_count = sum(1 for shape in self.shapes if shape.mood == 'angry')
        print(f"Shape mood distribution: {happy_count} Happy ({happy_count/len(self.shapes)*100:.1f}%), {angry_count} Angry ({angry_count/len(self.shapes)*100:.1f}%)")
        
        # Create character at the predetermined starting position
        self.character = Character(start_pos[0], start_pos[1])
        self.character.current_shape_id = char_start_shape
        
        # Create harpoon
        self.harpoon = Harpoon()
        
        # Setup Pymunk collision handler
        handler = space.add_default_collision_handler()
        handler.pre_solve = self.on_collision
        handler.separate = self.on_separate
        
        # Font for instructions and win screen
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 48)
        self.win_font = pygame.font.Font(None, 72)
        
        # Mouse position
        self.mouse_pos = (0, 0)
        
        # Physics time step
        self.dt = 1.0 / FPS
        self.game_time = 0
        
        pygame.display.set_caption(f"Find Your Mother! - {len(self.shapes)} Shapes to Explore")
        print(f"Game initialized with {len(self.shapes)} shapes in {WORLD_WIDTH}x{WORLD_HEIGHT} world")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.game_won or self.game_over:
                    # Restart game on click after winning or game over
                    self.__init__()
                    return True
                elif event.button == 1 and not self.harpoon.active and not self.character.being_pulled:
                    # Launch harpoon
                    char_pos = (self.character.x, self.character.y)
                    world_mouse_pos = self.camera.screen_to_world(self.mouse_pos)
                    self.harpoon.launch(char_pos, world_mouse_pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif (self.game_won or self.game_over) and event.key == pygame.K_SPACE:
                    # Restart game on spacebar after winning or game over
                    self.__init__()
                    return True
        return True
    
    def on_collision(self, arbiter, space, data):
        """Handle shape collision using Pymunk collision data"""
        if self.shapes_frozen:
            return False  # Don't process collisions when game is won
        
        # Get the colliding shapes
        shape_a, shape_b = arbiter.shapes
        shape_a_id, shape_b_id = shape_a.shape_id, shape_b.shape_id
        
        # Get game shapes from our list using IDs stored in Pymunk shapes
        game_shape_a = self.shapes[shape_a_id]
        game_shape_b = self.shapes[shape_b_id]
        
        # Get collision normal and point
        normal = arbiter.normal
        
        # Calculate relative velocity along normal
        rel_velocity = [
            game_shape_b.velocity[0] - game_shape_a.velocity[0],
            game_shape_b.velocity[1] - game_shape_a.velocity[1]
        ]
        normal_velocity = rel_velocity[0] * normal[0] + rel_velocity[1] * normal[1]
        
        # Only resolve if objects are moving towards each other
        if normal_velocity < 0:
            # Apply bounce using our damping factor
            restitution = BOUNCE_DAMPING
            j = -(1 + restitution) * normal_velocity
            j /= (1/game_shape_a.mass + 1/game_shape_b.mass)
            
            # Apply impulse to both shapes' velocities
            game_shape_a.velocity[0] -= j * normal[0] / game_shape_a.mass
            game_shape_a.velocity[1] -= j * normal[1] / game_shape_a.mass
            game_shape_b.velocity[0] += j * normal[0] / game_shape_b.mass
            game_shape_b.velocity[1] += j * normal[1] / game_shape_b.mass
        
        return True
    
    def on_separate(self, arbiter, space, data):
        """Called when two shapes separate"""
        return True
    
    def check_win_condition(self):
        """Check if player has reached the mother shape"""
        if (self.character.current_shape_id == self.mother_shape_id and 
            not self.character.being_pulled and not self.harpoon.active):
            if not self.game_won:
                self.game_won = True
                self.win_time = pygame.time.get_ticks()
                self.shapes_frozen = True
                print("GAME WON! Player found their mother!")
    
    def update(self):
        if self.game_won or self.game_over:
            return  # Stop updating when game is won or over
        
        keys = pygame.key.get_pressed()
        
        # Update game time
        self.game_time += self.dt
        
        # Update screen flash timer
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= self.dt
        
        # Health system - check current shape mood and apply effects
        if not self.character.being_pulled and not self.harpoon.active:
            current_shape = self.shapes[self.character.current_shape_id]
            
            if current_shape.mood == 'angry':
                # Take damage on angry shapes
                damage_this_frame = DAMAGE_RATE * self.dt
                old_health = self.character.current_health
                self.character.current_health = max(0, self.character.current_health - damage_this_frame)
                
                # Trigger screen flash if we actually took damage and haven't flashed recently
                if old_health > self.character.current_health and self.screen_flash_timer <= 0:
                    self.screen_flash_timer = 0.15  # Flash for 0.15 seconds
                
            elif current_shape.mood == 'happy':
                # Heal on happy shapes
                heal_this_frame = HEAL_RATE * self.dt
                self.character.current_health = min(self.character.max_health, self.character.current_health + heal_this_frame)
        
        # Check for game over condition
        if self.character.current_health <= 0:
            self.game_over = True
            self.shapes_frozen = True
            return
        
        # Update all shapes with physics
        if not self.shapes_frozen:
            for shape in self.shapes:
                shape.update_physics(self.dt, self.game_time)
            
            # Step the Pymunk space to handle collisions
            space.step(self.dt)
        
        # Update character position relative to moving shape
        self.character.update_position_on_moving_shape(self.shapes)
        
        # Update camera to follow character
        self.camera.update(self.character.x, self.character.y)
        
        # Update harpoon
        self.harpoon.update()
        
        # Check for harpoon collisions
        if self.harpoon.launching and not self.harpoon.hit_shape:
            char_pos = (self.character.x, self.character.y)
            for i, shape in enumerate(self.shapes):
                if i != self.character.current_shape_id:
                    hit_pos = shape.check_line_intersection(char_pos, self.harpoon.current_pos)
                    if hit_pos:
                        self.harpoon.start_pulling_character(hit_pos, shape)
                        self.character.start_being_pulled(hit_pos, i)
                        break
        
        # Update character
        if self.character.being_pulled:
            pull_completed = self.character.update_pull(self.shapes)
            self.harpoon.start_pos = (self.character.x, self.character.y)
            
            # Check if pull completed
            if pull_completed:
                # End harpoon first, then check win condition
                self.harpoon.active = False
                # Now check if we reached mother
                self.check_win_condition()
        elif not self.harpoon.active:
            # Normal movement along current shape
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.character.angle -= self.character.speed
                if self.character.angle < 0:
                    self.character.angle += 1
            
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.character.angle += self.character.speed
                if self.character.angle > 1:
                    self.character.angle -= 1
            
            # Update character position
            current_shape = self.shapes[self.character.current_shape_id]
            pos = current_shape.get_position_on_perimeter(self.character.angle)
            self.character.x, self.character.y = pos
        else:
            # Harpoon active but not pulling - stay on current shape
            current_shape = self.shapes[self.character.current_shape_id]
            pos = current_shape.get_position_on_perimeter(self.character.angle)
            self.character.x, self.character.y = pos
        
        # End harpoon when not pulling
        if self.harpoon.pulling_character and not self.character.being_pulled and not self.game_won:
            self.harpoon.active = False
    
    def draw(self):
        # Background
        create_gradient_background(self.screen)
        
        # Draw all shapes
        for i, shape in enumerate(self.shapes):
            is_active = (i == self.character.current_shape_id)
            shape.draw(self.screen, self.camera, is_active)
        
        # Draw harpoon range indicator
        if not self.harpoon.active and not self.character.being_pulled and not self.game_won and not self.game_over:
            char_screen_pos = self.camera.world_to_screen((self.character.x, self.character.y))
            range_radius = self.camera.scale_size(HARPOON_MAX_DISTANCE)
            pygame.draw.circle(self.screen, (50, 50, 50), 
                             (int(char_screen_pos[0]), int(char_screen_pos[1])), 
                             int(range_radius), max(1, int(self.camera.scale_size(1))))
        
        # Draw harpoon
        self.harpoon.draw(self.screen, self.camera)
        
        # Draw character
        self.character.draw(self.screen, self.camera)
        
        # Draw directional arrow
        if not self.game_won and not self.game_over:
            mother_shape = self.shapes[self.mother_shape_id]
            self.arrow.draw(self.screen, self.character, mother_shape, self.camera)
        
        # Draw crosshair
        if not self.harpoon.active and not self.character.being_pulled and not self.game_won and not self.game_over:
            draw_crosshair(self.screen, self.mouse_pos, self.camera)
        
        # Draw screen flash effect for damage
        if self.screen_flash_timer > 0:
            flash_alpha = int(100 * (self.screen_flash_timer / 0.15))  # Fade out effect
            flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, flash_alpha))  # Red flash
            self.screen.blit(flash_surface, (0, 0))
        
        # Draw health bar (always visible during gameplay)
        if not self.game_won and not self.game_over:
            self.draw_health_bar()
        
        # Draw UI overlays
        if self.game_over:
            # Game Over screen
            overlay_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay_surface.fill((0, 0, 0, 150))  # Semi-transparent overlay
            self.screen.blit(overlay_surface, (0, 0))
            
            # Game Over message
            game_over_text = self.win_font.render("GAME OVER", True, RED)
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(game_over_text, game_over_rect)
            
            subtitle = self.title_font.render("Your health reached zero!", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(subtitle, subtitle_rect)
            
            # Restart instruction
            restart_text = self.font.render("Click anywhere or press SPACE to try again", True, YELLOW)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
            self.screen.blit(restart_text, restart_rect)
            
        elif self.game_won:
            # Win screen
            win_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            win_surface.fill((0, 0, 0, 150))  # Semi-transparent overlay
            self.screen.blit(win_surface, (0, 0))
            
            # Win message
            win_text = self.win_font.render("REUNITED!", True, MOTHER_COLOR)
            win_rect = win_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(win_text, win_rect)
            
            subtitle = self.title_font.render("You found your mother!", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(subtitle, subtitle_rect)
            
            # Restart instruction
            restart_text = self.font.render("Click anywhere or press SPACE to play again", True, YELLOW)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
            self.screen.blit(restart_text, restart_rect)
        else:
            # Normal gameplay UI
            title = self.title_font.render("Find Your Mother!", True, MOTHER_COLOR)
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 30))
            self.screen.blit(title, title_rect)
            
            # Current shape mood indicator
            current_shape = self.shapes[self.character.current_shape_id]
            mood_color = LIGHT_GREEN if current_shape.mood == 'happy' else RED
            mood_text = f"Current Shape: {'😊 Happy' if current_shape.mood == 'happy' else '😠 Angry'} - {'Healing' if current_shape.mood == 'happy' else 'Taking Damage!'}"
            
            mood_surface = self.font.render(mood_text, True, mood_color)
            mood_rect = mood_surface.get_rect(center=(SCREEN_WIDTH // 2, 70))
            self.screen.blit(mood_surface, mood_rect)
            
            # Instructions (shortened for space with health bar)
            instructions = [
                "🎯 Navigate to the pulsating golden mother shape to win!",
                "😊 Happy shapes (smiling) heal you over time",
                "😠 Angry shapes (frowning) drain your health - avoid staying too long!",
                "🏹 Use harpoon (Left Click) to escape dangerous shapes",
                "📍 Follow the golden arrow when mother is off-screen"
            ]
            
            y_start = 100
            for i, instruction in enumerate(instructions):
                if "mother" in instruction.lower():
                    color = MOTHER_COLOR
                elif "😊" in instruction or "heal" in instruction.lower():
                    color = LIGHT_GREEN
                elif "😠" in instruction or "drain" in instruction.lower() or "dangerous" in instruction.lower():
                    color = RED
                else:
                    color = WHITE
                
                text = self.font.render(instruction, True, color)
                self.screen.blit(text, (10, y_start + i * 25))
        
        pygame.display.flip()
    
    def draw_health_bar(self):
        """Draw the player's health bar in the top-left corner"""
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 20
        
        # Background (red for lost health)
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, RED, bg_rect)
        
        # Foreground (green for current health)
        health_ratio = self.character.current_health / self.character.max_health
        health_width = int(bar_width * health_ratio)
        if health_width > 0:
            health_rect = pygame.Rect(bar_x, bar_y, health_width, bar_height)
            pygame.draw.rect(self.screen, LIGHT_GREEN, health_rect)
        
        # Border
        pygame.draw.rect(self.screen, WHITE, bg_rect, 2)
        
        # Health text
        health_text = f"Health: {int(self.character.current_health)}/{self.character.max_health}"
        health_surface = self.font.render(health_text, True, WHITE)
        self.screen.blit(health_surface, (bar_x, bar_y + bar_height + 5))
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
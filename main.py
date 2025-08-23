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

# Physics constants
WORLD_BOUNDS = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)  # Use screen boundaries as walls
BOUNCE_DAMPING = 1  # Energy loss on wall bounces
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
        else:
            # Move towards target
            self.x += (dx/distance) * PULL_SPEED
            self.y += (dy/distance) * PULL_SPEED
    
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
        
        # Eyes
        eye_offset = camera.scale_size(4)
        eye_radius = camera.scale_size(2)
        pygame.draw.circle(screen, BLACK, 
                         (int(screen_x - eye_offset), int(screen_y - camera.scale_size(2))), int(eye_radius))
        pygame.draw.circle(screen, BLACK, 
                         (int(screen_x + eye_offset), int(screen_y - camera.scale_size(2))), int(eye_radius))
        
        # Smile
        smile_size = camera.scale_size(10)
        smile_rect = pygame.Rect(screen_x - smile_size/2, screen_y, smile_size, camera.scale_size(6))
        pygame.draw.arc(screen, BLACK, smile_rect, 0, math.pi, max(1, int(camera.scale_size(2))))

class Shape:
    def __init__(self, vertices=None, center=None, radius=None, color=WHITE, shape_id=0):
        self.vertices = vertices
        self.center = center
        self.radius = radius
        self.color = color
        self.is_circle = center is not None and radius is not None
        self.shape_id = shape_id
        
        # Physics properties - consistent speed of 32 in random direction
        angle = random.uniform(0, 2 * math.pi)  # Random direction
        speed = 32  # Fixed high speed
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
        # Safety check: if velocity is somehow zero, give it a new random direction
        current_speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if current_speed < 1:  # Very slow or stationary
            angle = random.uniform(0, 2 * math.pi)
            speed = 32
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
        
        # Bounce off screen boundaries (walls)
        self.check_screen_bounds()
    
    def check_screen_bounds(self):
        """Check and handle collisions with screen boundaries"""
        bounds = self.collision_bounds
        
        # Check left/right screen bounds
        if bounds[0] < 0:  # Hit left edge
            self.velocity[0] = abs(self.velocity[0]) * BOUNCE_DAMPING
            if self.is_circle:
                self.center = (self.radius, self.center[1])
            else:
                offset = 0 - bounds[0]
                self.vertices = [(v[0] + offset, v[1]) for v in self.vertices]
        
        elif bounds[2] > SCREEN_WIDTH:  # Hit right edge
            self.velocity[0] = -abs(self.velocity[0]) * BOUNCE_DAMPING
            if self.is_circle:
                self.center = (SCREEN_WIDTH - self.radius, self.center[1])
            else:
                offset = SCREEN_WIDTH - bounds[2]
                self.vertices = [(v[0] + offset, v[1]) for v in self.vertices]
        
        # Check top/bottom screen bounds
        if bounds[1] < 0:  # Hit top edge
            self.velocity[1] = abs(self.velocity[1]) * BOUNCE_DAMPING
            if self.is_circle:
                self.center = (self.center[0], self.radius)
            else:
                offset = 0 - bounds[1]
                self.vertices = [(v[0], v[1] + offset) for v in self.vertices]
        
        elif bounds[3] > SCREEN_HEIGHT:  # Hit bottom edge
            self.velocity[1] = -abs(self.velocity[1]) * BOUNCE_DAMPING
            if self.is_circle:
                self.center = (self.center[0], SCREEN_HEIGHT - self.radius)
            else:
                offset = SCREEN_HEIGHT - bounds[3]
                self.vertices = [(v[0], v[1] + offset) for v in self.vertices]
        
        # Update bounds after boundary correction
        self.collision_bounds = self.calculate_bounds()
    
    def check_collision_with(self, other, current_time):
        """This method is now handled by Pymunk"""
        return None
        
        # Broad phase - check bounding boxes
        if not self.bounds_overlap(other):
            return None
        
        # Narrow phase - detailed collision detection
        if self.is_circle and other.is_circle:
            return self.circle_circle_collision(other)
        elif self.is_circle and not other.is_circle:
            return self.circle_polygon_collision(other)
        elif not self.is_circle and other.is_circle:
            return other.circle_polygon_collision(self)
        else:
            return self.polygon_polygon_collision(other)
    
    def bounds_overlap(self, other):
        """Quick check if bounding boxes overlap"""
        a = self.collision_bounds
        b = other.collision_bounds
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])
    
    def circle_circle_collision(self, other):
        """Detect collision between two circles"""
        dx = other.center[0] - self.center[0]
        dy = other.center[1] - self.center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        min_distance = self.radius + other.radius
        
        if distance < min_distance:
            # Collision detected
            if distance == 0:  # Prevent division by zero
                normal = (1, 0)
            else:
                normal = (dx / distance, dy / distance)
            
            overlap = min_distance - distance
            return {
                'normal': normal,
                'overlap': overlap,
                'contact_point': (
                    self.center[0] + normal[0] * self.radius,
                    self.center[1] + normal[1] * self.radius
                )
            }
        return None
    
    def circle_polygon_collision(self, polygon):
        """Detect collision between circle and polygon"""
        closest_distance = float('inf')
        collision_normal = None
        contact_point = None
        
        # Check distance to each edge
        for i in range(len(polygon.vertices)):
            start = polygon.vertices[i]
            end = polygon.vertices[(i + 1) % len(polygon.vertices)]
            
            # Find closest point on edge to circle center
            edge_vec = (end[0] - start[0], end[1] - start[1])
            edge_length = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2)
            
            if edge_length == 0:
                continue
            
            edge_unit = (edge_vec[0] / edge_length, edge_vec[1] / edge_length)
            to_center = (self.center[0] - start[0], self.center[1] - start[1])
            
            # Project circle center onto edge
            projection = max(0, min(edge_length, 
                to_center[0] * edge_unit[0] + to_center[1] * edge_unit[1]))
            
            closest_point = (
                start[0] + edge_unit[0] * projection,
                start[1] + edge_unit[1] * projection
            )
            
            # Distance from circle center to closest point
            dist_vec = (self.center[0] - closest_point[0], self.center[1] - closest_point[1])
            distance = math.sqrt(dist_vec[0]**2 + dist_vec[1]**2)
            
            if distance < closest_distance:
                closest_distance = distance
                if distance == 0:
                    collision_normal = (-edge_unit[1], edge_unit[0])  # Perpendicular to edge
                else:
                    collision_normal = (dist_vec[0] / distance, dist_vec[1] / distance)
                contact_point = closest_point
        
        if closest_distance < self.radius:
            overlap = self.radius - closest_distance
            return {
                'normal': collision_normal,
                'overlap': overlap,
                'contact_point': contact_point
            }
        
        return None
    
    def polygon_polygon_collision(self, other):
        """Detect collision between two polygons using SAT"""
        # Get all edge normals from both polygons
        axes = []
        
        # Get normals from first polygon
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]
            edge = (end[0] - start[0], end[1] - start[1])
            length = math.sqrt(edge[0]**2 + edge[1]**2)
            if length > 0:
                normal = (-edge[1] / length, edge[0] / length)
                axes.append(normal)
        
        # Get normals from second polygon
        for i in range(len(other.vertices)):
            start = other.vertices[i]
            end = other.vertices[(i + 1) % len(other.vertices)]
            edge = (end[0] - start[0], end[1] - start[1])
            length = math.sqrt(edge[0]**2 + edge[1]**2)
            if length > 0:
                normal = (-edge[1] / length, edge[0] / length)
                axes.append(normal)
        
        min_overlap = float('inf')
        collision_axis = None
        
        # Test each axis
        for axis in axes:
            # Project both polygons onto axis
            proj1 = self.project_onto_axis(axis)
            proj2 = other.project_onto_axis(axis)
            
            # Check for separation
            if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
                return None  # Separating axis found
            
            # Calculate overlap
            overlap = min(proj1[1], proj2[1]) - max(proj1[0], proj2[0])
            if overlap < min_overlap:
                min_overlap = overlap
                collision_axis = axis
        
        # All axes overlap - collision detected
        if collision_axis:
            # Calculate contact point (approximation)
            center1 = self.get_center()
            center2 = other.get_center()
            contact_point = (
                (center1[0] + center2[0]) / 2,
                (center1[1] + center2[1]) / 2
            )
            
            return {
                'normal': collision_axis,
                'overlap': min_overlap,
                'contact_point': contact_point
            }
        
        return None
    
    def project_onto_axis(self, axis):
        """Project polygon vertices onto an axis"""
        dots = []
        for vertex in self.vertices:
            dot = vertex[0] * axis[0] + vertex[1] * axis[1]
            dots.append(dot)
        return [min(dots), max(dots)]
    
    def get_center(self):
        """Get center point of polygon"""
        if self.is_circle:
            return self.center
        
        center_x = sum(v[0] for v in self.vertices) / len(self.vertices)
        center_y = sum(v[1] for v in self.vertices) / len(self.vertices)
        return (center_x, center_y)
    
    def resolve_collision(self, other, collision_info, current_time):
        """Resolve collision with physics response and anti-bounce protection"""
        normal = collision_info['normal']
        overlap = collision_info['overlap']
        
        # Set collision cooldowns to prevent immediate re-collision
        self.collision_cooldowns[other.shape_id] = current_time
        other.collision_cooldowns[self.shape_id] = current_time
        
        # More aggressive separation to prevent overlap-induced bouncing
        total_mass = self.mass + other.mass
        self_separation = (other.mass / total_mass) * overlap * 1.2  # 20% extra separation
        other_separation = (self.mass / total_mass) * overlap * 1.2
        
        # Move shapes apart
        if self.is_circle:
            self.center = (
                self.center[0] - normal[0] * self_separation,
                self.center[1] - normal[1] * self_separation
            )
        else:
            dx = -normal[0] * self_separation
            dy = -normal[1] * self_separation
            self.vertices = [(v[0] + dx, v[1] + dy) for v in self.vertices]
        
        if other.is_circle:
            other.center = (
                other.center[0] + normal[0] * other_separation,
                other.center[1] + normal[1] * other_separation
            )
        else:
            dx = normal[0] * other_separation
            dy = normal[1] * other_separation
            other.vertices = [(v[0] + dx, v[1] + dy) for v in other.vertices]
        
        # Calculate relative velocity
        rel_velocity = [
            other.velocity[0] - self.velocity[0],
            other.velocity[1] - self.velocity[1]
        ]
        
        # Velocity component along collision normal
        velocity_along_normal = (
            rel_velocity[0] * normal[0] + 
            rel_velocity[1] * normal[1]
        )
        
        # Don't resolve if velocities are separating
        if velocity_along_normal > 0:
            return
        
        # Collision impulse with slight damping to reduce energy buildup
        restitution = 0.85  # Slightly reduced for stability
        impulse_magnitude = -(1 + restitution) * velocity_along_normal
        impulse_magnitude /= (1/self.mass + 1/other.mass)
        
        # Apply impulse
        impulse = [normal[0] * impulse_magnitude, normal[1] * impulse_magnitude]
        
        self.velocity[0] -= impulse[0] / self.mass
        self.velocity[1] -= impulse[1] / self.mass
        other.velocity[0] += impulse[0] / other.mass
        other.velocity[1] += impulse[1] / other.mass
        
        # Limit extreme velocities to prevent chaos (increased limit for speed 32)
        max_velocity = 40
        self.velocity[0] = max(-max_velocity, min(max_velocity, self.velocity[0]))
        self.velocity[1] = max(-max_velocity, min(max_velocity, self.velocity[1]))
        other.velocity[0] = max(-max_velocity, min(max_velocity, other.velocity[0]))
        other.velocity[1] = max(-max_velocity, min(max_velocity, other.velocity[1]))
        
        # Update bounds after collision resolution
        self.collision_bounds = self.calculate_bounds()
        other.collision_bounds = other.calculate_bounds()
    
    def draw(self, screen, camera, is_active=False):
        color = self.color
        width = max(1, int(camera.scale_size(5 if is_active else 3)))
        
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
        
        # Draw velocity vector for debugging (set to False now that we have many shapes)
        if False:  # Disabled for cleaner visual with 22 shapes
            center = self.get_center()
            screen_center = camera.world_to_screen(center)
            vel_end = camera.world_to_screen((center[0] + self.velocity[0] * 5, center[1] + self.velocity[1] * 5))
            pygame.draw.line(screen, (255, 255, 0), screen_center, vel_end, max(1, int(camera.scale_size(2))))
    
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

def create_gradient_background(screen):
    for y in range(SCREEN_HEIGHT):
        intensity = int(15 + 10 * math.sin(y * 0.01))
        color = (intensity, intensity // 2, intensity * 2)
        pygame.draw.line(screen, color, (0, y), (SCREEN_WIDTH, y))

def draw_crosshair(screen, pos, camera):
    """Draw aiming crosshair at mouse position"""
    x, y = pos  # pos is already in screen coordinates
    size = camera.scale_size(15)
    line_width = max(1, int(camera.scale_size(2)))
    pygame.draw.line(screen, WHITE, (x-size, y), (x+size, y), line_width)
    pygame.draw.line(screen, WHITE, (x, y-size), (x, y+size), line_width)
    pygame.draw.circle(screen, WHITE, (int(x), int(y)), int(size//2), max(1, int(camera.scale_size(1))))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Create camera
        self.camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Create character
        self.character = Character(200, 200)
        
        # Create harpoon
        self.harpoon = Harpoon()
        
        # Setup Pymunk collision handler
        handler = space.add_default_collision_handler()
        handler.pre_solve = self.on_collision
        handler.separate = self.on_separate
        # Create multiple moving shapes spread across the screen - MORE SHAPES with size variety!
        self.shapes = [
            # Row 1 - Top
            Shape(vertices=[(200, 100), (120, 220), (280, 220)], 
                  color=RED, shape_id=0),
            Shape(vertices=[(400, 120), (550, 120), (550, 270), (400, 270)], 
                  color=CYAN, shape_id=1),
            Shape(center=(750, 200), radius=80, color=YELLOW, shape_id=2),
            Shape(vertices=generate_regular_polygon(5, 1000, 150, 70),
                  color=GREEN, shape_id=3),
            Shape(vertices=generate_regular_polygon(6, 1250, 180, 60),
                  color=BLUE, shape_id=4),
            Shape(vertices=generate_regular_polygon(8, 1500, 160, 45),
                  color=PURPLE, shape_id=5),
            
            # Row 2 - Middle
            Shape(center=(150, 400), radius=65, color=ORANGE, shape_id=6),
            Shape(vertices=[(350, 350), (450, 350), (450, 450), (350, 450)], 
                  color=PINK, shape_id=7),
            Shape(vertices=generate_regular_polygon(3, 650, 400, 70),
                  color=LIGHT_GREEN, shape_id=8),
            Shape(vertices=generate_regular_polygon(7, 900, 380, 50),
                  color=(255, 200, 100), shape_id=9),  # Orange-ish
            Shape(center=(1150, 420), radius=75, color=(100, 255, 200), shape_id=10),  # Mint
            Shape(vertices=generate_regular_polygon(4, 1400, 400, 80),  # Diamond orientation
                  color=(255, 100, 255), shape_id=11),  # Magenta
            
            # Row 3 - Bottom  
            Shape(vertices=generate_regular_polygon(5, 200, 650, 85),
                  color=GREEN, shape_id=12),
            Shape(vertices=generate_regular_polygon(6, 500, 700, 65),
                  color=BLUE, shape_id=13),
            Shape(vertices=generate_regular_polygon(8, 750, 680, 55),
                  color=PURPLE, shape_id=14),
            Shape(center=(1000, 650), radius=90, color=(200, 100, 255), shape_id=15),  # Purple-ish
            Shape(vertices=generate_regular_polygon(3, 1250, 700, 75),
                  color=(255, 150, 150), shape_id=16),  # Light red
            Shape(vertices=[(1450, 620), (1550, 620), (1600, 720), (1400, 720)], 
                  color=(150, 255, 150), shape_id=17),  # Light green
            
            # Row 4 - Extra bottom
            Shape(center=(300, 900), radius=60, color=(100, 200, 255), shape_id=18),  # Light blue
            Shape(vertices=generate_regular_polygon(7, 600, 880, 65),
                  color=(255, 255, 100), shape_id=19),  # Bright yellow
            Shape(vertices=generate_regular_polygon(9, 900, 900, 50),  # Nonagon!
                  color=(200, 200, 255), shape_id=20),  # Light purple
            Shape(vertices=[(1100, 850), (1200, 850), (1250, 920), (1200, 990), (1100, 990), (1050, 920)],
                  color=(255, 200, 200), shape_id=21),  # Hexagon-ish light pink
        ]
        
        # Define shape names for UI display
        self.shape_names = [
            'Red Triangle', 'Cyan Square', 'Yellow Circle', 'Green Pentagon', 'Blue Hexagon', 'Purple Octagon',
            'Orange Circle', 'Pink Square', 'Light Green Triangle', 'Orange Heptagon', 'Mint Circle', 'Magenta Diamond',
            'Green Pentagon 2', 'Blue Hexagon 2', 'Purple Octagon 2', 'Purple Circle', 'Light Red Triangle', 'Light Green Trapezoid',
            'Light Blue Circle', 'Bright Yellow Heptagon', 'Light Purple Nonagon', 'Pink Hexagon'
        ]
        
        # Font for instructions
        self.font = pygame.font.Font(None, 16)  # Slightly smaller for more instructions
        self.title_font = pygame.font.Font(None, 36)
        
        # Mouse position
        self.mouse_pos = (0, 0)
        
        # Physics time step
        self.dt = 1.0 / FPS
        self.game_time = 0  # Track game time for collision cooldowns
        
        # Set window caption after shapes are created
        pygame.display.set_caption(f"Harpoon Navigator - {len(self.shapes)} Moving Shapes!")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not self.harpoon.active and not self.character.being_pulled:  # Left click
                    # Launch harpoon - convert screen coordinates to world coordinates
                    char_pos = (self.character.x, self.character.y)
                    world_mouse_pos = self.camera.screen_to_world(self.mouse_pos)
                    self.harpoon.launch(char_pos, world_mouse_pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def on_collision(self, arbiter, space, data):
        """Handle shape collision using Pymunk collision data"""
        # Get the colliding shapes
        shape_a, shape_b = arbiter.shapes
        shape_a_id, shape_b_id = shape_a.shape_id, shape_b.shape_id
        
        # Get game shapes from our list using IDs stored in Pymunk shapes
        game_shape_a = self.shapes[shape_a_id]
        game_shape_b = self.shapes[shape_b_id]
        
        # Get collision normal and point
        normal = arbiter.normal
        point = arbiter.contact_point_set.points[0].point_a
        
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
            
            # Separate the shapes to prevent overlap
            penetration = arbiter.contact_point_set.points[0].distance
            percent = 0.2  # Penetration resolution percentage
            slop = 0.1  # Penetration allowance
            
            if penetration < -slop:
                separation = -(penetration + slop) * percent
                
                # Move shapes apart based on their masses
                mass_sum = game_shape_a.mass + game_shape_b.mass
                game_shape_a_amount = separation * (game_shape_b.mass / mass_sum)
                game_shape_b_amount = separation * (game_shape_a.mass / mass_sum)
                
                if game_shape_a.is_circle:
                    game_shape_a.center = (
                        game_shape_a.center[0] - normal[0] * game_shape_a_amount,
                        game_shape_a.center[1] - normal[1] * game_shape_a_amount
                    )
                    game_shape_a.pm_body.position = game_shape_a.center
                else:
                    dx = -normal[0] * game_shape_a_amount
                    dy = -normal[1] * game_shape_a_amount
                    game_shape_a.vertices = [(v[0] + dx, v[1] + dy) for v in game_shape_a.vertices]
                    center_x = sum(v[0] for v in game_shape_a.vertices) / len(game_shape_a.vertices)
                    center_y = sum(v[1] for v in game_shape_a.vertices) / len(game_shape_a.vertices)
                    game_shape_a.pm_body.position = (center_x, center_y)
                
                if game_shape_b.is_circle:
                    game_shape_b.center = (
                        game_shape_b.center[0] + normal[0] * game_shape_b_amount,
                        game_shape_b.center[1] + normal[1] * game_shape_b_amount
                    )
                    game_shape_b.pm_body.position = game_shape_b.center
                else:
                    dx = normal[0] * game_shape_b_amount
                    dy = normal[1] * game_shape_b_amount
                    game_shape_b.vertices = [(v[0] + dx, v[1] + dy) for v in game_shape_b.vertices]
                    center_x = sum(v[0] for v in game_shape_b.vertices) / len(game_shape_b.vertices)
                    center_y = sum(v[1] for v in game_shape_b.vertices) / len(game_shape_b.vertices)
                    game_shape_b.pm_body.position = (center_x, center_y)
        
        # Let Pymunk know we've handled the collision
        return True

    def on_separate(self, arbiter, space, data):
        """Called when two shapes separate"""
        return True

    def update(self):
        keys = pygame.key.get_pressed()
        
        # Update game time
        self.game_time += self.dt
        
        # Update all shapes with physics
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
        
        # Check for harpoon collisions with moving shapes
        if self.harpoon.launching and not self.harpoon.hit_shape:
            char_pos = (self.character.x, self.character.y)
            for i, shape in enumerate(self.shapes):
                if i != self.character.current_shape_id:  # Can't grapple current shape
                    hit_pos = shape.check_line_intersection(char_pos, self.harpoon.current_pos)
                    if hit_pos:
                        # Harpoon hit this shape!
                        self.harpoon.start_pulling_character(hit_pos, shape)
                        self.character.start_being_pulled(hit_pos, i)
                        break
        
        # Update character
        if self.character.being_pulled:
            # Update pull target if harpoon is attached to a moving shape
            if self.harpoon.pulling_character and self.harpoon.hit_pos:
                # The hit position moves with the shape - we need to track it
                # For now, keep the original hit position (simpler)
                pass
            
            self.character.update_pull(self.shapes)
            # Update harpoon start position while pulling
            self.harpoon.start_pos = (self.character.x, self.character.y)
        elif not self.harpoon.active:
            # Normal movement along current shape (only when harpoon is not active)
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.character.angle -= self.character.speed
                if self.character.angle < 0:
                    self.character.angle += 1
            
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.character.angle += self.character.speed
                if self.character.angle > 1:
                    self.character.angle -= 1
            
            # Update character position based on current shape
            current_shape = self.shapes[self.character.current_shape_id]
            pos = current_shape.get_position_on_perimeter(self.character.angle)
            self.character.x, self.character.y = pos
        else:
            # Harpoon is active but not pulling - character stays in current position on moving shape
            current_shape = self.shapes[self.character.current_shape_id]
            pos = current_shape.get_position_on_perimeter(self.character.angle)
            self.character.x, self.character.y = pos
        
        # End harpoon pull when character reaches target
        if self.harpoon.pulling_character and not self.character.being_pulled:
            self.harpoon.active = False
    
    def draw(self):
        # Background
        create_gradient_background(self.screen)
        
        # Draw all shapes with camera
        for i, shape in enumerate(self.shapes):
            is_active = (i == self.character.current_shape_id)
            shape.draw(self.screen, self.camera, is_active)
        
        # Draw harpoon range indicator (subtle circle around character)
        if not self.harpoon.active and not self.character.being_pulled:
            char_screen_pos = self.camera.world_to_screen((self.character.x, self.character.y))
            range_radius = self.camera.scale_size(HARPOON_MAX_DISTANCE)
            pygame.draw.circle(self.screen, (50, 50, 50), 
                             (int(char_screen_pos[0]), int(char_screen_pos[1])), 
                             int(range_radius), max(1, int(self.camera.scale_size(1))))
        
        # Draw harpoon with camera
        self.harpoon.draw(self.screen, self.camera)
        
        # Draw character with camera
        self.character.draw(self.screen, self.camera)
        
        # Draw crosshair at mouse position (screen coordinates, no camera transform needed)
        if not self.harpoon.active and not self.character.being_pulled:
            draw_crosshair(self.screen, self.mouse_pos, self.camera)
        
        # Draw UI (screen coordinates, no camera transform)
        title = self.title_font.render(f"Harpoon Navigator - {len(self.shapes)} Shape Chaos!", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 25))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "A/D or Arrow Keys: Move along current shape (disabled while harpoon active)",
            "Left Click: Launch harpoon at cursor to grab nearby moving shapes",
            f"Current Shape: {self.shape_names[self.character.current_shape_id]} ({self.character.current_shape_id + 1}/{len(self.shapes)})",
            f"Harpoon Range: {HARPOON_MAX_DISTANCE} pixels (gray circle)",
            f"NEW: {len(self.shapes)} shapes moving at SPEED 32 in random directions!",
            "Triangles, squares, circles, pentagons, hexagons, octagons, and more!",
            "Anti-bounce protection prevents infinite collision loops!",
            "Shapes bounce off screen walls with consistent physics!",
            f"Camera Zoom: {self.camera.zoom:.2f}x"
        ]
        
        for i, instruction in enumerate(instructions):
            if i == 2:  # Current shape
                color = LIGHT_GREEN
            elif i in [4, 5]:  # New features
                color = YELLOW
            else:
                color = WHITE
            text = self.font.render(instruction, True, color)
            self.screen.blit(text, (10, 50 + i * 18))
        
        pygame.display.flip()
    
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
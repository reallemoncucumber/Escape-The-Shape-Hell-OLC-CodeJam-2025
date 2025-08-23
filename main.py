import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1920 
SCREEN_HEIGHT = 1080    
FPS = 60
HARPOON_MAX_DISTANCE = 200
HARPOON_SPEED = 8
PULL_SPEED = 6
CAMERA_SMOOTH = 0.1

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
        
        # Calculate zoom to ensure harpoon range is always visible
        # We want at least HARPOON_MAX_DISTANCE * 2.5 visible in both directions
        required_width = HARPOON_MAX_DISTANCE * 2.5 * 2  # 2.5x range in each direction
        required_height = HARPOON_MAX_DISTANCE * 2.5 * 2
        
        zoom_x = screen_width / required_width
        zoom_y = screen_height / required_height
        self.zoom = min(zoom_x, zoom_y)  # Use smaller zoom to fit both dimensions
        print(self.zoom)
        # Ensure minimum zoom for gameplay
        self.zoom = max(0.8, min(2.0, self.zoom))
        print(self.zoom)
    
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
    
    def start_being_pulled(self, target_pos, target_shape_id):
        self.being_pulled = True
        self.pull_target = target_pos
        self.pull_target_shape = target_shape_id
    
    def update_pull(self, shapes):
        if not self.being_pulled or not self.pull_target:
            return
        
        # Move towards pull target
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
            
            self.being_pulled = False
            self.pull_target = None
            self.pull_target_shape = None
        else:
            # Move towards target
            self.x += (dx/distance) * PULL_SPEED
            self.y += (dy/distance) * PULL_SPEED
    
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
                center_x = sum(v[0] for v in self.vertices) / len(self.vertices)
                center_y = sum(v[1] for v in self.vertices) / len(self.vertices)
                glow_vertices = []
                for v in self.vertices:
                    dx, dy = v[0] - center_x, v[1] - center_y
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx, dy = dx/length * 3, dy/length * 3
                    glow_world_pos = (v[0] + dx, v[1] + dy)
                    glow_vertices.append(camera.world_to_screen(glow_world_pos))
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
            pygame.draw.polygon(screen, color, screen_vertices, width)
    
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
                t = (target_distance - current_distance) / edge_length
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                return (x, y)
            
            current_distance += edge_length
        
        return self.vertices[0]
    
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
        pygame.display.set_caption("Harpoon Shape Navigator - Python/Pygame")
        self.clock = pygame.time.Clock()
        
        # Create camera
        self.camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Create character
        self.character = Character(200, 200)
        
        # Create harpoon
        self.harpoon = Harpoon()
        
        # Create multiple shapes spread across the screen
        self.shapes = [
            # Triangle (top-left)
            Shape(vertices=[(200, 100), (120, 220), (280, 220)], 
                  color=RED, shape_id=0),
            
            # Square (top-center)
            Shape(vertices=[(400, 120), (550, 120), (550, 270), (400, 270)], 
                  color=CYAN, shape_id=1),
            
            # Circle (top-right)  
            Shape(center=(750, 200), radius=80, color=YELLOW, shape_id=2),
            
            # Pentagon (bottom-left)
            Shape(vertices=generate_regular_polygon(5, 200, 450, 80),
                  color=GREEN, shape_id=3),
            
            # Hexagon (bottom-center)
            Shape(vertices=generate_regular_polygon(6, 500, 500, 70),
                  color=BLUE, shape_id=4),
            
            # Octagon (bottom-right)
            Shape(vertices=generate_regular_polygon(8, 750, 480, 60),
                  color=PURPLE, shape_id=5)
        ]
        
        # Font for instructions
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 36)
        
        # Mouse position
        self.mouse_pos = (0, 0)
    
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
    
    def update(self):
        keys = pygame.key.get_pressed()
        
        # Update camera to follow character
        self.camera.update(self.character.x, self.character.y)
        
        # Update harpoon
        self.harpoon.update()
        
        # Check for harpoon collisions
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
            # Harpoon is active but not pulling - character stays in current position
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
        title = self.title_font.render("Harpoon Shape Navigator", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 25))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "A/D or Arrow Keys: Move along current shape (disabled while harpoon active)",
            "Left Click: Launch harpoon at cursor to grab nearby shapes",
            f"Current Shape: {['Triangle', 'Square', 'Circle', 'Pentagon', 'Hexagon', 'Octagon'][self.character.current_shape_id]}",
            f"Harpoon Range: {HARPOON_MAX_DISTANCE} pixels (gray circle)",
            "Note: You cannot move while harpoon is launching or retracting!",
            f"Camera Zoom: {self.camera.zoom:.2f}x"
        ]
        
        for i, instruction in enumerate(instructions):
            color = LIGHT_GREEN if i == 2 else WHITE
            text = self.font.render(instruction, True, color)
            self.screen.blit(text, (10, 50 + i * 22))
        
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
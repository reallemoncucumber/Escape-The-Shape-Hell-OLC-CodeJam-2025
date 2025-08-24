# /// script
# dependencies = [
# "pygame-ce",
# "cffi",
# "pymunk",
# ]
# ///

import pygame
import math
import sys
import random
import pymunk
import asyncio
import platform
import os
import pickle

# Platform-specific setup for web
if sys.platform == 'emscripten':
    platform.window.canvas.style.imageRendering = 'pixelated'

# Initialize Pygame (Pymunk not needed for physics anymore)
pygame.init()
pygame.mixer.init()

# Function to start/restart background music
def start_background_music():
    try:
        pygame.mixer.music.load('assets/soundtrack.ogg')
        pygame.mixer.music.play(-1)
        pygame.mixer.music.set_volume(0.5)
    except Exception as e:
        print(f"Could not load music: {e}")

# Start background music initially
start_background_music()

# Constants
SCREEN_WIDTH = 1920 
SCREEN_HEIGHT = 1080    
FPS = 60
HARPOON_MAX_DISTANCE = 200
HARPOON_SPEED = 8
PULL_SPEED = 6
CAMERA_SMOOTH = 0.1

WORLD_WIDTH = SCREEN_WIDTH * 2
WORLD_HEIGHT = SCREEN_HEIGHT * 2
WORLD_BOUNDS = (0, 0, WORLD_WIDTH, WORLD_HEIGHT)

DAMAGE_RATE = 25
HEAL_RATE = 15

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

class PrecomputedMovementData:
    """Handles loading and playback of precomputed movement data"""
    
    def __init__(self):
        self.data = None
        self.metadata = None
        self.current_frame = 0
        self.current_time = 0.0
        self.loaded = False
    
    def load_data(self, filename='movement_data.bin'):
        """Load precomputed movement data from file"""
        try:
            print("Loading precomputed movement data...")
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.metadata = data['metadata']
            self.data = data['frames']
            self.loaded = True
            
            print(f"Loaded {self.metadata['total_frames']} frames")
            print(f"Duration: {self.metadata['duration']} seconds")
            print(f"Shapes: {self.metadata['num_shapes']}")
            
        except FileNotFoundError:
            print(f"ERROR: {filename} not found!")
            print("Please run generate_movement_data.py first to create the precomputed data.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading movement data: {e}")
            sys.exit(1)
    
    def get_frame_data(self, time):
        """Get frame data for a specific time, with wrapping"""
        if not self.loaded:
            return None
        
        # Wrap time to loop the simulation
        wrapped_time = time % self.metadata['duration']
        
        # Calculate frame index
        frame_index = int(wrapped_time * self.metadata['fps'])
        frame_index = min(frame_index, len(self.data) - 1)
        
        return self.data[frame_index]
    
    def get_interpolated_frame(self, time):
        """Get interpolated frame data for smooth playback"""
        if not self.loaded:
            return None
        
        wrapped_time = time % self.metadata['duration']
        frame_float = wrapped_time * self.metadata['fps']
        frame_index = int(frame_float)
        frame_fraction = frame_float - frame_index
        
        if frame_index >= len(self.data) - 1:
            return self.data[-1]
        
        current_frame = self.data[frame_index]
        next_frame = self.data[frame_index + 1]
        
        # Simple interpolation for positions (could be enhanced for smoother movement)
        if frame_fraction < 0.5:
            return current_frame
        else:
            return next_frame

# Global movement data manager
movement_data = PrecomputedMovementData()

class Camera:
    def __init__(self, screen_width, screen_height):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        visibility_multiplier = 1.8
        required_width = HARPOON_MAX_DISTANCE * visibility_multiplier * 2
        required_height = HARPOON_MAX_DISTANCE * visibility_multiplier * 2
        
        zoom_x = screen_width / required_width
        zoom_y = screen_height / required_height
        base_zoom = min(zoom_x, zoom_y)
        
        zoom_multiplier = 1.4
        self.zoom = base_zoom * zoom_multiplier
        
        self.zoom = max(1.5, min(3.5, self.zoom))
    
    def update(self, target_x, target_y):
        self.target_x = target_x - self.screen_width / (2 * self.zoom)
        self.target_y = target_y - self.screen_height / (2 * self.zoom)
        
        self.x += (self.target_x - self.x) * CAMERA_SMOOTH
        self.y += (self.target_y - self.y) * CAMERA_SMOOTH
    
    def world_to_screen(self, world_pos):
        world_x, world_y = world_pos
        screen_x = (world_x - self.x) * self.zoom
        screen_y = (world_y - self.y) * self.zoom
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_pos):
        screen_x, screen_y = screen_pos
        world_x = screen_x / self.zoom + self.x
        world_y = screen_y / self.zoom + self.y
        return (world_x, world_y)
    
    def scale_size(self, size):
        return size * self.zoom
    
    def is_visible(self, shape):
        """Check if a shape is visible within the camera view (frustum culling)"""
        # Get the bounding box of the shape
        if shape.is_circle:
            # For circles, calculate screen bounds
            center_x, center_y = shape.center
            radius = shape.radius
            
            # Calculate screen bounds of the circle
            left = (center_x - radius - self.x) * self.zoom
            right = (center_x + radius - self.x) * self.zoom
            top = (center_y - radius - self.y) * self.zoom
            bottom = (center_y + radius - self.y) * self.zoom
        else:
            # For polygons, use the pre-calculated collision bounds for efficiency
            bounds = shape.collision_bounds
            left = (bounds[0] - self.x) * self.zoom
            right = (bounds[2] - self.x) * self.zoom
            top = (bounds[1] - self.y) * self.zoom
            bottom = (bounds[3] - self.y) * self.zoom
        
        # Check if the shape's bounding box intersects with the screen
        return not (right < 0 or left > self.screen_width or 
                   bottom < 0 or top > self.screen_height)

class Harpoon:
    def __init__(self):
        self.active = False
        self.launching = False
        self.retracting = False
        self.pulling_character = False
        self.current_pos = (0, 0)
        self.target_pos = (0, 0)
        self.hit_pos = None
        self.hit_shape = None
        self.max_distance = HARPOON_MAX_DISTANCE
        self.launch_pos = (0, 0)
        
        try:
            self.launch_sound = pygame.mixer.Sound('assets/harpoon.ogg')
            self.launch_sound.set_volume(0.4)
        except Exception as e:
            print(f"Could not load harpoon sound: {e}")
            self.launch_sound = None
    
    def launch(self, start_pos, target_pos):
        self.active = True
        self.launching = True
        self.retracting = False
        self.pulling_character = False
        self.launch_pos = start_pos
        self.current_pos = start_pos
        self.target_pos = target_pos
        self.hit_pos = None
        self.hit_shape = None
        
        if self.launch_sound:
            self.launch_sound.play()
        
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > self.max_distance:
            factor = self.max_distance / distance
            self.target_pos = (
                start_pos[0] + dx * factor,
                start_pos[1] + dy * factor
            )
    
    def update(self, character_pos):
        if not self.active:
            return
        
        if self.launching:
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
            dx = character_pos[0] - self.current_pos[0]
            dy = character_pos[1] - self.current_pos[1]
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
    
    def draw(self, screen, camera, character_pos):
        if not self.active:
            return
        
        screen_start = camera.world_to_screen(character_pos)
        
        if self.pulling_character and self.hit_pos:
            screen_hit = camera.world_to_screen(self.hit_pos)
            pygame.draw.line(screen, ORANGE, screen_start, screen_hit, max(1, int(camera.scale_size(3))))
            pygame.draw.circle(screen, RED, (int(screen_hit[0]), int(screen_hit[1])), int(camera.scale_size(5)))
        else:
            screen_current = camera.world_to_screen(self.current_pos)
            pygame.draw.line(screen, WHITE, screen_start, screen_current, max(1, int(camera.scale_size(2))))
            pygame.draw.circle(screen, RED, (int(screen_current[0]), int(screen_current[1])), int(camera.scale_size(3)))

class Character:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 12
        self.angle = 0
        self.speed = 0.012
        self.current_shape_id = 0
        self.being_pulled = False
        self.pull_target = None
        self.pull_target_shape = None
        self.shape_relative_angle = 0
        
        self.max_health = 100
        self.current_health = 100
    
    def start_being_pulled(self, target_pos, target_shape_id):
        self.being_pulled = True
        self.pull_target = target_pos
        self.pull_target_shape = target_shape_id
    
    def update_pull(self, shapes):
        if not self.being_pulled or not self.pull_target:
            return
        
        dx = self.pull_target[0] - self.x
        dy = self.pull_target[1] - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < PULL_SPEED:
            self.x = self.pull_target[0]
            self.y = self.pull_target[1]
            self.current_shape_id = self.pull_target_shape
            
            self.angle = self.calculate_angle_on_shape(shapes[self.current_shape_id])
            self.shape_relative_angle = self.angle
            
            self.being_pulled = False
            self.pull_target = None
            self.pull_target_shape = None
            
            return True
        else:
            self.x += (dx/distance) * PULL_SPEED
            self.y += (dy/distance) * PULL_SPEED
            return False
    
    def update_position_on_moving_shape(self, shapes):
        if self.being_pulled:
            return
        
        current_shape = shapes[self.current_shape_id]
        pos = current_shape.get_position_on_perimeter(self.angle)
        self.x, self.y = pos
    
    def calculate_angle_on_shape(self, shape):
        if shape.is_circle:
            dx = self.x - shape.center[0]
            dy = self.y - shape.center[1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            return angle / (2 * math.pi)
        else:
            min_dist = float('inf')
            best_angle = 0
            
            total_perimeter = shape.get_total_perimeter()
            current_distance = 0
            
            for i in range(len(shape.vertices)):
                start = shape.vertices[i]
                end = shape.vertices[(i + 1) % len(shape.vertices)]
                
                edge_dx = end[0] - start[0]
                edge_dy = end[1] - start[1]
                edge_length = math.sqrt(edge_dx*edge_dx + edge_dy*edge_dy)
                
                if edge_length > 0:
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
        
        eye_offset = camera.scale_size(4)
        eye_radius = camera.scale_size(2)
        left_eye = (int(screen_x - eye_offset), int(screen_y - camera.scale_size(3)))
        right_eye = (int(screen_x + eye_offset), int(screen_y - camera.scale_size(3)))
        pygame.draw.circle(screen, BLACK, left_eye, int(eye_radius))
        pygame.draw.circle(screen, BLACK, right_eye, int(eye_radius))
        
        smile_size = camera.scale_size(8)
        smile_rect = pygame.Rect(screen_x - smile_size/2, screen_y - camera.scale_size(1), smile_size, camera.scale_size(5))
        pygame.draw.arc(screen, BLACK, smile_rect, math.pi, 2 * math.pi, max(1, int(camera.scale_size(2))))
        
        cheek_radius = camera.scale_size(2)
        cheek_color = (255, 200, 200)
        left_cheek = (int(screen_x - camera.scale_size(7)), int(screen_y + camera.scale_size(1)))
        right_cheek = (int(screen_x + camera.scale_size(7)), int(screen_y + camera.scale_size(1)))
        pygame.draw.circle(screen, cheek_color, left_cheek, int(cheek_radius))
        pygame.draw.circle(screen, cheek_color, right_cheek, int(cheek_radius))

class Shape:
    """Shape class that uses precomputed data instead of real-time physics"""
    
    def __init__(self, shape_data):
        # Initialize from precomputed data
        self.shape_id = shape_data['shape_id']
        self.is_circle = shape_data['is_circle']
        self.center = shape_data['center']
        self.radius = shape_data['radius']
        self.vertices = shape_data['vertices']
        self.color = shape_data['color']
        self.mood = shape_data['mood']
        self.is_mother = shape_data['is_mother']
        self.mother_pulse_time = shape_data['mother_pulse_time']
        
        # Current state (will be updated from precomputed data)
        self.collision_bounds = shape_data['collision_bounds']
    
    def update_from_precomputed(self, shape_data):
        """Update shape state from precomputed frame data"""
        self.center = shape_data['center']
        self.radius = shape_data['radius']
        self.vertices = shape_data['vertices']
        self.collision_bounds = shape_data['collision_bounds']
        self.mother_pulse_time = shape_data['mother_pulse_time']
        # Note: mood, color, is_mother etc. don't change during simulation
    
    def get_center(self):
        if self.is_circle:
            return self.center
        
        center_x = sum(v[0] for v in self.vertices) / len(self.vertices)
        center_y = sum(v[1] for v in self.vertices) / len(self.vertices)
        return (center_x, center_y)
    
    def draw(self, screen, camera, is_active=False):
        color = self.color
        width = max(1, int(camera.scale_size(5 if is_active else 3)))
        
        if self.is_mother:
            pulse_factor = 0.8 + 0.4 * math.sin(self.mother_pulse_time)
            
            glow_size = max(1, int(camera.scale_size(8 * pulse_factor)))
            if self.is_circle:
                screen_center = camera.world_to_screen(self.center)
                glow_radius = camera.scale_size(self.radius + 10 * pulse_factor)
                
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                alpha = int(100 * pulse_factor)
                pygame.draw.circle(glow_surf, (255, 255, 245, alpha), 
                                 (glow_radius, glow_radius), glow_radius)
                screen.blit(glow_surf, (screen_center[0] - glow_radius, screen_center[1] - glow_radius))
            
            width = max(1, int(camera.scale_size(3 + 4 * pulse_factor)))
        
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
        
        if self.is_circle:
            screen_center = camera.world_to_screen(self.center)
            screen_radius = camera.scale_size(self.radius)
            
            if self.is_mother:
                pygame.draw.circle(screen, color, 
                                 (int(screen_center[0]), int(screen_center[1])), 
                                 int(screen_radius))
                pygame.draw.circle(screen, BLACK,
                                 (int(screen_center[0]), int(screen_center[1])), 
                                 int(screen_radius), max(1, int(camera.scale_size(2))))
            else:
                pygame.draw.circle(screen, color, 
                                 (int(screen_center[0]), int(screen_center[1])), 
                                 int(screen_radius), width)
        else:
            screen_vertices = [camera.world_to_screen(v) for v in self.vertices]
            if len(screen_vertices) > 2:
                if self.is_mother:
                    pygame.draw.polygon(screen, color, screen_vertices)
                    pygame.draw.polygon(screen, BLACK, screen_vertices, max(1, int(camera.scale_size(2))))
                else:
                    pygame.draw.polygon(screen, color, screen_vertices, width)
        
        self.draw_face(screen, camera)
    
    def draw_face(self, screen, camera):
        center = self.get_center()
        screen_center = camera.world_to_screen(center)
        
        if self.is_circle:
            face_scale = min(self.radius / 60, 1.5)
        else:
            min_x = min(v[0] for v in self.vertices)
            max_x = max(v[0] for v in self.vertices)
            min_y = min(v[1] for v in self.vertices)
            max_y = max(v[1] for v in self.vertices)
            avg_size = ((max_x - min_x) + (max_y - min_y)) / 4
            face_scale = min(avg_size / 60, 1.5)
        
        face_scale = max(0.3, face_scale)
        screen_face_scale = camera.scale_size(face_scale)
        
        eye_offset_x = camera.scale_size(12 * face_scale)
        eye_offset_y = camera.scale_size(8 * face_scale)
        eye_radius = max(1, int(camera.scale_size(3 * face_scale)))
        
        left_eye_pos = (int(screen_center[0] - eye_offset_x), int(screen_center[1] - eye_offset_y))
        right_eye_pos = (int(screen_center[0] + eye_offset_x), int(screen_center[1] - eye_offset_y))
        
        if self.mood == 'happy':
            pygame.draw.circle(screen, BLACK, left_eye_pos, eye_radius)
            pygame.draw.circle(screen, BLACK, right_eye_pos, eye_radius)
            
            smile_rect = pygame.Rect(
                screen_center[0] - camera.scale_size(15 * face_scale),
                screen_center[1] + camera.scale_size(5 * face_scale),
                camera.scale_size(30 * face_scale),
                camera.scale_size(15 * face_scale)
            )
            pygame.draw.arc(screen, BLACK, smile_rect, math.pi, 2 * math.pi, max(1, int(camera.scale_size(2 * face_scale))))
            
            if self.is_mother:
                cheek_radius = camera.scale_size(6 * face_scale)
                cheek_color = (255, 200, 200)
                left_cheek = (int(screen_center[0] - camera.scale_size(20 * face_scale)), 
                             int(screen_center[1] + camera.scale_size(5 * face_scale)))
                right_cheek = (int(screen_center[0] + camera.scale_size(20 * face_scale)), 
                              int(screen_center[1] + camera.scale_size(5 * face_scale)))
                pygame.draw.circle(screen, cheek_color, left_cheek, int(cheek_radius))
                pygame.draw.circle(screen, cheek_color, right_cheek, int(cheek_radius))
            
        elif self.mood == 'angry':
            brow_length = camera.scale_size(8 * face_scale)
            brow_width = max(1, int(camera.scale_size(2 * face_scale)))
            
            left_brow_start = (int(left_eye_pos[0] - brow_length//2), int(left_eye_pos[1] - camera.scale_size(6 * face_scale)))
            left_brow_end = (int(left_eye_pos[0] + brow_length//2), int(left_eye_pos[1] - camera.scale_size(3 * face_scale)))
            pygame.draw.line(screen, RED, left_brow_start, left_brow_end, brow_width)
            
            right_brow_start = (int(right_eye_pos[0] - brow_length//2), int(right_eye_pos[1] - camera.scale_size(3 * face_scale)))
            right_brow_end = (int(right_eye_pos[0] + brow_length//2), int(right_eye_pos[1] - camera.scale_size(6 * face_scale)))
            pygame.draw.line(screen, RED, right_brow_start, right_brow_end, brow_width)
            
            pygame.draw.circle(screen, RED, left_eye_pos, eye_radius)
            pygame.draw.circle(screen, RED, right_eye_pos, eye_radius)
            
            frown_rect = pygame.Rect(
                screen_center[0] - camera.scale_size(12 * face_scale),
                screen_center[1] + camera.scale_size(2 * face_scale),
                camera.scale_size(24 * face_scale),
                camera.scale_size(12 * face_scale)
            )
            pygame.draw.arc(screen, RED, frown_rect, 0, math.pi, max(1, int(camera.scale_size(2 * face_scale))))
    
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
        if self.is_circle:
            return self.line_circle_intersection(start_pos, end_pos)
        else:
            return self.line_polygon_intersection(start_pos, end_pos)
    
    def line_circle_intersection(self, start_pos, end_pos):
        cx, cy = self.center
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        dx = x2 - x1
        dy = y2 - y1
        
        fx = x1 - cx
        fy = y1 - cy
        
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - self.radius*self.radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None
        
        discriminant = math.sqrt(discriminant)
        
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        for t in [t1, t2]:
            if 0 <= t <= 1:
                hit_x = x1 + t * dx
                hit_y = y1 + t * dy
                return (hit_x, hit_y)
        
        return None
    
    def line_polygon_intersection(self, start_pos, end_pos):
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

class BackgroundShape:
    """Background shape using precomputed data"""
    
    def __init__(self, bg_data):
        self.x = bg_data['x']
        self.y = bg_data['y']
        self.size = bg_data['size']
        self.alpha = bg_data['alpha']
        self.is_circle = bg_data['is_circle']
        self.vertices = bg_data.get('vertices', None)

    def draw(self, screen):
        shape_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        if self.is_circle:
            pygame.draw.circle(shape_surface, (255, 255, 255, self.alpha), (self.x, self.y), self.size)
        elif self.vertices:
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
    
    for bg_shape in background_shapes:
        bg_shape.draw(screen)

def draw_crosshair(screen, pos, camera):
    x, y = pos
    size = camera.scale_size(15)
    line_width = max(1, int(camera.scale_size(2)))
    pygame.draw.line(screen, WHITE, (x-size, y), (x+size, y), line_width)
    pygame.draw.line(screen, WHITE, (x, y-size), (x, y+size), line_width)
    pygame.draw.circle(screen, WHITE, (int(x), int(y)), int(size//2), max(1, int(camera.scale_size(1))))

class DirectionalArrow:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.arrow_size = 35
        self.orbit_radius = 80
        self.orbit_angle = 0
        self.orbit_speed = 1
    
    def draw(self, screen, character, mother_shape, camera):
        char_screen_pos = camera.world_to_screen((character.x, character.y))
        mother_center = mother_shape.get_center()
        
        dx = mother_center[0] - character.x
        dy = mother_center[1] - character.y
        target_angle = math.degrees(math.atan2(dy, dx))
        
        self.orbit_angle += self.orbit_speed
        if self.orbit_angle >= 360:
            self.orbit_angle = 0
        
        orbit_rad = math.radians(self.orbit_angle)
        arrow_pos = (
            char_screen_pos[0] + math.cos(orbit_rad) * self.orbit_radius,
            char_screen_pos[1] + math.sin(orbit_rad) * self.orbit_radius
        )
        angle = target_angle
        
        arrow_points = [
            (self.arrow_size, 0),
            (0, -self.arrow_size//2),
            (self.arrow_size//3, 0),
            (0, self.arrow_size//2),
        ]
        
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        
        rotated_points = []
        for px, py in arrow_points:
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            rotated_points.append((arrow_pos[0] + rx, arrow_pos[1] + ry))
        
        glow_color = (255, 215, 0, 150)
        main_color = (255, 223, 0)
        
        glow_points = []
        for px, py in arrow_points:
            px *= 1.5
            py *= 1.5
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            glow_points.append((arrow_pos[0] + rx, arrow_pos[1] + ry))
        
        glow_surf = pygame.Surface((self.arrow_size * 3, self.arrow_size * 3), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surf, glow_color, 
                           [(p[0] - arrow_pos[0] + self.arrow_size * 1.5, 
                             p[1] - arrow_pos[1] + self.arrow_size * 1.5) for p in glow_points])
        screen.blit(glow_surf, (arrow_pos[0] - self.arrow_size * 1.5, arrow_pos[1] - self.arrow_size * 1.5))
        
        pygame.draw.polygon(screen, main_color, rotated_points)
        pygame.draw.polygon(screen, (255, 235, 122), rotated_points, 3)

class FloatingAsset:
    def __init__(self, image_path, x, y, max_drift=30, scale=1.0):
        self.original_image = pygame.image.load(image_path)
        self.current_scale = scale
        self.rescale_image(scale)
        
        self.original_x = x
        self.original_y = y
        self.max_drift = max_drift
        
        self.start_x = max(0, min(x, SCREEN_WIDTH - self.image.get_width()))
        self.start_y = max(0, min(y, SCREEN_HEIGHT - self.image.get_height()))
        self.x = self.start_x
        self.y = self.start_y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
    
    def rescale_image(self, scale):
        new_size = (int(self.original_image.get_width() * scale), 
                   int(self.original_image.get_height() * scale))
        self.image = pygame.transform.smoothscale(self.original_image, new_size)
        self.current_scale = scale
    
    def get_rect(self):
        drift_rect = pygame.Rect(
            self.start_x - self.max_drift,
            self.start_y - self.max_drift,
            self.image.get_width() + 2 * self.max_drift,
            self.image.get_height() + 2 * self.max_drift
        )
        return drift_rect
    
    def overlaps(self, other):
        return self.get_rect().colliderect(other.get_rect())
    
    def update(self):
        self.vx += random.uniform(-0.1, 0.1)
        self.vy += random.uniform(-0.1, 0.1)
        
        self.vx *= 0.98
        self.vy *= 0.98
        
        new_x = self.x + self.vx
        new_y = self.y + self.vy
        
        if new_x < 0 or new_x > SCREEN_WIDTH - self.image.get_width():
            self.vx *= -0.5
            new_x = max(0, min(new_x, SCREEN_WIDTH - self.image.get_width()))
        if new_y < 0 or new_y > SCREEN_HEIGHT - self.image.get_height():
            self.vy *= -0.5
            new_y = max(0, min(new_y, SCREEN_HEIGHT - self.image.get_height()))
        
        dx = new_x - self.start_x
        dy = new_y - self.start_y
        if abs(dx) > self.max_drift:
            new_x = self.start_x + (self.max_drift if dx > 0 else -self.max_drift)
            self.vx *= -0.5
        if abs(dy) > self.max_drift:
            new_y = self.start_y + (self.max_drift if dy > 0 else -self.max_drift)
            self.vy *= -0.5
            
        self.x = new_x
        self.y = new_y
    
    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

class StartScreen:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.done = False
        
        self.margin = 60
        self.min_spacing = 40
        
        self.assets = self.create_layout()
        
        self.button_rect = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT - 100, 200, 50)
        self.button_color = (46, 204, 113)
        self.button_hover_color = (39, 174, 96)
        self.button_text = "Start Game"
        self.font = pygame.font.Font(None, 36)
    
    def create_layout(self):
        assets = []
        title_margin = 20
        
        base_scale = 0.8
        title_scale = 1.0
        max_attempts = 10
        
        while max_attempts > 0:
            assets.clear()
            overlapping = False
            
            title = FloatingAsset('assets/game-title.png',
                                SCREEN_WIDTH//2 - 150, title_margin,
                                max_drift=10,
                                scale=title_scale)
            
            available_height = SCREEN_HEIGHT - title.image.get_height() - 2 * self.margin
            
            positions = [
                (self.margin, title.image.get_height() + self.margin),
                (SCREEN_WIDTH//2 - 50, title.image.get_height() + self.margin),
                (SCREEN_WIDTH - self.margin - 100, title.image.get_height() + self.margin),
                (self.margin + 50, SCREEN_HEIGHT - self.margin - 100),
                (SCREEN_WIDTH//2, SCREEN_HEIGHT - self.margin - 100),
                (SCREEN_WIDTH - self.margin - 150, SCREEN_HEIGHT - self.margin - 100)
            ]
            
            temp_assets = [
                FloatingAsset('assets/main-character-shape.png', positions[0][0], positions[0][1], max_drift=20, scale=base_scale),
                FloatingAsset('assets/angry-shape.png', positions[1][0], positions[1][1], max_drift=20, scale=base_scale),
                FloatingAsset('assets/angry-shape.png', positions[2][0], positions[2][1], max_drift=20, scale=base_scale),
                FloatingAsset('assets/happy-shape.png', positions[3][0], positions[3][1], max_drift=20, scale=base_scale),
                FloatingAsset('assets/happy-shape.png', positions[4][0], positions[4][1], max_drift=20, scale=base_scale),
                FloatingAsset('assets/happy-shape.png', positions[5][0], positions[5][1], max_drift=20, scale=base_scale)
            ]
            
            for i, asset in enumerate(temp_assets):
                if asset.overlaps(title):
                    overlapping = True
                    break
                for j in range(i):
                    if asset.overlaps(temp_assets[j]):
                        overlapping = True
                        break
                if overlapping:
                    break
            
            if not overlapping:
                assets.append(title)
                assets.extend(temp_assets)
                break
            
            base_scale *= 0.9
            title_scale *= 0.95
            max_attempts -= 1
        
        if not assets:
            base_scale = 0.4
            title_scale = 0.6
            assets.append(FloatingAsset('assets/game-title.png',
                                      SCREEN_WIDTH//2 - 100, title_margin,
                                      max_drift=10,
                                      scale=title_scale))
            
            for pos, img in zip(positions, [
                'assets/main-character-shape.png',
                'assets/angry-shape.png',
                'assets/angry-shape.png',
                'assets/happy-shape.png',
                'assets/happy-shape.png',
                'assets/happy-shape.png'
            ]):
                assets.append(FloatingAsset(img, pos[0], pos[1],
                                          max_drift=20,
                                          scale=base_scale))
        
        return assets
    
    def update(self):
        for asset in self.assets:
            asset.update()
            
        mouse_pos = pygame.mouse.get_pos()
        self.button_hover = self.button_rect.collidepoint(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.MOUSEBUTTONDOWN and self.button_hover:
                self.done = True
        
        return False
    
    def draw(self):
        # Draw gradient background
        for y in range(0, SCREEN_HEIGHT, 2):
            progress = y / SCREEN_HEIGHT
            color = (
                int(69 + (30 - 69) * progress),
                int(183 + (60 - 183) * progress),
                int(209 + (114 - 209) * progress)
            )
            pygame.draw.line(surface, color, (0, y), (SCREEN_WIDTH, y))
        return surface
    
    def draw(self):
        # Draw gradient background
        self.screen.blit(self.gradient_surface, (0, 0))
        
        for asset in self.assets:
            asset.draw(self.screen)
        
        color = self.button_hover_color if self.button_hover else self.button_color
        pygame.draw.rect(self.screen, color, self.button_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.button_rect, width=2, border_radius=10)
        
        text = self.font.render(self.button_text, True, (255, 255, 255))
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)
        
        pygame.display.flip()

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.start_screen = StartScreen(self.screen)
        self.game_started = False
        
        self.game_won = False
        self.game_over = False
        self.win_time = 0
        self.shapes_frozen = False
        
        self.dt = 1.0 / FPS
        self.game_time = 0
    
    def init_game(self):
        """Initialize game components after start screen"""
        # Load precomputed movement data
        movement_data.load_data()
        
        self.screen_flash_timer = 0
        self.last_damage_time = 0
        
        self.camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.arrow = DirectionalArrow(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        self.setup_game_objects()
        
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 48)
        self.win_font = pygame.font.Font(None, 72)
        
        self.mouse_pos = (0, 0)
        self.game_time = 0
        
        if not pygame.mixer.music.get_busy():
            start_background_music()
    
    def setup_game_objects(self):
        """Create all game objects using precomputed data"""
        # Get initial frame data
        initial_frame = movement_data.get_frame_data(0.0)
        if not initial_frame:
            print("ERROR: Could not load initial frame data!")
            sys.exit(1)
        
        # Create shapes from initial frame data
        self.shapes = []
        for shape_data in initial_frame['shapes']:
            shape = Shape(shape_data)
            self.shapes.append(shape)
        
        # Get mother shape ID from metadata
        self.mother_shape_id = movement_data.metadata['mother_shape_id']
        
        # Find a good starting position on a happy shape
        happy_shapes = [i for i, shape in enumerate(self.shapes) if shape.mood == 'happy']
        if not happy_shapes:
            char_start_shape = 0
            self.shapes[0].mood = 'happy'
        else:
            char_start_shape = random.choice(happy_shapes)
        
        start_shape = self.shapes[char_start_shape]
        start_pos = start_shape.get_position_on_perimeter(0.0)
        
        # Create character at starting position
        self.character = Character(start_pos[0], start_pos[1])
        self.character.current_shape_id = char_start_shape
        
        # Create harpoon
        self.harpoon = Harpoon()
        
        # Log initialization stats
        mother_center = self.shapes[self.mother_shape_id].get_center()
        final_distance = math.sqrt((start_pos[0] - mother_center[0])**2 + 
                                 (start_pos[1] - mother_center[1])**2)
        
        print(f"Game initialized with precomputed data")
        print(f"Total shapes: {len(self.shapes)} (including mother)")
        print(f"Mother shape: Shape {self.mother_shape_id}")
        print(f"Distance to mother: {final_distance:.0f} pixels")
        
        happy_count = sum(1 for shape in self.shapes if shape.mood == 'happy')
        print(f"Shape moods: {happy_count} Happy, {len(self.shapes)-happy_count} Angry")
        
        pygame.display.set_caption(f"Find Your Mother! - {len(self.shapes)} Shapes to Explore (Optimized)")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.game_won or self.game_over:
                    self.__init__()
                    return True
                elif event.button == 1 and not self.harpoon.active and not self.character.being_pulled:
                    char_pos = (self.character.x, self.character.y)
                    world_mouse_pos = self.camera.screen_to_world(self.mouse_pos)
                    self.harpoon.launch(char_pos, world_mouse_pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif (self.game_won or self.game_over) and event.key == pygame.K_SPACE:
                    self.__init__()
                    return True
        return True
    
    def check_win_condition(self):
        if (self.character.current_shape_id == self.mother_shape_id and 
            not self.character.being_pulled and not self.harpoon.active):
            if not self.game_won:
                self.game_won = True
                self.win_time = pygame.time.get_ticks()
                self.shapes_frozen = True
                
                try:
                    victory_sound = pygame.mixer.Sound('assets/victory.ogg')
                    victory_sound.set_volume(0.6)
                    victory_sound.play()
                    pygame.mixer.music.stop()
                except Exception as e:
                    print(f"Could not play victory sound: {e}")
                
                print("GAME WON! Player found their mother!")
    
    def update(self):
        if self.game_won or self.game_over:
            return
        
        keys = pygame.key.get_pressed()
        
        self.game_time += self.dt
        
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= self.dt
        
        # Health system
        if not self.character.being_pulled:
            current_shape = self.shapes[self.character.current_shape_id]
            
            if current_shape.mood == 'angry':
                damage_this_frame = DAMAGE_RATE * self.dt
                old_health = self.character.current_health
                self.character.current_health = max(0, self.character.current_health - damage_this_frame)
                
                if old_health > self.character.current_health and self.screen_flash_timer <= 0:
                    self.screen_flash_timer = 0.15
                
            elif current_shape.mood == 'happy':
                heal_this_frame = HEAL_RATE * self.dt
                self.character.current_health = min(self.character.max_health, self.character.current_health + heal_this_frame)
        
        if self.character.current_health <= 0:
            self.game_over = True
            self.shapes_frozen = True
            return
        
        # Update shapes from precomputed data (instead of physics simulation)
        if not self.shapes_frozen:
            frame_data = movement_data.get_interpolated_frame(self.game_time)
            if frame_data:
                for i, shape_data in enumerate(frame_data['shapes']):
                    if i < len(self.shapes):
                        self.shapes[i].update_from_precomputed(shape_data)
        
        # Update character position relative to moving shape
        self.character.update_position_on_moving_shape(self.shapes)
        
        # Update camera
        self.camera.update(self.character.x, self.character.y)
        
        # Update harpoon
        self.harpoon.update((self.character.x, self.character.y))
        
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
            
            if pull_completed:
                self.harpoon.active = False
                self.check_win_condition()
        elif not self.harpoon.active:
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.character.angle -= self.character.speed
                if self.character.angle < 0:
                    self.character.angle += 1
            
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.character.angle += self.character.speed
                if self.character.angle > 1:
                    self.character.angle -= 1
            
            current_shape = self.shapes[self.character.current_shape_id]
            pos = current_shape.get_position_on_perimeter(self.character.angle)
            self.character.x, self.character.y = pos
        else:
            current_shape = self.shapes[self.character.current_shape_id]
            pos = current_shape.get_position_on_perimeter(self.character.angle)
            self.character.x, self.character.y = pos
        
        if self.harpoon.pulling_character and not self.character.being_pulled and not self.game_won:
            self.harpoon.active = False
    
    def draw(self):
        # Get current frame data for background
        frame_data = movement_data.get_interpolated_frame(self.game_time)
        background_time = frame_data['background_time'] if frame_data else 0
        
        # Create background shapes from frame data
        background_shapes = []
        if frame_data and 'background_shapes' in frame_data:
            for bg_data in frame_data['background_shapes']:
                background_shapes.append(BackgroundShape(bg_data))
        
        # Background
        create_gradient_background(self.screen, background_time, background_shapes)
        
        # Draw only visible shapes (frustum culling)
        for i, shape in enumerate(self.shapes):
            # Only draw shapes that are visible on screen
            if self.camera.is_visible(shape):
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
        self.harpoon.draw(self.screen, self.camera, (self.character.x, self.character.y))
        
        # Draw character
        self.character.draw(self.screen, self.camera)
        
        # Draw directional arrow
        if not self.game_won and not self.game_over:
            mother_shape = self.shapes[self.mother_shape_id]
            self.arrow.draw(self.screen, self.character, mother_shape, self.camera)
        
        # Draw crosshair
        if not self.harpoon.active and not self.character.being_pulled and not self.game_won and not self.game_over:
            draw_crosshair(self.screen, self.mouse_pos, self.camera)
        
        # Draw screen flash effect
        if self.screen_flash_timer > 0:
            flash_alpha = int(100 * (self.screen_flash_timer / 0.15))
            flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))
        
        # Draw health bar
        if not self.game_won and not self.game_over:
            self.draw_health_bar()
        
        # Draw UI overlays
        if self.game_over:
            overlay_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay_surface.fill((0, 0, 0, 150))
            self.screen.blit(overlay_surface, (0, 0))
            
            game_over_text = self.win_font.render("GAME OVER", True, RED)
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(game_over_text, game_over_rect)
            
            subtitle = self.title_font.render("Your health reached zero!", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(subtitle, subtitle_rect)
            
            restart_text = self.font.render("Click anywhere or press SPACE to try again", True, YELLOW)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
            self.screen.blit(restart_text, restart_rect)
            
        elif self.game_won:
            win_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            win_surface.fill((0, 0, 0, 150))
            self.screen.blit(win_surface, (0, 0))
            
            win_text = self.win_font.render("REUNITED!", True, MOTHER_COLOR)
            win_rect = win_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(win_text, win_rect)
            
            subtitle = self.title_font.render("You found your mother!", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(subtitle, subtitle_rect)
            
            restart_text = self.font.render("Click anywhere or press SPACE to play again", True, YELLOW)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
            self.screen.blit(restart_text, restart_rect)
        else:
            title = self.title_font.render("Find Your Mother!", True, MOTHER_COLOR)
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 30))
            self.screen.blit(title, title_rect)
            
            current_shape = self.shapes[self.character.current_shape_id]
            mood_color = LIGHT_GREEN if current_shape.mood == 'happy' else RED
            mood_text = f"Current Shape: {'[Happy]' if current_shape.mood == 'happy' else '[Angry]'} - {'Healing' if current_shape.mood == 'happy' else 'Taking Damage!'}"
            
            mood_surface = self.font.render(mood_text, True, mood_color)
            mood_rect = mood_surface.get_rect(center=(SCREEN_WIDTH // 2, 70))
            self.screen.blit(mood_surface, mood_rect)
            
            instructions = [
                "[!] Navigate to the pulsating golden mother shape to win! (OPTIMIZED)",
                "(*) Happy shapes (smiling) heal you over time",
                "(!) Angry shapes (frowning) drain your health - avoid staying too long!",
                ">>> Use harpoon (Left Click) to escape dangerous shapes",
                "--> Follow the golden arrow when mother is off-screen"
            ]
            
            y_start = 100
            for i, instruction in enumerate(instructions):
                if "mother" in instruction.lower():
                    color = MOTHER_COLOR
                elif "(*)" in instruction or "heal" in instruction.lower():
                    color = LIGHT_GREEN
                elif "(!)" in instruction or "drain" in instruction.lower() or "dangerous" in instruction.lower():
                    color = RED
                else:
                    color = WHITE
                
                text = self.font.render(instruction, True, color)
                self.screen.blit(text, (10, y_start + i * 25))
        
        pygame.display.flip()
    
    def draw_health_bar(self):
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 20
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, RED, bg_rect)
        
        health_ratio = self.character.current_health / self.character.max_health
        health_width = int(bar_width * health_ratio)
        if health_width > 0:
            health_rect = pygame.Rect(bar_x, bar_y, health_width, bar_height)
            pygame.draw.rect(self.screen, LIGHT_GREEN, health_rect)
        
        pygame.draw.rect(self.screen, WHITE, bg_rect, 2)
        
        health_text = f"Health: {int(self.character.current_health)}/{self.character.max_health}"
        health_surface = self.font.render(health_text, True, WHITE)
        self.screen.blit(health_surface, (bar_x, bar_y + bar_height + 5))
    
    async def run(self):
        running = True
        while running:
            if not self.game_started:
                running = not self.start_screen.update()
                if running:
                    self.start_screen.draw()
                    if self.start_screen.done:
                        self.game_started = True
                        self.init_game()
                    else:
                        self.clock.tick(FPS)
                        await asyncio.sleep(0)
                        continue
            
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
            await asyncio.sleep(0)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    asyncio.run(game.run())
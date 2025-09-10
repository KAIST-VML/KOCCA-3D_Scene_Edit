# scene_composition_pytorch3d.py - Scene composition using PyTorch3D and Open3D
import os
import sys
import numpy as np
import re
import json
import math
import trimesh
from typing import Dict, List, Optional, Tuple
import open3d as o3d

class SceneComfaposer:
    def __init__(self, root_path=None, clip_results_path="/source/sumin/stylin/FlairGPT/retrieval/clip_rerank_results"):
        """
        Initialize the scene composer with standardized folder structure
        
        Args:
            root_path: Root directory path. If None, uses script directory.
            clip_results_path: Path to CLIP rerank results folder
        """
        # Determine root path
        if root_path is None:
            self.root_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.root_path = os.path.abspath(root_path)
            
        print(f"Using root path: {self.root_path}")
        
        # Set up standardized paths
        self.clip_results_path = clip_results_path
        self.layout_file_path = os.path.join(self.root_path, "layout.txt")
        self.background_color_path = os.path.join(self.root_path, "background_color.txt")
        self.output_path = os.path.join(self.root_path, "output")
        
        # 3D-FUTURE dataset paths
        self.dataset_paths = [
                "../../dataset/3D-FUTURE-model-part1",
                "../../dataset/3D-FUTURE-model-part2", 
                "../../dataset/3D-FUTURE-model-part3",
                "../../dataset/3D-FUTURE-model-part4"
        ]        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Validate required files
        self.validate_structure()
        
        # Parse configuration files
        self.layout_data = self.parse_layout_file()
        self.background_colors = self.parse_background_colors()
        
        # Store all scene meshes
        self.scene_meshes = []
        
    def validate_structure(self):
        """Validate that required files exist"""
        required_paths = [
            (self.layout_file_path, "layout.txt"),
        ]
        
        missing = []
        for path, name in required_paths:
            if not os.path.exists(path):
                missing.append(name)
                
        if missing:
            print(f"❌ Missing required files: {', '.join(missing)}")
            print(f"Expected structure in: {self.root_path}")
            print("Required:")
            print("  - layout.txt")
            sys.exit(1)
        else:
            print("✅ File structure validated")
    
    # parse_layout_file 함수 수정 (색상 로딩 개선)
    def parse_layout_file(self):
        """Parse the layout text file and extract room and object information - FIXED VERSION"""
        layout_data = {
            'room_width': None,
            'room_length': None,
            'objects': {},
            'style': None,
            'prompt': None
        }
        
        print(f"📖 Reading layout file: {self.layout_file_path}")
        
        with open(self.layout_file_path, 'r') as f:
            content = f.read()
        
        # Debug: Show first part of content
        print(f"📄 Layout file preview:")
        lines = content.split('\n')[:10]  # First 10 lines
        for i, line in enumerate(lines):
            print(f"  {i+1:2d}: {line}")
        total_lines = len(content.split('\n'))
        if total_lines > 10:
            print(f"  ... (total {total_lines} lines)")
        
        # Extract room dimensions
        room_width_match = re.search(r'room_width:\s*(\d+(?:\.\d+)?)', content)
        room_length_match = re.search(r'room_length:\s*(\d+(?:\.\d+)?)', content)
        
        if room_width_match:
            layout_data['room_width'] = float(room_width_match.group(1))
            print(f"🏠 Found room width: {layout_data['room_width']}")
        if room_length_match:
            layout_data['room_length'] = float(room_length_match.group(1))
            print(f"🏠 Found room length: {layout_data['room_length']}")
        
        # Extract prompt
        prompt_match = re.search(r'prompt:\s*(.+)', content)
        if prompt_match:
            layout_data['prompt'] = prompt_match.group(1)
            print(f"📝 Found prompt: {layout_data['prompt'][:50]}...")
        
        # FIXED: Extract ALL objects from position format (including doors and windows)
        print(f"\n🔍 Searching for ALL objects with position data...")
        
        # Updated pattern to catch all object names (including with numbers)
        object_pattern = r'(\w+\d*):\s*\{\'position\':\s*\(([\d\w\.\(\), -]+)\),\s*\'width\':\s*([\d\.]+),\s*\'length\':\s*([\d\.]+)\}'
        
        all_objects_found = 0
        doors_windows_found = 0
        furniture_found = 0
        
        for match in re.finditer(object_pattern, content):
            obj_name = match.group(1)
            all_objects_found += 1
            
            # Parse position tuple
            position_str = match.group(2)
            pos_numbers = re.findall(r'np\.float64\(([\d\.-]+)\)|(\d+(?:\.\d+)?)', position_str)
            position = []
            for num_match in pos_numbers:
                if num_match[0]:  # np.float64 format
                    position.append(float(num_match[0]))
                elif num_match[1]:  # regular float format
                    position.append(float(num_match[1]))
            
            if len(position) >= 3:
                obj_data = {
                    'position': (position[0], position[1], position[2]),
                    'width': float(match.group(3)),
                    'length': float(match.group(4))
                }
                
                # Check if it's a door or window
                is_door_window = any(keyword in obj_name.lower() for keyword in ['door', 'window'])
                
                if is_door_window:
                    doors_windows_found += 1
                    print(f"  🚪/🪟 Found door/window: '{obj_name}' at {obj_data['position']}, size {obj_data['width']}x{obj_data['length']}")
                else:
                    furniture_found += 1
                    print(f"  🪑 Found furniture: '{obj_name}' at {obj_data['position']}, size {obj_data['width']}x{obj_data['length']}")
                
                layout_data['objects'][obj_name] = obj_data
            else:
                print(f"  ❌ Invalid position data for {obj_name}: {position_str}")
        
        print(f"\n📊 Parsing summary:")
        print(f"  📦 Total objects found: {all_objects_found}")
        print(f"  🚪 Doors/Windows: {doors_windows_found}")
        print(f"  🪑 Furniture: {furniture_found}")
        
        # Extract numbered object names (for reference only, already parsed above)
        pattern = re.compile(
            r'^\s*(\d+)[\.\)\-:\s]+\*\*([^*]+)\*\*[\s\S]*?(?=^\s*\d+[\.\)\-:\s]+\*\*|$\Z)',
            re.MULTILINE
        )
        
        numbered_objects = []
        for match in pattern.finditer(content):
            object_name = match.group(2).strip()
            numbered_objects.append(object_name)
        
        if numbered_objects:
            print(f"📋 Found {len(numbered_objects)} numbered objects (for reference): {numbered_objects}")
        
        # Extract style description
        style_match = re.search(r'style:\s*(.+)', content, re.DOTALL)
        if style_match:
            layout_data['style'] = style_match.group(1).strip()
            print(f"🎨 Found style: {layout_data['style'][:100]}...")
        
        print(f"\n✅ Final object list:")
        for name, data in layout_data['objects'].items():
            obj_type = "🚪/🪟" if any(keyword in name.lower() for keyword in ['door', 'window']) else "🪑"
            print(f"  {obj_type} {name}: pos={data['position']}, size={data['width']}x{data['length']}")
        
        return layout_data

    # parse_background_colors 함수 수정 (layout.txt 색상 우선 사용)
    def parse_background_colors(self):
        """Parse layout.txt and extract floor and wall colors from both formats"""
        # 기본 색상값 설정
        colors = {
            'floor_color': (0.8, 0.6, 0.4),  # 기본 바닥 색상
            'wall_color': (0.6, 0.7, 0.6)    # 기본 벽 색상
        }
        
        # layout.txt에서 색상 정보 파싱
        if hasattr(self, 'layout_file_path') and os.path.exists(self.layout_file_path):
            print(f"📄 Parsing colors from layout.txt: {self.layout_file_path}")
            try:
                with open(self.layout_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 두 가지 형식 모두 시도
                colors = self.parse_colors_from_content(content, colors)
                        
            except Exception as e:
                print(f"❌ Error reading layout.txt: {e}")
                print("Using default colors")
        else:
            print(f"📄 Layout file not found, using default colors")
            
        print(f"🎨 Final colors - Floor: {colors['floor_color']}, Wall: {colors['wall_color']}")
        return colors

    def parse_colors_from_content(self, content, default_colors):
        """Parse colors from content with multiple format support"""
        colors = default_colors.copy()
        
        # 우선순위 1: 간단한 형식 (wall_color: (0.95, 0.95, 0.95, 1.0))
        print("🔍 Trying simple format first...")
        
        # wall_color 간단한 형식
        wall_simple_match = re.search(r"wall_color\s*:\s*\(([^)]+)\)", content)
        if wall_simple_match:
            try:
                rgba_str = wall_simple_match.group(1)
                rgba_values = [float(x.strip()) for x in rgba_str.split(',')]
                if len(rgba_values) >= 3:
                    colors['wall_color'] = tuple(rgba_values[:3])  # RGB만 사용
                    print(f"🎨 Parsed wall color (simple format): {colors['wall_color']}")
            except Exception as e:
                print(f"❌ Error parsing wall color (simple): {e}")
        
        # floor_color 간단한 형식
        floor_simple_match = re.search(r"floor_color\s*:\s*\(([^)]+)\)", content)
        if floor_simple_match:
            try:
                rgba_str = floor_simple_match.group(1)
                rgba_values = [float(x.strip()) for x in rgba_str.split(',')]
                if len(rgba_values) >= 3:
                    colors['floor_color'] = tuple(rgba_values[:3])  # RGB만 사용
                    print(f"🎨 Parsed floor color (simple format): {colors['floor_color']}")
            except Exception as e:
                print(f"❌ Error parsing floor color (simple): {e}")
        
        # 우선순위 2: 딕셔너리 형식 (간단한 형식에서 찾지 못한 경우만)
        if colors['wall_color'] == default_colors['wall_color'] or colors['floor_color'] == default_colors['floor_color']:
            print("🔍 Trying dictionary format...")
            
            # wall_color 딕셔너리 형식 (간단한 형식에서 못 찾은 경우만)
            if colors['wall_color'] == default_colors['wall_color']:
                wall_dict_match = re.search(r"'wall_color'\s*:\s*\{[^}]*'rgba'\s*:\s*\(([^)]+)\)", content)
                if wall_dict_match:
                    try:
                        rgba_str = wall_dict_match.group(1)
                        rgba_values = [float(x.strip()) for x in rgba_str.split(',')]
                        if len(rgba_values) >= 3:
                            colors['wall_color'] = tuple(rgba_values[:3])  # RGB만 사용
                            print(f"🎨 Parsed wall color (dict format): {colors['wall_color']}")
                    except Exception as e:
                        print(f"❌ Error parsing wall color (dict): {e}")
            
            # floor_color 딕셔너리 형식 (간단한 형식에서 못 찾은 경우만)
            if colors['floor_color'] == default_colors['floor_color']:
                floor_dict_match = re.search(r"'floor_color'\s*:\s*\{[^}]*'rgba'\s*:\s*\(([^)]+)\)", content)
                if floor_dict_match:
                    try:
                        rgba_str = floor_dict_match.group(1)
                        rgba_values = [float(x.strip()) for x in rgba_str.split(',')]
                        if len(rgba_values) >= 3:
                            colors['floor_color'] = tuple(rgba_values[:3])  # RGB만 사용
                            print(f"🎨 Parsed floor color (dict format): {colors['floor_color']}")
                    except Exception as e:
                        print(f"❌ Error parsing floor color (dict): {e}")
        
        return colors

    
    def create_room_mesh(self):
        """Create room mesh with floor and walls - ENHANCED VERSION"""
        width = self.layout_data['room_width']
        length = self.layout_data['room_length']
        height = 3.0  # Standard room height
        wall_thickness = 0.1
        
        print(f"🏠 Creating room: {width}m x {length}m x {height}m")
        
        # Create floor
        floor_mesh = self.create_floor_mesh(width, length)
        self.scene_meshes.append({
            'mesh': floor_mesh,
            'name': 'floor',
            'color': self.background_colors['floor_color']
        })
        
        # Create walls with openings - ENHANCED
        wall_meshes = self.create_wall_meshes_enhanced(width, length, height, wall_thickness)
        for wall_mesh, wall_name in wall_meshes:
            self.scene_meshes.append({
                'mesh': wall_mesh,
                'name': wall_name,
                'color': self.background_colors['wall_color']
            })
    
    def create_floor_mesh(self, width, length):
        """Create floor mesh"""
        # Create floor as a box mesh
        floor_mesh = trimesh.creation.box(
            extents=[width, length, 0.01],
            transform=trimesh.transformations.translation_matrix([width/2, length/2, -0.005])
        )
        return floor_mesh
    
    def prepare_mesh_for_boolean(self, mesh):
        """Check if mesh is ready for boolean operations"""
        try:
            # Check if mesh is watertight
            if hasattr(mesh, 'is_watertight'):
                return mesh.is_watertight
            # Fallback check
            return len(mesh.vertices) > 0 and len(mesh.faces) > 0
        except:
            return False

    def fix_mesh_for_boolean(self, mesh):
        """Attempt to fix mesh for boolean operations"""
        try:
            print(f"    🔧 Fixing mesh for boolean operations...")
            
            # Remove duplicate vertices and faces
            if hasattr(mesh, 'remove_duplicate_faces'):
                mesh.remove_duplicate_faces()
            if hasattr(mesh, 'remove_degenerate_faces'):
                mesh.remove_degenerate_faces()
            
            # Fix normals
            if hasattr(mesh, 'fix_normals'):
                mesh.fix_normals()
            
            # Fill holes if possible
            if hasattr(mesh, 'fill_holes'):
                mesh.fill_holes()
            
            # Process mesh to clean it up
            if hasattr(mesh, 'process'):
                mesh.process()
            
            print(f"    ✅ Mesh fixed")
            return mesh
            
        except Exception as e:
            print(f"    ⚠️  Mesh fixing failed: {e}")
            return mesh

    def manual_mesh_subtraction(self, wall_mesh, cutter_mesh, opening, wall_name):
        """Manual mesh subtraction as fallback method"""
        print(f"    🔧 Performing manual mesh subtraction...")
        
        # Get wall and cutter bounds
        wall_bounds = wall_mesh.bounds
        cutter_bounds = cutter_mesh.bounds
        
        # Create a new wall mesh by removing vertices/faces that overlap with cutter
        vertices = wall_mesh.vertices.copy()
        faces = wall_mesh.faces.copy()
        
        # Find vertices inside the cutter bounds (simplified approach)
        inside_mask = np.ones(len(vertices), dtype=bool)
        
        for i, vertex in enumerate(vertices):
            # Check if vertex is inside cutter bounds
            if (cutter_bounds[0][0] <= vertex[0] <= cutter_bounds[1][0] and
                cutter_bounds[0][1] <= vertex[1] <= cutter_bounds[1][1] and
                cutter_bounds[0][2] <= vertex[2] <= cutter_bounds[1][2]):
                inside_mask[i] = False
        
        # Keep only vertices outside the cutter
        outside_vertices = vertices[inside_mask]
        
        # Map old vertex indices to new indices
        vertex_map = {}
        new_idx = 0
        for old_idx, keep in enumerate(inside_mask):
            if keep:
                vertex_map[old_idx] = new_idx
                new_idx += 1
        
        # Filter faces that reference removed vertices
        new_faces = []
        for face in faces:
            if all(idx in vertex_map for idx in face):
                new_face = [vertex_map[idx] for idx in face]
                new_faces.append(new_face)
        
        if len(outside_vertices) > 0 and len(new_faces) > 0:
            # Create new mesh
            new_mesh = trimesh.Trimesh(vertices=outside_vertices, faces=new_faces)
            return new_mesh
        else:
            print(f"    ⚠️  Manual subtraction resulted in empty mesh")
            return wall_mesh
    
    def debug_wall_cutting(self, wall_mesh, cutter_mesh, opening, wall_name):
        """Debug function to understand why cutting isn't working"""
        print(f"🔍 DEBUGGING WALL CUTTING for {wall_name}")
        
        # 1. Check mesh properties
        print(f"  📊 Wall mesh info:")
        print(f"    - Vertices: {len(wall_mesh.vertices)}")
        print(f"    - Faces: {len(wall_mesh.faces)}")
        print(f"    - Is watertight: {getattr(wall_mesh, 'is_watertight', 'unknown')}")
        print(f"    - Bounds: {wall_mesh.bounds}")
        
        print(f"  📊 Cutter mesh info:")
        print(f"    - Vertices: {len(cutter_mesh.vertices)}")
        print(f"    - Faces: {len(cutter_mesh.faces)}")
        print(f"    - Is watertight: {getattr(cutter_mesh, 'is_watertight', 'unknown')}")
        print(f"    - Bounds: {cutter_mesh.bounds}")
        
        # 2. Check intersection
        try:
            # Check if meshes actually intersect
            intersection = wall_mesh.intersection(cutter_mesh)
            if intersection is not None and len(intersection.vertices) > 0:
                print(f"  ✅ Meshes DO intersect - intersection volume: {getattr(intersection, 'volume', 'unknown')}")
            else:
                print(f"  ❌ Meshes DO NOT intersect properly")
                return False
        except Exception as e:
            print(f"  ⚠️  Could not check intersection: {e}")
        
        # 3. Check if cutter is inside wall bounds
        wall_bounds = wall_mesh.bounds
        cutter_bounds = cutter_mesh.bounds
        
        print(f"  📏 Bounds comparison:")
        print(f"    Wall:   X({wall_bounds[0][0]:.3f}, {wall_bounds[1][0]:.3f}) Y({wall_bounds[0][1]:.3f}, {wall_bounds[1][1]:.3f}) Z({wall_bounds[0][2]:.3f}, {wall_bounds[1][2]:.3f})")
        print(f"    Cutter: X({cutter_bounds[0][0]:.3f}, {cutter_bounds[1][0]:.3f}) Y({cutter_bounds[0][1]:.3f}, {cutter_bounds[1][1]:.3f}) Z({cutter_bounds[0][2]:.3f}, {cutter_bounds[1][2]:.3f})")
        
        # Check overlap
        overlap_x = not (cutter_bounds[1][0] < wall_bounds[0][0] or cutter_bounds[0][0] > wall_bounds[1][0])
        overlap_y = not (cutter_bounds[1][1] < wall_bounds[0][1] or cutter_bounds[0][1] > wall_bounds[1][1])
        overlap_z = not (cutter_bounds[1][2] < wall_bounds[0][2] or cutter_bounds[0][2] > wall_bounds[1][2])
        
        print(f"    Overlap: X={overlap_x}, Y={overlap_y}, Z={overlap_z}")
        
        if not (overlap_x and overlap_y and overlap_z):
            print(f"  ❌ NO OVERLAP - This is why cutting fails!")
            return False
        
        return True


    def cut_wall_openings_enhanced(self, wall_mesh, wall_name, openings, room_width, room_length, room_height):
        """Enhanced wall cutting with better debugging"""
        if not openings:
            return wall_mesh
        
        print(f"🚪 Cutting {len(openings)} openings in {wall_name}...")
        
        current_wall = wall_mesh.copy()
        
        for i, opening in enumerate(openings):
            try:
                print(f"  🔧 Processing opening {i+1}/{len(openings)}: {opening['type']} '{opening['name']}'")
                
                # Create cutter mesh
                cutter_mesh = self.create_opening_cutter_enhanced(opening, wall_name, room_width, room_length)
                
                if cutter_mesh is None:
                    print(f"    ❌ Failed to create cutter for {opening['name']}")
                    continue
                
                # Debug intersection
                print(f"    🔍 Wall bounds: {current_wall.bounds}")
                print(f"    🔍 Cutter bounds: {cutter_mesh.bounds}")
                
                # Try boolean operation
                success = False
                
                # Method 1: Try manifold
                try:
                    print(f"    🎯 Trying manifold engine...")
                    result = current_wall.difference(cutter_mesh, engine='manifold')
                    if result is not None and len(result.vertices) > 0:
                        # Check if it actually changed
                        if len(result.vertices) != len(current_wall.vertices):
                            current_wall = result
                            success = True
                            print(f"    ✅ Manifold successful: {len(current_wall.vertices)} vertices")
                        else:
                            print(f"    ⚠️  Manifold returned same mesh")
                    else:
                        print(f"    ❌ Manifold returned empty result")
                except Exception as e:
                    print(f"    ❌ Manifold failed: {e}")
                
                # Method 2: Try blender if manifold failed
                if not success:
                    try:
                        print(f"    🎯 Trying blender engine...")
                        result = current_wall.difference(cutter_mesh, engine='blender')
                        if result is not None and len(result.vertices) > 0:
                            if len(result.vertices) != len(current_wall.vertices):
                                current_wall = result
                                success = True
                                print(f"    ✅ Blender successful: {len(current_wall.vertices)} vertices")
                            else:
                                print(f"    ⚠️  Blender returned same mesh")
                        else:
                            print(f"    ❌ Blender returned empty result")
                    except Exception as e:
                        print(f"    ❌ Blender failed: {e}")
                
                if success:
                    print(f"    ✅ Successfully cut {opening['type']} opening")
                else:
                    print(f"    ❌ Failed to cut {opening['type']} opening")
                    
            except Exception as e:
                print(f"    ❌ Error processing opening {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        return current_wall

    def create_multiple_cutters(self, opening, wall_name, room_width, room_length, room_height):
        """Create multiple cutter strategies with different sizes and positions"""
        wall_thickness = 0.1
        pos_x, pos_y = opening['position']
        opening_width = opening['width']
        opening_height = opening['height']
        opening_bottom = opening['bottom_height']
        
        wall_direction = wall_name.replace('_wall', '')
        cutters = []
        
        # Strategy 1: Exact size cutter
        cutter1 = self.create_exact_cutter(pos_x, pos_y, opening_width, opening_height, opening_bottom, 
                                        wall_direction, room_width, room_length, wall_thickness)
        cutters.append((cutter1, "Exact size"))
        
        # Strategy 2: Oversized cutter
        cutter2 = self.create_oversized_cutter(pos_x, pos_y, opening_width, opening_height, opening_bottom,
                                            wall_direction, room_width, room_length, wall_thickness)
        cutters.append((cutter2, "Oversized"))
        
        # Strategy 3: Extended through cutter
        cutter3 = self.create_extended_cutter(pos_x, pos_y, opening_width, opening_height, opening_bottom,
                                            wall_direction, room_width, room_length, wall_thickness, room_height)
        cutters.append((cutter3, "Extended through"))
        
        return cutters

    def create_exact_cutter(self, pos_x, pos_y, width, height, bottom, wall_direction, room_width, room_length, wall_thickness):
        """Create exact size cutter"""
        try:
            if wall_direction == 'front':
                pos = [pos_x, -wall_thickness/2, bottom + height/2]
                extents = [width, wall_thickness + 0.02, height]
            elif wall_direction == 'back':
                pos = [pos_x, room_length + wall_thickness/2, bottom + height/2]
                extents = [width, wall_thickness + 0.02, height]
            elif wall_direction == 'left':
                pos = [-wall_thickness/2, pos_y, bottom + height/2]
                extents = [wall_thickness + 0.02, width, height]
            elif wall_direction == 'right':
                pos = [room_width + wall_thickness/2, pos_y, bottom + height/2]
                extents = [wall_thickness + 0.02, width, height]
            else:
                return None
            
            return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(pos))
        except:
            return None

    def create_oversized_cutter(self, pos_x, pos_y, width, height, bottom, wall_direction, room_width, room_length, wall_thickness):
        """Create oversized cutter with margins"""
        try:
            margin = 0.1  # 10cm margin
            
            if wall_direction == 'front':
                pos = [pos_x, -wall_thickness/2, bottom + height/2]
                extents = [width + margin, wall_thickness + 0.1, height + margin]
            elif wall_direction == 'back':
                pos = [pos_x, room_length + wall_thickness/2, bottom + height/2]
                extents = [width + margin, wall_thickness + 0.1, height + margin]
            elif wall_direction == 'left':
                pos = [-wall_thickness/2, pos_y, bottom + height/2]
                extents = [wall_thickness + 0.1, width + margin, height + margin]
            elif wall_direction == 'right':
                pos = [room_width + wall_thickness/2, pos_y, bottom + height/2]
                extents = [wall_thickness + 0.1, width + margin, height + margin]
            else:
                return None
            
            return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(pos))
        except:
            return None

    def create_extended_cutter(self, pos_x, pos_y, width, height, bottom, wall_direction, room_width, room_length, wall_thickness, room_height):
        """Create cutter that extends through entire room"""
        try:
            if wall_direction == 'front':
                pos = [pos_x, room_length/2, bottom + height/2]
                extents = [width, room_length + wall_thickness * 2, height]
            elif wall_direction == 'back':
                pos = [pos_x, room_length/2, bottom + height/2]
                extents = [width, room_length + wall_thickness * 2, height]
            elif wall_direction == 'left':
                pos = [room_width/2, pos_y, bottom + height/2]
                extents = [room_width + wall_thickness * 2, width, height]
            elif wall_direction == 'right':
                pos = [room_width/2, pos_y, bottom + height/2]
                extents = [room_width + wall_thickness * 2, width, height]
            else:
                return None
            
            return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(pos))
        except:
            return None

    def try_boolean_methods(self, wall_mesh, cutter_mesh, strategy_name):
        """Try different boolean methods in order"""
        methods = [
            ('manifold', lambda w, c: w.difference(c, engine='manifold')),
            ('blender', lambda w, c: w.difference(c, engine='blender')),
            ('auto', lambda w, c: w.difference(c)),
            ('trimesh_boolean', self.trimesh_boolean_difference),
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"      🔄 Trying {method_name} method...")
                result = method_func(wall_mesh, cutter_mesh)
                
                if result is not None and len(result.vertices) > 0:
                    print(f"      ✅ {method_name} method successful")
                    return result
                else:
                    print(f"      ❌ {method_name} method returned empty result")
                    
            except Exception as e:
                print(f"      ❌ {method_name} method failed: {e}")
        
        return None

    def trimesh_boolean_difference(self, wall_mesh, cutter_mesh):
        """Use trimesh.boolean module directly"""
        import trimesh.boolean
        try:
            return trimesh.boolean.difference([wall_mesh, cutter_mesh], engine='auto')
        except:
            return trimesh.boolean.difference([wall_mesh, cutter_mesh])

    def mesh_actually_changed(self, original_mesh, new_mesh):
        """Check if the mesh actually changed after boolean operation"""
        try:
            # Compare vertex counts
            if len(original_mesh.vertices) == len(new_mesh.vertices):
                # If same number of vertices, check if positions are different
                if np.allclose(original_mesh.vertices, new_mesh.vertices, atol=1e-6):
                    return False
            
            # Compare volumes if available
            try:
                original_volume = getattr(original_mesh, 'volume', None)
                new_volume = getattr(new_mesh, 'volume', None)
                
                if original_volume is not None and new_volume is not None:
                    volume_diff = abs(original_volume - new_volume)
                    if volume_diff < 1e-6:  # Very small difference
                        return False
                    print(f"      📊 Volume changed: {original_volume:.6f} → {new_volume:.6f} (diff: {volume_diff:.6f})")
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"      ⚠️  Could not compare meshes: {e}")
            return True  # Assume it changed if we can't tell

    def manual_vertex_removal(self, wall_mesh, opening, wall_name, room_width, room_length):
        """Last resort: manually remove vertices in the opening area"""
        print(f"    🔧 Attempting manual vertex removal...")
        
        try:
            wall_direction = wall_name.replace('_wall', '')
            pos_x, pos_y = opening['position']
            width = opening['width']
            height = opening['height']
            bottom = opening['bottom_height']
            
            vertices = wall_mesh.vertices.copy()
            faces = wall_mesh.faces.copy()
            
            # Define removal bounds based on wall direction
            if wall_direction in ['front', 'back']:
                # Remove vertices in X-Z plane
                x_min, x_max = pos_x - width/2, pos_x + width/2
                z_min, z_max = bottom, bottom + height
                
                remove_mask = (
                    (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                    (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
                )
            else:  # left, right
                # Remove vertices in Y-Z plane
                y_min, y_max = pos_y - width/2, pos_y + width/2
                z_min, z_max = bottom, bottom + height
                
                remove_mask = (
                    (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
                    (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
                )
            
            # Keep vertices outside the opening area
            keep_mask = ~remove_mask
            removed_count = np.sum(remove_mask)
            
            if removed_count > 0:
                print(f"    📊 Removing {removed_count} vertices")
                
                # Keep only vertices outside opening
                new_vertices = vertices[keep_mask]
                
                # Create vertex mapping
                vertex_map = {}
                new_idx = 0
                for old_idx, keep in enumerate(keep_mask):
                    if keep:
                        vertex_map[old_idx] = new_idx
                        new_idx += 1
                
                # Filter faces
                new_faces = []
                for face in faces:
                    if all(idx in vertex_map for idx in face):
                        new_face = [vertex_map[idx] for idx in face]
                        new_faces.append(new_face)
                
                if len(new_vertices) > 0 and len(new_faces) > 0:
                    result_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                    print(f"    ✅ Manual removal successful: {len(new_vertices)} vertices, {len(new_faces)} faces")
                    return result_mesh
            
            print(f"    ⚠️  Manual removal had no effect")
            return wall_mesh
            
        except Exception as e:
            print(f"    ❌ Manual removal failed: {e}")
            return wall_mesh

    def create_wall_meshes_enhanced(self, width, length, height, wall_thickness):
        """Create wall meshes with door and window openings - ENHANCED VERSION"""
        walls = []
        
        # Wall positions and dimensions
        wall_configs = [
            ('back_wall', [width/2, length + wall_thickness/2, height/2], [width + 2*wall_thickness, wall_thickness, height]),
            ('front_wall', [width/2, -wall_thickness/2, height/2], [width + 2*wall_thickness, wall_thickness, height]),
            ('left_wall', [-wall_thickness/2, length/2, height/2], [wall_thickness, length, height]),
            ('right_wall', [width + wall_thickness/2, length/2, height/2], [wall_thickness, length, height])
        ]
        
        for wall_name, position, dimensions in wall_configs:
            try:
                print(f"🏗️  Creating {wall_name}...")
                
                # Create basic wall
                wall_mesh = trimesh.creation.box(
                    extents=dimensions,
                    transform=trimesh.transformations.translation_matrix(position)
                )
                
                # Check for openings on this wall
                openings = self.get_openings_for_wall(wall_name, width, length)
                
                if openings:
                    print(f"  🚪 Found {len(openings)} openings for {wall_name}")
                    # Cut openings
                    wall_mesh = self.cut_wall_openings_enhanced(wall_mesh, wall_name, openings, width, length, height)
                else:
                    print(f"  ℹ️  No openings for {wall_name}")
                
                walls.append((wall_mesh, wall_name))
                print(f"  ✅ {wall_name} created successfully")
                    
            except Exception as e:
                print(f"  ❌ Error creating {wall_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Create simple wall as fallback
                simple_wall = trimesh.creation.box(
                    extents=dimensions,
                    transform=trimesh.transformations.translation_matrix(position)
                )
                walls.append((simple_wall, wall_name))
        
        return walls


    def determine_wall_for_opening(self, pos_x, pos_y, room_width, room_length):
        """Determine which wall an opening belongs to based on its position - IMPROVED"""
        tolerance = 0.3  # Increased tolerance
        
        print(f"    🧮 Determining wall for position ({pos_x:.2f}, {pos_y:.2f})")
        print(f"    🏠 Room dimensions: {room_width}m x {room_length}m")
        
        # Calculate distances to each wall
        distances = {
            'front': abs(pos_y - 0),           # Distance to front wall (y=0)
            'back': abs(pos_y - room_length),  # Distance to back wall (y=room_length)
            'left': abs(pos_x - 0),            # Distance to left wall (x=0)
            'right': abs(pos_x - room_width)   # Distance to right wall (x=room_width)
        }
        
        print(f"    📏 Distances to walls:")
        for wall, dist in distances.items():
            print(f"      {wall}: {dist:.3f}m")
        
        # Find the closest wall
        closest_wall = min(distances, key=distances.get)
        closest_distance = distances[closest_wall]
        
        print(f"    🎯 Closest wall: {closest_wall} (distance: {closest_distance:.3f}m)")
        
        # Check if it's within tolerance
        if closest_distance <= tolerance:
            print(f"    ✅ Within tolerance ({tolerance}m), assigning to {closest_wall} wall")
            return closest_wall
        else:
            print(f"    ⚠️  Distance {closest_distance:.3f}m > tolerance {tolerance}m, but assigning to {closest_wall} anyway")
            return closest_wall


    def get_openings_for_wall(self, wall_name, room_width, room_length):
        """Get door and window openings for a specific wall - IMPROVED VERSION"""
        openings = []
        wall_direction = wall_name.replace('_wall', '')
        
        print(f"🔍 Checking for openings on {wall_name} (direction: {wall_direction})")
        
        doors_windows_found = 0
        
        for obj_name, obj_data in self.layout_data['objects'].items():
            # Check if this is a door or window
            if any(keyword in obj_name.lower() for keyword in ['door', 'window']):
                doors_windows_found += 1
                opening_type = 'door' if 'door' in obj_name.lower() else 'window'
                pos_x, pos_y, rotation = obj_data['position']
                width = obj_data['width']
                
                print(f"  🔍 Checking {opening_type} '{obj_name}' at ({pos_x:.2f}, {pos_y:.2f}) for {wall_direction} wall")
                
                wall_for_opening = self.determine_wall_for_opening(pos_x, pos_y, room_width, room_length)
                
                print(f"    📍 Determined wall: {wall_for_opening}")
                
                if wall_for_opening == wall_direction:
                    opening_data = {
                        'type': opening_type,
                        'position': (pos_x, pos_y),
                        'width': width,
                        'height': 2.1 if opening_type == 'door' else 1.2,
                        'bottom_height': 0 if opening_type == 'door' else 1.0,
                        'name': obj_name
                    }
                    openings.append(opening_data)
                    print(f"    ✅ Added {opening_type} '{obj_name}' to {wall_direction} wall")
                else:
                    print(f"    ⏭️  {opening_type} '{obj_name}' belongs to {wall_for_opening} wall, not {wall_direction}")
        
        print(f"  📊 Found {doors_windows_found} total doors/windows, {len(openings)} for {wall_name}")
        return openings


    def determine_wall_for_opening(self, pos_x, pos_y, room_width, room_length):
        """Determine which wall an opening belongs to"""
        tolerance = 0.2
        
        if abs(pos_y) < tolerance:
            return 'front'
        elif abs(pos_y - room_length) < tolerance:
            return 'back'
        elif abs(pos_x) < tolerance:
            return 'left'
        elif abs(pos_x - room_width) < tolerance:
            return 'right'
        else:
            distances = {
                'front': pos_y,
                'back': room_length - pos_y,
                'left': pos_x,
                'right': room_width - pos_x
            }
            return min(distances, key=distances.get)


    def cut_wall_openings_enhanced(self, wall_mesh, wall_name, openings, room_width, room_length, room_height):
        """Enhanced wall cutting with better debugging"""
        if not openings:
            return wall_mesh
        
        print(f"🚪 Cutting {len(openings)} openings in {wall_name}...")
        
        current_wall = wall_mesh.copy()
        
        for i, opening in enumerate(openings):
            try:
                print(f"  🔧 Processing opening {i+1}/{len(openings)}: {opening['type']} '{opening['name']}'")
                
                # Create cutter mesh
                cutter_mesh = self.create_opening_cutter_enhanced(opening, wall_name, room_width, room_length)
                
                if cutter_mesh is None:
                    print(f"    ❌ Failed to create cutter for {opening['name']}")
                    continue
                
                # Debug intersection
                print(f"    🔍 Wall bounds: {current_wall.bounds}")
                print(f"    🔍 Cutter bounds: {cutter_mesh.bounds}")
                
                # Try boolean operation
                success = False
                
                # Method 1: Try manifold
                try:
                    print(f"    🎯 Trying manifold engine...")
                    result = current_wall.difference(cutter_mesh, engine='manifold')
                    if result is not None and len(result.vertices) > 0:
                        # Check if it actually changed
                        if len(result.vertices) != len(current_wall.vertices):
                            current_wall = result
                            success = True
                            print(f"    ✅ Manifold successful: {len(current_wall.vertices)} vertices")
                        else:
                            print(f"    ⚠️  Manifold returned same mesh")
                    else:
                        print(f"    ❌ Manifold returned empty result")
                except Exception as e:
                    print(f"    ❌ Manifold failed: {e}")
                
                # Method 2: Try blender if manifold failed
                if not success:
                    try:
                        print(f"    🎯 Trying blender engine...")
                        result = current_wall.difference(cutter_mesh, engine='blender')
                        if result is not None and len(result.vertices) > 0:
                            if len(result.vertices) != len(current_wall.vertices):
                                current_wall = result
                                success = True
                                print(f"    ✅ Blender successful: {len(current_wall.vertices)} vertices")
                            else:
                                print(f"    ⚠️  Blender returned same mesh")
                        else:
                            print(f"    ❌ Blender returned empty result")
                    except Exception as e:
                        print(f"    ❌ Blender failed: {e}")
                
                if success:
                    print(f"    ✅ Successfully cut {opening['type']} opening")
                else:
                    print(f"    ❌ Failed to cut {opening['type']} opening")
                    
            except Exception as e:
                print(f"    ❌ Error processing opening {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        return current_wall

    def create_opening_cutter_enhanced(self, opening, wall_name, room_width, room_length):
        """Enhanced opening cutter creation"""
        wall_thickness = 0.1
        pos_x, pos_y = opening['position']
        opening_width = opening['width']
        opening_height = opening['height']
        opening_bottom = opening['bottom_height']
        
        wall_direction = wall_name.replace('_wall', '')
        
        # Add extra margin for reliable cutting
        margin = 0.05
        extended_thickness = wall_thickness + 2 * margin
        
        print(f"    📦 Creating cutter for {opening['type']} on {wall_direction} wall")
        print(f"    📍 Position: ({pos_x:.2f}, {pos_y:.2f})")
        print(f"    📏 Size: {opening_width}m x {opening_height}m")
        
        # Determine cutter position based on wall direction
        if wall_direction == 'front':
            cutter_pos = [pos_x, -wall_thickness/2, opening_bottom + opening_height/2]
            cutter_extents = [opening_width + margin, extended_thickness, opening_height + margin]
        elif wall_direction == 'back':
            cutter_pos = [pos_x, room_length + wall_thickness/2, opening_bottom + opening_height/2]
            cutter_extents = [opening_width + margin, extended_thickness, opening_height + margin]
        elif wall_direction == 'left':
            cutter_pos = [-wall_thickness/2, pos_y, opening_bottom + opening_height/2]
            cutter_extents = [extended_thickness, opening_width + margin, opening_height + margin]
        elif wall_direction == 'right':
            cutter_pos = [room_width + wall_thickness/2, pos_y, opening_bottom + opening_height/2]
            cutter_extents = [extended_thickness, opening_width + margin, opening_height + margin]
        else:
            print(f"    ❌ Unknown wall direction: {wall_direction}")
            return None
        
        try:
            cutter_mesh = trimesh.creation.box(
                extents=cutter_extents,
                transform=trimesh.transformations.translation_matrix(cutter_pos)
            )
            
            print(f"    📦 Cutter created: pos={cutter_pos}, extents={cutter_extents}")
            return cutter_mesh
            
        except Exception as e:
            print(f"    ❌ Failed to create cutter: {e}")
            return None
    
    def get_object_ids_from_clip(self, object_name):
            """Get object IDs from CLIP rerank results with flexible matching"""
            # Remove numbers from object name for base name
            import re
            base_name = re.sub(r'\d+$', '', object_name).strip()
            
            print(f"🔍 Searching CLIP results for '{object_name}' (base: '{base_name}')")
            
            # First, try exact match
            exact_file = os.path.join(self.clip_results_path, f"{object_name.lower()}.txt")
            if os.path.exists(exact_file):
                try:
                    with open(exact_file, 'r') as f:
                        ids = [line.strip() for line in f.readlines() if line.strip()]
                        print(f"📎 Found exact match: {len(ids)} IDs for {object_name}")
                        return ids
                except Exception as e:
                    print(f"❌ Error reading exact match: {e}")
            
            # Second, search for files containing the base name
            try:
                clip_files = [f for f in os.listdir(self.clip_results_path) if f.endswith('.txt')]
                
                # Search for files that contain the base name (case-insensitive)
                matching_files = []
                for clip_file in clip_files:
                    # Extract the object name from the file
                    file_obj_name = clip_file.replace('.txt', '')
                    
                    # Check if base name is in the file name (as a word)
                    if base_name.lower() in file_obj_name.lower().split():
                        matching_files.append((clip_file, file_obj_name))
                        print(f"  📄 Found candidate: {clip_file}")
                
                # If we found matches, use the first one (or the best match)
                if matching_files:
                    # Sort by similarity (prefer exact word match over partial)
                    best_match = None
                    for clip_file, file_obj_name in matching_files:
                        # Exact word match is best
                        if base_name.lower() == file_obj_name.lower():
                            best_match = clip_file
                            break
                        # Otherwise, take the first match
                        elif best_match is None:
                            best_match = clip_file
                    
                    if best_match:
                        file_path = os.path.join(self.clip_results_path, best_match)
                        with open(file_path, 'r') as f:
                            ids = [line.strip() for line in f.readlines() if line.strip()]
                            print(f"📎 Using {best_match}: {len(ids)} IDs for {object_name}")
                            return ids
                
                # Third fallback: try variations of the name
                name_variations = [
                    base_name.lower().replace(' ', '_'),
                    base_name.lower().replace(' ', '-'),
                    base_name.lower().replace('_', ' '),
                    base_name.lower()
                ]
                
                for variation in name_variations:
                    for clip_file in clip_files:
                        file_obj_name = clip_file.replace('.txt', '').lower()
                        # Check if variation matches exactly
                        if variation == file_obj_name:
                            file_path = os.path.join(self.clip_results_path, clip_file)
                            with open(file_path, 'r') as f:
                                ids = [line.strip() for line in f.readlines() if line.strip()]
                                print(f"📎 Found variation match {clip_file}: {len(ids)} IDs for {object_name}")
                                return ids
                
            except Exception as e:
                print(f"❌ Error searching CLIP results: {e}")
                import traceback
                traceback.print_exc()
            
            # Debug: show available files
            try:
                available_files = [f for f in os.listdir(self.clip_results_path) if f.endswith('.txt')]
                if available_files:
                    print(f"  📂 Available CLIP files in {self.clip_results_path}:")
                    for f in available_files[:5]:  # Show first 5 files
                        print(f"     - {f}")
                    if len(available_files) > 5:
                        print(f"     ... and {len(available_files) - 5} more files")
            except:
                pass
            
            return []
    
    def find_object_in_dataset(self, object_id):
        """Find object folder in 3D-FUTURE dataset"""
        for dataset_path in self.dataset_paths:
            obj_folder = os.path.join(dataset_path, object_id)
            if os.path.exists(obj_folder):
                return obj_folder
        return None
    
    def load_furniture_object(self, object_name, obj_data):
        """Load furniture object from 3D-FUTURE dataset with texture support"""
        # Get CLIP results
        clip_ids = self.get_object_ids_from_clip(object_name)
        
        if not clip_ids:
            print(f"❌ No CLIP results for {object_name}, creating placeholder")
            return self.create_placeholder_mesh(object_name, obj_data)
        
        # Try to load from first few IDs
        for i, clip_id in enumerate(clip_ids[:3]):
            obj_folder = self.find_object_in_dataset(clip_id)
            if obj_folder:
                obj_path = os.path.join(obj_folder, "normalized_model.obj")
                if os.path.exists(obj_path):
                    try:
                        print(f"📦 Loading {object_name} from {clip_id}")
                        
                        # Try trimesh first for better texture support
                        try:
                            mesh = trimesh.load(obj_path)
                            
                            # Scene 객체인 경우 mesh 추출
                            if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                                mesh = list(mesh.geometry.values())[0]
                            
                            if len(mesh.vertices) == 0:
                                raise Exception("Empty mesh from trimesh")
                                
                            print(f"  ✅ Loaded with trimesh: {len(mesh.vertices)} vertices")
                            
                            # 텍스처 적용
                            self.apply_texture_to_mesh(mesh, obj_folder)
                            
                        except Exception as trimesh_error:
                            print(f"  ⚠️  Trimesh loading failed: {trimesh_error}")
                            print(f"  🔄 Falling back to Open3D...")
                            
                            # Fallback to Open3D (기존 로직)
                            import open3d as o3d
                            o3d_mesh = o3d.io.read_triangle_mesh(obj_path)
                            
                            if len(o3d_mesh.vertices) == 0:
                                print(f"❌ Empty mesh for {clip_id}")
                                continue
                            
                            # 메시 데이터 검증 및 정리 (기존 로직 유지)
                            vertices_raw = np.asarray(o3d_mesh.vertices)
                            faces_raw = np.asarray(o3d_mesh.triangles)
                            
                            print(f"  📊 Raw data info - Vertices: {vertices_raw.shape}, Faces: {faces_raw.shape}")
                            
                            # Vertices 안전하게 처리
                            if len(vertices_raw) == 0:
                                print(f"❌ No vertices in mesh for {clip_id}")
                                continue
                                
                            if vertices_raw.shape[1] != 3:
                                print(f"❌ Invalid vertex array shape for {clip_id}: {vertices_raw.shape}")
                                continue
                            
                            # 깨끗한 vertices 배열 생성
                            vertices = np.array(vertices_raw, dtype=np.float64, copy=True)
                            
                            print(f"  📊 Raw faces info - Type: {type(faces_raw)}, Shape: {faces_raw.shape}, Dtype: {faces_raw.dtype}")
                            
                            # Face 배열 검증
                            if len(faces_raw) == 0:
                                print(f"❌ No faces in mesh for {clip_id}")
                                continue
                            
                            # Face 배열을 더 안전하게 처리 (기존 로직 유지)
                            try:
                                faces_list = []
                                max_vertex_idx = len(vertices) - 1
                                
                                for i, face_raw in enumerate(faces_raw):
                                    try:
                                        face = np.asarray(face_raw, dtype=np.int32).flatten()
                                        
                                        if len(face) == 3:
                                            if np.all(face >= 0) and np.all(face <= max_vertex_idx):
                                                faces_list.append([int(face[0]), int(face[1]), int(face[2])])
                                            else:
                                                print(f"  ⚠️  Invalid face indices at {i}: {face}")
                                        else:
                                            print(f"  ⚠️  Face {i} has {len(face)} vertices instead of 3: {face}")
                                            
                                    except Exception as face_error:
                                        print(f"  ⚠️  Error processing face {i}: {face_error}")
                                        continue
                                
                                if not faces_list:
                                    print(f"❌ No valid faces after processing for {clip_id}")
                                    continue
                                
                                valid_faces = np.array(faces_list, dtype=np.int32)
                                print(f"  ✅ Processed {len(valid_faces)} valid faces out of {len(faces_raw)}")
                                
                            except Exception as processing_error:
                                print(f"❌ Error processing faces for {clip_id}: {processing_error}")
                                continue
                            
                            print(f"  📊 Mesh stats: {len(vertices)} vertices, {len(valid_faces)} faces")
                            
                            # trimesh 객체 생성 (기존 로직 유지)
                            try:
                                mesh = trimesh.Trimesh(vertices=vertices, faces=valid_faces)
                                print(f"  ✅ Basic trimesh created successfully")
                            except Exception as basic_error:
                                print(f"  ❌ Basic trimesh creation failed: {basic_error}")
                                try:
                                    vertices_copy = np.array(vertices, dtype=np.float64, copy=True)
                                    faces_copy = np.array(valid_faces, dtype=np.int32, copy=True)
                                    mesh = trimesh.Trimesh(vertices=vertices_copy, faces=faces_copy)
                                    print(f"  ✅ Trimesh created with copied data")
                                except Exception as copy_error:
                                    print(f"  ❌ Trimesh creation with copied data failed: {copy_error}")
                                    continue
                            
                            # 메시 유효성 검사 (기존 로직 유지)
                            try:
                                is_valid = mesh.is_valid if hasattr(mesh, 'is_valid') else True
                                if not is_valid:
                                    print(f"⚠️  Invalid mesh for {clip_id}, attempting to fix...")
                                    if hasattr(mesh, 'fix_normals'):
                                        mesh.fix_normals()
                                    if hasattr(mesh, 'remove_duplicate_faces'):
                                        mesh.remove_duplicate_faces()
                                    if hasattr(mesh, 'remove_degenerate_faces'):
                                        mesh.remove_degenerate_faces()
                            except Exception as validation_error:
                                print(f"⚠️  Mesh validation warning for {clip_id}: {validation_error}")                            
                            # 기본 색상 적용 (Open3D 경로에서는 텍스처 로드 불가)
                            self.apply_basic_material_color(mesh, obj_folder)
                        
                        # Apply transformations (원본 로직 그대로 사용)
                        transformed_mesh = self.transform_furniture_mesh(mesh, obj_data)
                        
                        print(f"  ✅ Successfully loaded and transformed {object_name}")
                        return transformed_mesh
                                
                    except Exception as e:
                        print(f"❌ Failed to load {clip_id}: {e}")
                        continue
        
        print(f"❌ Could not load {object_name}, creating placeholder")
        return self.create_placeholder_mesh(object_name, obj_data)


    def apply_texture_to_mesh(self, mesh, obj_folder):
        """Apply texture to mesh from object folder"""
        try:
            texture_path = os.path.join(obj_folder, "texture.png")
            mtl_path = os.path.join(obj_folder, "model.mtl")
            
            if os.path.exists(texture_path):
                print(f"  🎨 Found texture file: {texture_path}")
                
                try:
                    from PIL import Image
                    texture_image = Image.open(texture_path)
                    print(f"  📏 Texture size: {texture_image.size}")
                    
                    # UV 좌표가 있는지 확인
                    has_uv = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
                    print(f"  🗺️  Has UV coordinates: {has_uv}")
                    
                    if has_uv:
                        try:
                            # UV가 있으면 TextureVisuals 사용
                            texture_visual = trimesh.visual.TextureVisuals(
                                uv=mesh.visual.uv,
                                image=texture_image
                            )
                            mesh.visual = texture_visual
                            print(f"  ✅ Applied texture with UV mapping")
                        except Exception as uv_error:
                            print(f"  ⚠️  UV texture application failed: {uv_error}")
                            self.apply_basic_material_color(mesh, obj_folder)
                    else:
                        print(f"  ⚠️  No UV coordinates, using material color")
                        self.apply_basic_material_color(mesh, obj_folder)
                        
                except Exception as tex_error:
                    print(f"  ❌ Texture loading failed: {tex_error}")
                    self.apply_basic_material_color(mesh, obj_folder)
                    
            else:
                print(f"  📄 No texture file, checking material file")
                self.apply_basic_material_color(mesh, obj_folder)
                
        except Exception as e:
            print(f"  ❌ Texture application error: {e}")
            try:
                mesh.visual.face_colors = [180, 180, 180, 255]
            except:
                pass


    def apply_basic_material_color(self, mesh, obj_folder):
        """Apply color from MTL file or use default"""
        try:
            mtl_path = os.path.join(obj_folder, "model.mtl")
            
            if os.path.exists(mtl_path):
                print(f"  📄 Processing MTL file: {mtl_path}")
                
                with open(mtl_path, 'r') as f:
                    mtl_content = f.read()
                
                # Kd (diffuse color) 추출
                import re
                kd_match = re.search(r'Kd\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', mtl_content)
                if kd_match:
                    r, g, b = map(float, kd_match.groups())
                    color = [int(r * 255), int(g * 255), int(b * 255), 255]
                    mesh.visual.face_colors = color
                    print(f"  🎨 Applied MTL diffuse color: RGB({r:.2f}, {g:.2f}, {b:.2f})")
                    return
                
                # Ka (ambient color) 시도
                ka_match = re.search(r'Ka\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', mtl_content)
                if ka_match:
                    r, g, b = map(float, ka_match.groups())
                    color = [int(r * 255), int(g * 255), int(b * 255), 255]
                    mesh.visual.face_colors = color
                    print(f"  🎨 Applied MTL ambient color: RGB({r:.2f}, {g:.2f}, {b:.2f})")
                    return
                    
            # 기본 색상 적용 (가구별 다른 색상)
            default_colors = {
                'bed': [160, 140, 120, 255],      # 베드: 갈색
                'wardrobe': [140, 120, 100, 255], # 옷장: 진한 갈색
                'desk': [180, 160, 140, 255],     # 책상: 밝은 갈색
                'chair': [160, 140, 120, 255],    # 의자: 갈색
                'sofa': [120, 100, 80, 255],      # 소파: 어두운 갈색
                'table': [180, 160, 140, 255],    # 테이블: 밝은 갈색
            }
            
            # 객체 이름에서 가구 타입 추정
            mesh_color = [160, 140, 120, 255]  # 기본 나무색
            
            mesh.visual.face_colors = mesh_color
            print(f"  🎨 Applied default wood color")
            
        except Exception as e:
            print(f"  ⚠️  Material color application failed: {e}")
            try:
                mesh.visual.face_colors = [180, 180, 180, 255]  # 회색
            except:
                pass


    def transform_furniture_mesh(self, mesh, obj_data):
        """Transform furniture mesh to correct scale and position using Blender-style transformations"""
        try:
            # 텍스처 정보 백업
            original_visual = None
            if hasattr(mesh, 'visual'):
                try:
                    import copy
                    original_visual = copy.deepcopy(mesh.visual)
                    print(f"    💾 Backed up visual information")
                except Exception as backup_error:
                    print(f"    ⚠️  Visual backup failed: {backup_error}")
            
            # === 기존 transformation 로직 그대로 유지 ===
            
            # Step 1: Initial mesh orientation (Blender의 90도 X축 회전 적용)
            print(f"    🔄 Applying initial 90-degree X rotation...")
            try:
                # 90도 X축 회전 행렬 생성
                initial_rotation_matrix = np.eye(4)
                initial_rotation_matrix[1, 1] = 0  # cos(90°) = 0
                initial_rotation_matrix[1, 2] = -1  # -sin(90°) = -1
                initial_rotation_matrix[2, 1] = 1   # sin(90°) = 1
                initial_rotation_matrix[2, 2] = 0   # cos(90°) = 0
                
                mesh.apply_transform(initial_rotation_matrix)
                print(f"    ✅ Initial rotation applied")
            except Exception as rotation_error:
                print(f"    ❌ Initial rotation failed: {rotation_error}")
            
            # Step 2: Calculate scaling (Blender와 동일한 로직)
            bounds = mesh.bounds
            current_size = bounds[1] - bounds[0]
            current_width = current_size[0]
            current_length = current_size[1] 
            current_height = current_size[2]
            
            # Target dimensions
            target_width = obj_data['width']
            target_length = obj_data['length']
            
            # Calculate scale factors
            scale_x = target_width / current_width if current_width > 0 else 1
            scale_y = target_length / current_length if current_length > 0 else 1
            scale_z = max(scale_x, scale_y)  # Keep proportional height (Blender와 동일)
            
            print(f"    🔧 Scaling: {scale_x:.3f}, {scale_y:.3f}, {scale_z:.3f}")
            
            # Apply scaling
            try:
                scale_matrix = np.eye(4)
                scale_matrix[0, 0] = scale_x
                scale_matrix[1, 1] = scale_y  
                scale_matrix[2, 2] = scale_z
                mesh.apply_transform(scale_matrix)
                print(f"    ✅ Scaling applied")
            except Exception as scale_error:
                print(f"    ❌ Scaling failed: {scale_error}")
            
            # Step 3: Apply rotation - 가구가 벽을 바라보는 문제 수정
            rotation_z_original = obj_data['position'][2]

            # 방법 1: 180도 추가 회전 시도
            final_z_rot = rotation_z_original + math.pi  # 180도 추가

            # 또는 방법 2: 반대 방향으로 회전
            # final_z_rot = -rotation_z_original

            # 또는 방법 3: 원래 각도에서 180도 뺄셈
            # final_z_rot = rotation_z_original - math.pi

            print(f"    🔄 Final Z rotation: {final_z_rot:.3f} radians (original: {rotation_z_original:.3f} + π)")

            if abs(final_z_rot) > 0.001:
                cos_r = np.cos(final_z_rot)
                sin_r = np.sin(final_z_rot)
                rotation_matrix = np.array([
                    [cos_r, -sin_r, 0, 0],
                    [sin_r,  cos_r, 0, 0],
                    [0,      0,     1, 0],
                    [0,      0,     0, 1]
                ])
                mesh.apply_transform(rotation_matrix)
                print(f"    ✅ Rotation applied with 180° correction")
            else:
                print(f"    ⏭️  Skipping minimal rotation")

            
            # Step 4: Position at target location (Blender 방식 적용)
            try:
                # Blender 방식: 바운딩 박스 계산 후 단계별 위치 조정
                bounds = mesh.bounds
                
                # 바운딩 박스의 중심과 최소값 계산
                min_x, min_y, min_z = bounds[0]
                max_x, max_y, max_z = bounds[1]
                
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                print(f"    📏 Bounding box - Center: ({center_x:.3f}, {center_y:.3f}), Min Z: {min_z:.3f}")
                
                # Step 4a: 객체 원점을 바닥 중심으로 이동 (Blender 방식)
                # 중심을 원점으로, Z 최소값을 0으로
                origin_adjustment = np.array([-center_x, -center_y, -min_z])
                
                origin_matrix = np.eye(4)
                origin_matrix[0:3, 3] = origin_adjustment
                mesh.apply_transform(origin_matrix)
                
                print(f"    🎯 Origin adjustment: {origin_adjustment}")
                
                # Step 4b: 타겟 위치로 이동 (Blender 방식)
                target_position = np.array([
                    obj_data['position'][0], 
                    obj_data['position'][1], 
                    0  # 바닥에 위치
                ])
                
                target_matrix = np.eye(4)
                target_matrix[0:3, 3] = target_position
                mesh.apply_transform(target_matrix)
                
                print(f"    📍 Target position: {target_position}")
                print(f"    ✅ Positioning completed (Blender style)")
                    
            except Exception as translation_error:
                print(f"    ❌ Positioning failed: {translation_error}")
            
            # Step 5: 최종 바운딩 박스 확인 (디버깅용)
            try:
                final_bounds = mesh.bounds
                final_center = (final_bounds[0] + final_bounds[1]) / 2
                print(f"    📊 Final bounds center: ({final_center[0]:.3f}, {final_center[1]:.3f}, {final_center[2]:.3f})")
                print(f"    📊 Final bounds min Z: {final_bounds[0][2]:.3f}")
            except Exception as bounds_error:
                print(f"    ⚠️  Could not compute final bounds: {bounds_error}")
            
            # === 기존 transformation 로직 끝 ===
            
            # 텍스처 정보 복원 시도
            if original_visual is not None:
                try:
                    mesh.visual = original_visual
                    print(f"    ✅ Restored visual information")
                except Exception as restore_error:
                    print(f"    ⚠️  Visual restoration failed: {restore_error}")
                    # 기본 색상으로 대체
                    try:
                        mesh.visual.face_colors = [160, 140, 120, 255]
                    except:
                        pass
            
            return mesh
            
        except Exception as e:
            print(f"    ❌ Transform failed completely: {e}")
            # 변환이 완전히 실패하면 원본 메시 반환
            return mesh
    
    def create_placeholder_mesh(self, object_name, obj_data):
        """Create placeholder mesh for objects that can't be loaded"""
        position = obj_data['position']
        width = obj_data['width']
        length = obj_data['length']
        
        # Object height estimates
        height_map = {
            'bed': 0.6,
            'wardrobe': 2.0,
            'desk': 0.75,
        }
        height = height_map.get(object_name.lower(), 1.0)
        
        # Create placeholder box
        mesh = trimesh.creation.box(
            extents=[width, length, height],
            transform=trimesh.transformations.translation_matrix([
                position[0], position[1], height/2
            ])
        )
        
        print(f"📦 Created placeholder for {object_name}")
        return mesh
    
    def load_all_furniture(self):
        """Load all furniture objects from layout"""
        furniture_objects = [name for name in self.layout_data['objects'].keys() 
                           if 'door' not in name.lower() and 'window' not in name.lower()]
        
        print(f"\n🪑 Loading furniture: {furniture_objects}")
        
        for obj_name in furniture_objects:
            if obj_name in self.layout_data['objects']:
                try:
                    mesh = self.load_furniture_object(obj_name, self.layout_data['objects'][obj_name])                    if mesh is not None:
                        self.scene_meshes.append({
                            'mesh': mesh,
                            'name': f"{obj_name}_furniture",
                            'color': (0.7, 0.7, 0.7)  # Default furniture color
                        })
                except Exception as e:
                    print(f"❌ Error loading {obj_name}: {e}")
    
    def combine_scene_meshes(self):
        """Combine all scene meshes into a single mesh"""
        if not self.scene_meshes:
            print("❌ No meshes to combine")
            return None
        
        print(f"🔗 Combining {len(self.scene_meshes)} meshes...")
        
        combined_meshes = []
        mesh_info = []
        
        for scene_item in self.scene_meshes:
            mesh = scene_item['mesh']
            color = scene_item['color']
            name = scene_item['name']
            
            # Validate mesh before adding
            if mesh is None or len(mesh.vertices) == 0:
                print(f"⚠️  Skipping empty mesh: {name}")
                continue
            
            # Apply color to mesh
            if hasattr(mesh, 'visual'):
                # Convert color to RGBA (0-255)
                rgba_color = [int(c * 255) for c in color] + [255]
                mesh.visual.face_colors = rgba_color
            
            combined_meshes.append(mesh)
            mesh_info.append(f"{name}: {len(mesh.vertices)} vertices")
        
        if not combined_meshes:
            print("❌ No valid meshes to combine")
            return None
        
        # Combine all meshes
        try:
            combined_mesh = trimesh.util.concatenate(combined_meshes)
            print("✅ Successfully combined all meshes:")
            for info in mesh_info:
                print(f"   - {info}")
            print(f"   Total: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")
            return combined_mesh
        except Exception as e:
            print(f"❌ Error combining meshes: {e}")
            return None
    
    def save_scene_as_glb(self, output_filename="scene.glb"):
        """Save the complete scene as GLB file"""
        combined_mesh = self.combine_scene_meshes()
        
        if combined_mesh is None:
            print("❌ No combined mesh to save")
            return False
        
        output_path = os.path.join(self.output_path, output_filename)
        
        try:
            # Ensure the mesh is valid before export
            # if not combined_mesh.is_valid:
            #     print("⚠️  Fixing mesh before export...")
            #     combined_mesh.fix_normals()
            #     combined_mesh.remove_duplicate_faces()
            #     combined_mesh.remove_degenerate_faces()
            
            # Export as GLB
            combined_mesh.export(output_path)
            
            # Verify file was created and get size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"💾 Scene saved as GLB: {output_path}")
                print(f"   File size: {file_size:.2f} MB")
                
                # Also save summary
                self.save_scene_summary()
                return True
            else:
                print(f"❌ Failed to create output file: {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving GLB: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_scene_summary(self):
        """Save scene composition summary as JSON"""
        summary = {
            'room_dimensions': {
                'width': self.layout_data['room_width'],
                'length': self.layout_data['room_length'],
                'height': 4.0
            },
            'colors': self.background_colors,
            'objects': {},
            'statistics': {
                'total_meshes': len(self.scene_meshes),
                'furniture_count': len([m for m in self.scene_meshes if 'furniture' in m['name']]),
                'room_elements': len([m for m in self.scene_meshes if any(x in m['name'] for x in ['wall', 'floor'])])
            }
        }
        
        # Add object details
        for name, data in self.layout_data['objects'].items():
            summary['objects'][name] = {
                'position': data['position'],
                'dimensions': {
                    'width': data['width'],
                    'length': data['length']
                },
                'type': 'door/window' if any(x in name.lower() for x in ['door', 'window']) else 'furniture'
            }
        
        summary_path = os.path.join(self.output_path, "scene_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Scene summary saved: {summary_path}")
    
    def compose_scene(self):
        """Main function to compose the entire scene"""
        print("=" * 50)
        print("🎬 SCENE COMPOSITION START")
        print("=" * 50)
        
        # Print layout info with colors
        print(f"\n🏠 Room Layout:")
        print(f"   Size: {self.layout_data['room_width']}m x {self.layout_data['room_length']}m")
        print(f"   Floor color: {self.background_colors['floor_color']} (RGB: {[int(c*255) for c in self.background_colors['floor_color']]})")
        print(f"   Wall color: {self.background_colors['wall_color']} (RGB: {[int(c*255) for c in self.background_colors['wall_color']]})")
        
        print(f"\n📦 Objects in layout:")
        for name, data in self.layout_data['objects'].items():
            obj_type = "🚪 Door/Window" if ('door' in name.lower() or 'window' in name.lower()) else "🪑 Furniture"
            print(f"   {obj_type} {name}: pos={data['position']}, size={data['width']}x{data['length']}")
        
        # Create room
        self.create_room_mesh()
        
        # Load furniture
        self.load_all_furniture()
        
        # Save scene
        success = self.save_scene_as_glb()
        
        # Final summary
        print("\n" + "=" * 50)
        if success:
            print("✅ SCENE COMPOSITION COMPLETE")
        else:
            print("❌ SCENE COMPOSITION FAILED")
        print("=" * 50)
        
        print(f"📊 Final scene contains {len(self.scene_meshes)} objects:")
        for item in self.scene_meshes:
            icon = "🏠" if "wall" in item['name'] or "floor" in item['name'] else "🪑"
            color_info = ""
            if 'floor' in item['name']:
                color_info = f" (Color: {self.background_colors['floor_color']})"
            elif 'wall' in item['name']:
                color_info = f" (Color: {self.background_colors['wall_color']})"
            print(f"   {icon} {item['name']}{color_info}")


    # combine_scene_meshes 함수 수정 (색상 적용 개선)
    def combine_scene_meshes(self):
        """Combine all scene meshes into a single mesh with proper colors"""
        if not self.scene_meshes:
            print("❌ No meshes to combine")
            return None
        
        print(f"🔗 Combining {len(self.scene_meshes)} meshes with colors...")
        
        combined_meshes = []
        mesh_info = []
        
        for scene_item in self.scene_meshes:
            mesh = scene_item['mesh']
            color = scene_item['color']
            name = scene_item['name']
            
            # Validate mesh before adding
            if mesh is None or len(mesh.vertices) == 0:
                print(f"⚠️  Skipping empty mesh: {name}")
                continue
            
            # Apply color to mesh (개선된 색상 적용)
            if hasattr(mesh, 'visual'):
                try:
                    # Convert color to RGBA (0-255)
                    if len(color) == 3:  # RGB
                        rgba_color = [int(c * 255) for c in color] + [255]
                    else:  # Already RGBA
                        rgba_color = [int(c * 255) if c <= 1.0 else int(c) for c in color]
                    
                    # 색상 적용
                    mesh.visual.face_colors = rgba_color
                    
                    # 색상 정보 출력
                    rgb_str = f"RGB({rgba_color[0]}, {rgba_color[1]}, {rgba_color[2]})"
                    print(f"   🎨 Applied color to {name}: {rgb_str}")
                    
                except Exception as color_error:
                    print(f"   ⚠️  Color application failed for {name}: {color_error}")
            
            combined_meshes.append(mesh)
            mesh_info.append(f"{name}: {len(mesh.vertices)} vertices")
        
        if not combined_meshes:
            print("❌ No valid meshes to combine")
            return None
        
        # Combine all meshes
        try:
            combined_mesh = trimesh.util.concatenate(combined_meshes)
            print("✅ Successfully combined all meshes:")
            for info in mesh_info:
                print(f"   - {info}")
            print(f"   Total: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")
            return combined_mesh
        except Exception as e:
            print(f"❌ Error combining meshes: {e}")
            return None
def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compose 3D scene from layout data')
    parser.add_argument(
        '--root', '-r', default="/source/sumin/stylin/FlairGPT/Scene_Synthesis/Result_txt", type=str,
        required=True,
        help='Path to folder containing layout.txt and (optionally) background_color.txt'
    )
    parser.add_argument(
        '--clip-results', '-c', default="/source/sumin/stylin/FlairGPT/Scene_Synthesis/Result_retrieval/clip_retrieval", type=str,
        required=True,
        help='Folder path to CLIP rerank result txt files'
    )
    parser.add_argument(
        '--output', '-o', default="/source/sumin/stylin/FlairGPT/Scene_Synthesis/Result", type=str,
        required=True,
        help='Folder to save final scene.glb and summary.json'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose traceback on error'
    )
    args = parser.parse_args()
    
    try:
            composer = SceneComposer(
                root_path=args.root,
                clip_results_path=args.clip_results
            )

            # output_path 오버라이드
            composer.output_path = os.path.abspath(args.output)
            os.makedirs(composer.output_path, exist_ok=True)

            composer.compose_scene()
            print("\n🎉 SUCCESS: Scene composition completed!")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
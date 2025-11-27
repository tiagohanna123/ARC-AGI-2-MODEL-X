"""
ARC-AGI-2 Solver using Model X Pattern Recognition

This module implements pattern recognition and transformation logic
for solving ARC-AGI-2 tasks.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy
import math

# Common marker color used in ARC-AGI tasks to indicate regions of interest
# Color 8 (azure/light blue) is frequently used as a marker in ARC tasks
MARKER_COLOR = 8


class Grid:
    """Represents an ARC-AGI grid with utility methods."""
    
    def __init__(self, data: List[List[int]]):
        self.data = data
        self.height = len(data)
        self.width = len(data[0]) if data else 0
    
    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return self.data == other.data
    
    def __repr__(self):
        return f"Grid({self.height}x{self.width})"
    
    def get_cell(self, row: int, col: int) -> int:
        """Get cell value at position."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.data[row][col]
        return -1
    
    def get_colors(self) -> set:
        """Get all unique colors in the grid."""
        colors = set()
        for row in self.data:
            colors.update(row)
        return colors
    
    def get_color_counts(self) -> Dict[int, int]:
        """Count occurrences of each color."""
        counts = {}
        for row in self.data:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
        return counts
    
    def rotate_90(self) -> 'Grid':
        """Rotate grid 90 degrees clockwise."""
        rotated = [[self.data[self.height - 1 - j][i] 
                   for j in range(self.height)] 
                  for i in range(self.width)]
        return Grid(rotated)
    
    def rotate_180(self) -> 'Grid':
        """Rotate grid 180 degrees."""
        return self.rotate_90().rotate_90()
    
    def rotate_270(self) -> 'Grid':
        """Rotate grid 270 degrees clockwise."""
        return self.rotate_90().rotate_90().rotate_90()
    
    def flip_horizontal(self) -> 'Grid':
        """Flip grid horizontally."""
        flipped = [row[::-1] for row in self.data]
        return Grid(flipped)
    
    def flip_vertical(self) -> 'Grid':
        """Flip grid vertically."""
        flipped = self.data[::-1]
        return Grid(flipped)
    
    def transpose(self) -> 'Grid':
        """Transpose the grid."""
        transposed = [[self.data[j][i] for j in range(self.height)] 
                      for i in range(self.width)]
        return Grid(transposed)
    
    def find_objects(self, background: int = 0) -> List[Tuple[int, int, int, int]]:
        """Find connected objects in the grid. Returns bounding boxes."""
        visited = [[False] * self.width for _ in range(self.height)]
        objects = []
        
        def bfs(start_row: int, start_col: int) -> Tuple[int, int, int, int]:
            """BFS to find connected component."""
            queue = [(start_row, start_col)]
            min_row, max_row = start_row, start_row
            min_col, max_col = start_col, start_col
            
            while queue:
                row, col = queue.pop(0)
                if visited[row][col]:
                    continue
                visited[row][col] = True
                
                min_row = min(min_row, row)
                max_row = max(max_row, row)
                min_col = min(min_col, col)
                max_col = max(max_col, col)
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.height and 0 <= nc < self.width 
                        and not visited[nr][nc] 
                        and self.data[nr][nc] != background):
                        queue.append((nr, nc))
            
            return (min_row, min_col, max_row, max_col)
        
        for row in range(self.height):
            for col in range(self.width):
                if not visited[row][col] and self.data[row][col] != background:
                    bbox = bfs(row, col)
                    objects.append(bbox)
        
        return objects
    
    def extract_region(self, min_row: int, min_col: int, 
                       max_row: int, max_col: int) -> 'Grid':
        """Extract a region from the grid."""
        region = [row[min_col:max_col+1] 
                 for row in self.data[min_row:max_row+1]]
        return Grid(region)
    
    def to_list(self) -> List[List[int]]:
        """Convert grid to list of lists."""
        return deepcopy(self.data)


class PatternMatcher:
    """Pattern matching and transformation detection for ARC-AGI tasks."""
    
    def __init__(self):
        self.transformations = []
    
    def analyze_size_change(self, input_grid: Grid, output_grid: Grid) -> Dict:
        """Analyze how the size changes between input and output."""
        return {
            'input_size': (input_grid.height, input_grid.width),
            'output_size': (output_grid.height, output_grid.width),
            'height_ratio': output_grid.height / input_grid.height if input_grid.height else 0,
            'width_ratio': output_grid.width / input_grid.width if input_grid.width else 0,
            'same_size': (input_grid.height == output_grid.height and 
                         input_grid.width == output_grid.width)
        }
    
    def analyze_color_mapping(self, input_grid: Grid, 
                               output_grid: Grid) -> Dict[int, int]:
        """Try to detect color mapping between input and output."""
        in_colors = input_grid.get_color_counts()
        out_colors = output_grid.get_color_counts()
        
        mapping = {}
        for in_color, in_count in in_colors.items():
            for out_color, out_count in out_colors.items():
                if in_count == out_count and out_color not in mapping.values():
                    mapping[in_color] = out_color
                    break
        
        return mapping
    
    def check_geometric_transform(self, input_grid: Grid, 
                                   output_grid: Grid) -> Optional[str]:
        """Check if output is a geometric transformation of input."""
        if input_grid == output_grid:
            return 'identity'
        if input_grid.rotate_90() == output_grid:
            return 'rotate_90'
        if input_grid.rotate_180() == output_grid:
            return 'rotate_180'
        if input_grid.rotate_270() == output_grid:
            return 'rotate_270'
        if input_grid.flip_horizontal() == output_grid:
            return 'flip_horizontal'
        if input_grid.flip_vertical() == output_grid:
            return 'flip_vertical'
        if input_grid.transpose() == output_grid:
            return 'transpose'
        return None
    
    def detect_pattern(self, train_pairs: List[Dict]) -> Dict:
        """Analyze training pairs to detect the transformation pattern."""
        pattern = {
            'geometric': [],
            'size_changes': [],
            'color_mappings': [],
            'consistent_geometric': None
        }
        
        for pair in train_pairs:
            input_grid = Grid(pair['input'])
            output_grid = Grid(pair['output'])
            
            # Check geometric transformations
            geo_transform = self.check_geometric_transform(input_grid, output_grid)
            pattern['geometric'].append(geo_transform)
            
            # Analyze size changes
            size_info = self.analyze_size_change(input_grid, output_grid)
            pattern['size_changes'].append(size_info)
            
            # Analyze color mappings
            color_map = self.analyze_color_mapping(input_grid, output_grid)
            pattern['color_mappings'].append(color_map)
        
        # Check if geometric transform is consistent
        if pattern['geometric'] and all(g == pattern['geometric'][0] 
                                         for g in pattern['geometric']):
            pattern['consistent_geometric'] = pattern['geometric'][0]
        
        return pattern


class ARCSolver:
    """Main solver for ARC-AGI-2 tasks."""
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
    
    def load_task(self, filepath: str) -> Dict:
        """Load a task from a JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def solve_task(self, task: Dict) -> List[List[List[int]]]:
        """Solve a task and return predictions for all test inputs."""
        train_pairs = task['train']
        test_pairs = task['test']
        
        # Detect pattern from training examples
        pattern = self.pattern_matcher.detect_pattern(train_pairs)
        
        predictions = []
        for test_case in test_pairs:
            input_grid = Grid(test_case['input'])
            prediction = self._apply_pattern(input_grid, pattern, train_pairs)
            predictions.append(prediction.to_list())
        
        return predictions
    
    def _apply_pattern(self, input_grid: Grid, pattern: Dict, 
                       train_pairs: List[Dict]) -> Grid:
        """Apply detected pattern to generate output."""
        
        # Try consistent geometric transformation first
        if pattern['consistent_geometric']:
            transform = pattern['consistent_geometric']
            if transform == 'identity':
                return input_grid
            elif transform == 'rotate_90':
                return input_grid.rotate_90()
            elif transform == 'rotate_180':
                return input_grid.rotate_180()
            elif transform == 'rotate_270':
                return input_grid.rotate_270()
            elif transform == 'flip_horizontal':
                return input_grid.flip_horizontal()
            elif transform == 'flip_vertical':
                return input_grid.flip_vertical()
            elif transform == 'transpose':
                return input_grid.transpose()
        
        # Try to infer from size patterns
        if pattern['size_changes']:
            size_info = pattern['size_changes'][0]
            
            # If output is smaller, try to extract a region
            if size_info['height_ratio'] < 1 or size_info['width_ratio'] < 1:
                return self._extract_pattern(input_grid, train_pairs)
            
            # If same size, try color replacement or local transforms
            if size_info['same_size']:
                return self._apply_local_transform(input_grid, train_pairs)
        
        # Default: return input unchanged
        return input_grid
    
    def _extract_pattern(self, input_grid: Grid, 
                         train_pairs: List[Dict]) -> Grid:
        """Extract a pattern or region from the input."""
        # Try to find objects marked with the marker color
        marker_colors = {MARKER_COLOR}
        
        for row_idx, row in enumerate(input_grid.data):
            for col_idx, cell in enumerate(row):
                if cell in marker_colors:
                    # Found marker, try to find the region to extract
                    return self._find_marked_region(input_grid, row_idx, col_idx, 
                                                    train_pairs)
        
        # If no marker, try to find unique objects
        objects = input_grid.find_objects()
        if objects:
            # Return the first non-trivial object
            for bbox in objects:
                min_row, min_col, max_row, max_col = bbox
                region = input_grid.extract_region(min_row, min_col, 
                                                   max_row, max_col)
                if region.height > 1 or region.width > 1:
                    return region
        
        return input_grid
    
    def _find_marked_region(self, input_grid: Grid, marker_row: int, 
                            marker_col: int, train_pairs: List[Dict]) -> Grid:
        """Find and extract a region marked by a specific pattern."""
        # Look for the expected output size from training
        if train_pairs:
            expected_h = len(train_pairs[0]['output'])
            expected_w = len(train_pairs[0]['output'][0])
            
            # Find region with marker color and extract adjacent region
            marker_positions = []
            for r, row in enumerate(input_grid.data):
                for c, cell in enumerate(row):
                    if cell == MARKER_COLOR:
                        marker_positions.append((r, c))
            
            if marker_positions:
                # Find bounding box of non-marker region adjacent to markers
                min_r = min(p[0] for p in marker_positions)
                max_r = max(p[0] for p in marker_positions)
                
                # Look for corresponding region
                # Extract the mirrored/symmetric region
                result_data = []
                for r in range(expected_h):
                    row_data = []
                    for c in range(expected_w):
                        # Map from output position to input position
                        in_r = min_r + r
                        # Find the symmetric position
                        in_c = marker_col - expected_w + c
                        if 0 <= in_r < input_grid.height and 0 <= in_c < input_grid.width:
                            val = input_grid.data[in_r][in_c]
                            if val != MARKER_COLOR:
                                row_data.append(val)
                            else:
                                row_data.append(0)
                        else:
                            row_data.append(0)
                    result_data.append(row_data)
                
                return Grid(result_data)
        
        return input_grid
    
    def _apply_local_transform(self, input_grid: Grid, 
                               train_pairs: List[Dict]) -> Grid:
        """Apply local transformations based on training examples."""
        result = deepcopy(input_grid.data)
        
        # Try to learn cell-by-cell transformation
        if train_pairs:
            train_in = Grid(train_pairs[0]['input'])
            train_out = Grid(train_pairs[0]['output'])
            
            # Find color mappings
            color_map = {}
            for r in range(min(train_in.height, train_out.height)):
                for c in range(min(train_in.width, train_out.width)):
                    in_val = train_in.data[r][c]
                    out_val = train_out.data[r][c]
                    if in_val != out_val:
                        color_map[in_val] = out_val
            
            # Apply color mapping
            for r in range(input_grid.height):
                for c in range(input_grid.width):
                    if result[r][c] in color_map:
                        result[r][c] = color_map[result[r][c]]
        
        return Grid(result)
    
    def evaluate_prediction(self, prediction: List[List[int]], 
                           expected: List[List[int]]) -> bool:
        """Check if prediction matches expected output."""
        return prediction == expected
    
    def solve_file(self, filepath: str) -> List[List[List[int]]]:
        """Load and solve a task from file."""
        task = self.load_task(filepath)
        return self.solve_task(task)


def solve_arc_task(task_path: str) -> List[List[List[int]]]:
    """Convenience function to solve an ARC-AGI task."""
    solver = ARCSolver()
    return solver.solve_file(task_path)


def evaluate_task(task_path: str) -> Dict[str, Any]:
    """Evaluate solver performance on a task with known answers."""
    solver = ARCSolver()
    task = solver.load_task(task_path)
    predictions = solver.solve_task(task)
    
    results = {
        'task_file': os.path.basename(task_path),
        'num_test_cases': len(task['test']),
        'predictions': predictions,
        'correct': []
    }
    
    for i, test_case in enumerate(task['test']):
        if 'output' in test_case:
            is_correct = solver.evaluate_prediction(predictions[i], 
                                                    test_case['output'])
            results['correct'].append(is_correct)
    
    return results

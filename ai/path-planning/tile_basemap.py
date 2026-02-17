"""
Helper module to load Google Maps tiles as matplotlib basemap
"""
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np


class TileBasemap:
    """Load and display downloaded Google Maps tiles as basemap"""
    
    def __init__(self, tiles_dir="tiles_z19_v2", index_file="tiles_index.csv"):
        """
        Initialize tile basemap loader
        
        Args:
            tiles_dir: Directory containing downloaded tiles
            index_file: CSV file with tile index
        """
        self.tiles_dir = Path(tiles_dir)
        self.index_path = self.tiles_dir / index_file
        self.tiles = []
        
        if self.index_path.exists():
            self._load_index()
        else:
            print(f"Warning: Tile index not found at {self.index_path}")
            print("Run download_map_tiles.py first to download tiles")
    
    def _load_index(self):
        """Load tile index from CSV"""
        with open(self.index_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] in ['OK', 'SKIP_EXISTS']:
                    self.tiles.append({
                        'x': int(row['x']),
                        'y': int(row['y']),
                        'z': int(row['z']),
                        'file': Path(row['file']),
                        'lat_min': float(row['lat_min']),
                        'lon_min': float(row['lon_min']),
                        'lat_max': float(row['lat_max']),
                        'lon_max': float(row['lon_max'])
                    })
        print(f"Loaded {len(self.tiles)} tiles from index")
    
    def add_to_plot(self, ax, alpha=0.6):
        """
        Add tiles to matplotlib axis as basemap
        
        Args:
            ax: matplotlib axis
            alpha: transparency (0-1)
        """
        if not self.tiles:
            print("No tiles available. Skipping basemap.")
            return
        
        for tile in self.tiles:
            if not tile['file'].exists():
                continue
            
            try:
                # Load tile image
                img = Image.open(tile['file'])
                img_array = np.array(img)
                
                # Plot at correct geographic position
                extent = [
                    tile['lon_min'], 
                    tile['lon_max'],
                    tile['lat_min'], 
                    tile['lat_max']
                ]
                
                ax.imshow(img_array, extent=extent, aspect='auto', 
                         alpha=alpha, zorder=0, interpolation='bilinear')
                
            except Exception as e:
                print(f"Error loading tile {tile['file']}: {e}")
        
        print(f"Added {len(self.tiles)} tiles to basemap")

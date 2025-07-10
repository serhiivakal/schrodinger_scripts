#!/usr/bin/env python
#############################################################################
# Script for converting PoseViewer files into protein-ligand complexes      #
# Modified version of Dr. Serhii Vakal's original script, Orion Pharma 2025 #
#                                                                           #
# Workflow:                                                                 #
# 1. Takes a single maegz file in PoseViewer format as input                #
# 2. Extracts receptor structure from the first entry                       #
# 3. Creates protein-ligand complexes for each pose                         #
# 4. Names each complex according to the ligand name                        #
# 5. Preserves all properties from the original structures                  #
# 6. Exports combined structures as *_complexes.maegz                       #
# 7. Exports ligand properties to a CSV file (*_properties.csv)             #
#                                                                           #
# Input:                                                                    #
# - PoseViewer file (*.maegz)                                               #
#   - First structure must be the receptor                                  #
#   - Following structures must be ligand poses                             #
#                                                                           #
# Output:                                                                   #
# - Combined complexes file (*_complexes.maegz)                             #
#   - Each complex contains receptor + single ligand pose                   #
#   - Preserves all properties from original structures                     #
# - Properties file (*_properties.csv)                                      #
#   - Contains all ligand properties in CSV format                          #
#   - Preserves r_user_* properties                                         #
#                                                                           #
# Requirements:                                                             #
# - Schrödinger Python API                                                  #
# - Write permissions in target directory                                   #
#                                                                           #
# To run the script:                                                        #
# $SCHRODINGER/run python3 transform-poseviewer-into-complexes.py input.maegz #
# Optional: -o output.maegz -p properties.csv -v (verbose mode)             #
#                                                                           #
#############################################################################

from schrodinger import structure
import os
import sys
import logging
import csv
from datetime import datetime
import argparse

class PoseViewerTransformer:
    def __init__(self):
        self.setup_logging()
        self.receptor = None
        self.ligands = []

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PoseViewerTransformer')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def generate_output_filename(self, input_file: str) -> str:
        """Generate output filename by adding _complexes before .maegz"""
        # Split the filename at .maegz
        base = input_file.rsplit('.maegz', 1)[0]
        return f"{base}_complexes.maegz"
        
    def generate_properties_filename(self, input_file: str) -> str:
        """Generate properties filename by adding _properties.csv"""
        # Split the filename at .maegz
        base = input_file.rsplit('.maegz', 1)[0]
        return f"{base}_properties.csv"

    def process_structures(self, input_file: str):
        """Read and process structures from input file"""
        with structure.StructureReader(input_file) as reader:
            # First structure should be receptor
            self.receptor = next(reader)
            self.receptor_name = self.receptor.title
            self.logger.info(f"Extracted receptor: {self.receptor_name}")

            # Process all ligand structures
            for st in reader:
                self.ligands.append(st)
                self.logger.debug(f"Added ligand: {st.title}")

    def create_complexes(self) -> list:
        """Create complex structures by combining receptor with each ligand"""
        complexes = []
        
        for lig in self.ligands:
            # Create new structure for complex
            complex_st = self.receptor.copy()
            
            # Extend the structure with new atoms
            complex_st.extend(lig)
            
            # Create complex title using the ligand's original name
            complex_st.title = f"{self.receptor_name}_{lig.title}"
            
            # Copy all properties from ligand to complex
            for prop in lig.property.keys():
                complex_st.property[prop] = lig.property[prop]
            
            complexes.append(complex_st)
            self.logger.debug(f"Created complex: {complex_st.title} with {len(complex_st.atom)} atoms")
                
        return complexes
    
    def write_complexes(self, complexes: list, output_file: str):
        """Write complex structures to output file"""
        with structure.StructureWriter(output_file) as writer:
            for complex_st in complexes:
                writer.append(complex_st)
        self.logger.info(f"Wrote {len(complexes)} complexes to {output_file}")
        
    def write_properties(self, properties_file: str):
        """Write ligand properties to CSV file"""
        # Get all unique property keys across all ligands
        all_keys = set()
        for lig in self.ligands:
            all_keys.update(lig.property.keys())
        
        # Filter properties to keep:
        # 1. All r_user_* properties
        # 2. Other important properties
        # 3. Exclude system properties that aren't useful
        filtered_keys = []
        for k in all_keys:
            # Keep all r_user_* properties
            if k.startswith('r_user_'):
                filtered_keys.append(k)
            # Keep other important properties but filter out system properties
            elif not (k.startswith('s_') or k.startswith('i_')):
                filtered_keys.append(k)
                
        # Log the properties we're keeping
        self.logger.debug(f"Properties being saved: {', '.join(filtered_keys)}")
        
        # Sort keys for consistent output
        header = ['Title'] + sorted(filtered_keys)
        
        with open(properties_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            
            for lig in self.ligands:
                # Create row with ligand title
                row = {'Title': lig.title}
                
                # Add all properties
                for key in filtered_keys:
                    if key in lig.property:
                        row[key] = lig.property[key]
                    else:
                        row[key] = ''
                        
                writer.writerow(row)
                
        self.logger.info(f"Wrote properties for {len(self.ligands)} ligands to {properties_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transform PoseViewer file into complexes")
    parser.add_argument("input_file", help="Input .maegz file in PoseViewer format (receptor + poses)")
    parser.add_argument("-o", "--output", help="Output file name (default: input_complexes.maegz)")
    parser.add_argument("-p", "--properties", help="Properties CSV file name (default: input_properties.csv)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Create transformer instance
    transformer = PoseViewerTransformer()
    
    # Set verbose logging if requested
    if args.verbose:
        transformer.logger.setLevel(logging.DEBUG)

    # Process input file
    transformer.logger.info(f"Processing input file: {args.input_file}")
    
    # Generate output filename if not specified
    if args.output:
        output_file = args.output
    else:
        output_file = transformer.generate_output_filename(args.input_file)
    transformer.logger.info(f"Output will be written to: {output_file}")
    
    # Generate properties filename if not specified
    if args.properties:
        properties_file = args.properties
    else:
        properties_file = transformer.generate_properties_filename(args.input_file)
    transformer.logger.info(f"Properties will be written to: {properties_file}")

    # Process structures
    transformer.process_structures(args.input_file)
    transformer.logger.info(f"Read {len(transformer.ligands)} ligand structures")

    # Create and write complexes
    complexes = transformer.create_complexes()
    transformer.write_complexes(complexes, output_file)
    
    # Write properties to CSV
    transformer.write_properties(properties_file)
    
    transformer.logger.info("Transformation complete!")

if __name__ == "__main__":
    main()
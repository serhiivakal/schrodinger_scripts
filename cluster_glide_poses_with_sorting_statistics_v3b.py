######################################################################################
# Script for clustering and analyzing Glide docking poses                            #
# Developed by Dr. Serhii Vakal, Orion Pharma 2025                                   #
#                                                                                    #
# Workflow:                                                                          #
# 1. Reads Glide pose viewer file or ligands-only structure file                     #
# 2. Automatically detects receptor presence and extracts poses                      #
# 3. Performs RMSD-based clustering with configurable threshold (Autodock4 criteria) #
# 4. Sorts poses by Glide score and/or structure names                               #
# 5. Exports clustered poses with cluster statistics                                 #
#                                                                                    #
# Input:                                                                             #
# Required:                                                                          #
#   --input/-i : Input structure file (.maegz)                                       #
# Optional:                                                                          #
#   --output/-o : Output file (default: input_clustered_poses.maegz)                 #
#   --rmsd : RMSD threshold in Å (default: 2.0)                                      #
#   --write-mode : 'representatives' or 'all' (default: representatives)             #
#   --sort-mode : 'score', 'name', or 'both' (default: name)                         #
#                                                                                    #
# Output:                                                                            #
# - Clustered poses file with added properties:                                      #
#   - i_cluster_number : Cluster ID                                                  #
#   - i_cluster_size : Number of poses in cluster                                    #
#   - b_cluster_representative : True for cluster representatives                    #
# - Detailed clustering log (pose_clustering.log)                                    #
#                                                                                    #
# Requirements:                                                                      #
# - Schrödinger Python API                                                           #
# - NumPy package                                                                    #
# - Valid Glide docking output                                                       #
#                                                                                    #
# To run the script:                                                                 #
# $SCHRODINGER/run python3 cluster_glide_poses_with_sorting_statistics_v3b.py \      #
#    -i input-file-with-docking-poses.maegz                                          #
######################################################################################

#!/usr/bin/env python
from schrodinger import structure
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple

class GlidePoseClusterer:
    def __init__(self, rmsd_threshold: float = 2.0):
        self.rmsd_threshold = rmsd_threshold
        self.clusters = []
        self.receptor = None
        self.setup_logging()

    def setup_logging(self):
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create and configure file handler
        file_handler = logging.FileHandler('clustering_results.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)

    def process_input_structures(self, structures: List[structure.Structure]) -> List[structure.Structure]:
        """
        Process input structures intelligently:
        - Detect if first structure is receptor (Glide Pose Viewer format)
        - Handle cases with only ligands
        Returns list of pose structures
        """
        if not structures:
            self.logger.error("No structures found in input file!")
            return []

        # Check if first structure is receptor by looking for protein atoms
        first_st = structures[0]
        # Using correct Schrodinger API method for atom selection
        protein_atoms = [at for at in first_st.atom if at.pdbres.strip()]  # Protein atoms have PDB residue names
        
        if protein_atoms:
            # This is likely a Glide Pose Viewer file with receptor
            self.receptor = first_st
            poses = structures[1:]  # Take all structures except first
            self.logger.info(f"Found receptor structure with {len(protein_atoms)} protein atoms")
            self.logger.info(f"Extracted {len(poses)} docking poses")
        else:
            # No receptor found, treat all structures as poses
            poses = structures
            self.logger.info("No receptor found in input file")
            self.logger.info(f"Processing all {len(poses)} structures as poses")

        # Validate that we have poses to cluster
        if not poses:
            self.logger.error("No poses found for clustering!")
            return []

        return poses

    def calculate_rmsd(self, st1, st2):
        """Calculate RMSD between two structures"""
        coords1 = []
        coords2 = []
        
        for at1, at2 in zip(st1.atom, st2.atom):
            if at1.atomic_number > 1:  # Skip hydrogens
                coords1.append([at1.x, at1.y, at1.z])
                coords2.append([at2.x, at2.y, at2.z])
        
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        
        diff = coords1 - coords2
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        return rmsd

    def cluster_poses(self, structures):
        self.logger.debug("Starting clustering process")
        
        # First, group structures by ligand title, preserving numbering
        ligand_groups = {}
        for struct in structures:
            # Handle ORM compounds differently from other naming patterns
            if struct.title.startswith('ORM-'):
                ligand_title = struct.title.split('_')[0]
            else:
                ligand_title = struct.title  # Keep the full name including numbering
                
            if ligand_title not in ligand_groups:
                ligand_groups[ligand_title] = []
            ligand_groups[ligand_title].append(struct)
        
        # Process each ligand group separately
        for ligand_title in sorted(ligand_groups.keys()):
            self.logger.info(f"\nProcessing ligand: {ligand_title}")
            poses = ligand_groups[ligand_title]
            
            # Validate poses have required properties
            valid_poses = []
            for pose in poses:
                try:
                    score = pose.property.get('r_i_docking_score')
                    if score is None:
                        self.logger.warning(f"Skipping pose of {ligand_title} - missing docking score")
                        continue
                    valid_poses.append(pose)
                except Exception as e:
                    self.logger.warning(f"Error processing pose of {ligand_title}: {str(e)}")
                    continue
            
            if not valid_poses:
                self.logger.error(f"No valid poses found for {ligand_title}")
                continue
                
            # Sort poses within this ligand group by Glide score
            try:
                valid_poses.sort(key=lambda x: x.property['r_i_docking_score'])
            except Exception as e:
                self.logger.error(f"Error sorting poses for {ligand_title}: {str(e)}")
                continue
            
            # Start new cluster group for this ligand
            ligand_clusters = [[valid_poses[0]]]
            
            # Cluster remaining poses for this ligand
            for i, current_st in enumerate(valid_poses[1:], 1):
                assigned = False
                
                for cluster in ligand_clusters:
                    try:
                        rmsd = self.calculate_rmsd(cluster[0], current_st)
                        self.logger.debug(f"RMSD between pose {i+1} and cluster rep: {rmsd:.2f}")
                        
                        if rmsd < self.rmsd_threshold:
                            cluster.append(current_st)
                            assigned = True
                            break
                    except Exception as e:
                        self.logger.warning(f"RMSD calculation failed for {ligand_title} pose {i+1}: {str(e)}")
                        continue
                
                if not assigned:
                    ligand_clusters.append([current_st])
            
            # Add this ligand's clusters to the main clusters list
            self.clusters.extend(ligand_clusters)
            
            # Log clustering results for this ligand
            self.logger.info(f"Created {len(ligand_clusters)} clusters for {ligand_title}")
            cluster_sizes = [len(c) for c in ligand_clusters]
            self.logger.info(f"Cluster sizes: {cluster_sizes}")
        
        # Write final summary
        if self.clusters:
            self.logger.info("\n=== CLUSTERING SUMMARY ===")
            self.logger.info(f"Total poses processed: {sum(len(cluster) for cluster in self.clusters)}")
            self.logger.info(f"Number of clusters: {len(self.clusters)}")
        else:
            self.logger.error("No clusters were created!")

    def sort_representatives(self, mode='score'):
        """
        Sort cluster representatives by name or docking score
        mode: 'score' (default), 'name', or 'both'
        """
        sorted_clusters = []
        
        cluster_data = []
        for i, cluster in enumerate(self.clusters):
            rep = cluster[0]
            score = rep.property.get('r_i_glide_gscore', 0.0)
            name = rep.title
            cluster_data.append({
                'cluster': cluster,
                'score': score,
                'name': name,
                'original_index': i
            })

        if mode == 'score':
            sorted_data = sorted(cluster_data, key=lambda x: x['score'])
        elif mode == 'name':
            sorted_data = sorted(cluster_data, key=lambda x: x['name'])
        elif mode == 'both':
            sorted_data = sorted(cluster_data, key=lambda x: (x['score'], x['name']))
        else:
            self.logger.warning(f"Unknown sorting mode: {mode}. Using 'score'")
            sorted_data = sorted(cluster_data, key=lambda x: x['score'])

        self.clusters = [item['cluster'] for item in sorted_data]
        
        self.logger.info(f"\nSorted representatives ({mode}):")
        for i, item in enumerate(sorted_data):
            self.logger.info(f"Position {i+1}: {item['name']}, Score: {item['score']:.2f}")

    def write_output(self, output_file: str, write_mode: str = 'representatives', sort_mode='both'):
        """Write clustered poses to output file with sorting"""
        self.sort_representatives(mode=sort_mode)
        
        with structure.StructureWriter(output_file) as writer:
            # Write receptor first if it exists
            if self.receptor:
                writer.append(self.receptor)
                self.logger.info("Wrote receptor structure to output file")
            else:
                self.logger.info("No receptor to write (ligands-only mode)")
            
            # Write clustered poses
            poses_written = 0
            for i, cluster in enumerate(self.clusters):
                if write_mode == 'representatives':
                    st = cluster[0]
                    st.property['i_cluster_number'] = i + 1
                    st.property['i_cluster_size'] = len(cluster)
                    st.property['r_i_glide_gscore'] = st.property.get('r_i_glide_gscore', 0.0)
                    writer.append(st)
                    poses_written += 1
                else:
                    for j, st in enumerate(cluster):
                        st.property['i_cluster_number'] = i + 1
                        st.property['i_cluster_size'] = len(cluster)
                        st.property['b_cluster_representative'] = (j == 0)
                        st.property['r_i_glide_gscore'] = st.property.get('r_i_glide_gscore', 0.0)
                        writer.append(st)
                        poses_written += 1

            self.logger.info(f"Wrote {poses_written} poses to output file")

def generate_output_filename(input_file: str) -> str:
    """Generate output filename based on input filename"""
    input_path = Path(input_file)
    new_stem = f"{input_path.stem}_clustered_poses"
    output_path = input_path.with_stem(new_stem)
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Cluster Glide docking poses')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input structure file (.maegz) - can be Glide Pose Viewer file or ligands-only file')
    parser.add_argument('--output', '-o', required=False, 
                       help='Output file for clustered poses (optional)')
    parser.add_argument('--rmsd', type=float, default=2.0, 
                       help='RMSD threshold for clustering (Å)')
    parser.add_argument('--write-mode', choices=['representatives', 'all'], 
                       default='representatives', help='Output mode')
    parser.add_argument('--sort-mode', choices=['score', 'name', 'both'],
                       default='name', help='Sorting mode for representatives')
    
    args = parser.parse_args()

    # Generate output filename if not provided
    output_file = args.output if args.output else generate_output_filename(args.input)
    print(f"Input file: {args.input}")
    print(f"Output will be written to: {output_file}")

    # Read all structures
    structures = []
    with structure.StructureReader(args.input) as reader:
        for st in reader:
            structures.append(st)
    
    print(f"Read {len(structures)} structures")

    # Create clusterer instance
    clusterer = GlidePoseClusterer(rmsd_threshold=args.rmsd)
    
    # Process input structures (handles both with and without receptor)
    poses = clusterer.process_input_structures(structures)
    
    if poses:
        # Perform clustering on poses
        clusterer.cluster_poses(poses)
        
        # Write sorted output
        clusterer.write_output(output_file, args.write_mode, args.sort_mode)
        print("Clustering and sorting complete!")
    else:
        print("Error: No poses to cluster!")
        sys.exit(1)

if __name__ == "__main__":
    main()
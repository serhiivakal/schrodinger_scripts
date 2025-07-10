#####################################################################################
# Script for selecting and extracting major cluster representatives                 #
# Developed by Dr. Serhii Vakal, Orion Pharma 2025                                  #
#                                                                                   #
# Workflow:                                                                         #
# 1. Locates clustered pose viewer files from previous docking analysis             #
# 2. Processes structures to identify major and best-scored clusters                #
# 3. Extracts representatives based on cluster size and Glide scores                #
# 4. Exports selected poses with descriptive titles                                 #
#                                                                                   #
# Selection criteria:                                                               #
# - For each compound:                                                              #
#   * Identifies the largest cluster                                                #
#   * Finds the best scoring pose                                                   #
#   * If largest cluster contains best score: exports one representative            #
#   * If different: exports both cluster and score representatives                  #
#                                                                                   #
# Input:                                                                            #
# - Directory containing clustered pose viewer files (*pv_clustered_poses.maegz)    #
#                                                                                   #
# Output:                                                                           #
# - New structure file (*_major_clusters.maegz) containing:                         #
#   * Original receptor structure (if present)                                      #
#   * Representative poses with modified titles:                                    #
#     - <compound>_size-major : largest cluster representative                      #
#     - <compound>_score-best : best scoring pose                                   #
#     - <compound>_size-major_score-best : when both criteria match                 #
#                                                                                   #
# Requirements:                                                                     #
# - Schrödinger Python API                                                          #
# - Previously clustered pose viewer file                                           #
# - Proper Glide docking properties in input structures                             #
#                                                                                   #
# To run the script:                                                                #
# $SCHRODINGER/run python3 select_major_clusters.py [directory_path]                #
# If no directory is specified, the current directory will be used.                 #
#                                                                                   #
# Logging:                                                                          #
# - Detailed execution log saved to select_clusters.log                             #
# - Console output for progress monitoring                                          #
#####################################################################################

import os
import sys
import logging
import traceback
print("1. Imports completed successfully")  # DEBUG

try:
    from schrodinger import structure
    from schrodinger.structure import StructureReader, StructureWriter
    print("2. Schrodinger imports successful")  # DEBUG
except Exception as e:
    print(f"ERROR importing Schrodinger modules: {e}")
    sys.exit(1)

def setup_logging():
    print("3. Entering setup_logging")  # DEBUG
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('select_clusters.log')
            ]
        )
        print("4. Logging configuration completed")  # DEBUG
        return logging.getLogger(__name__)
    except Exception as e:
        print(f"ERROR in setup_logging: {e}")
        raise

def find_clustered_file(directory='.'):
    print("5. Entering find_clustered_file")  # DEBUG
    logger = logging.getLogger(__name__)
    
    print(f"6. Current directory: {os.getcwd()}")  # DEBUG
    print(f"7. Looking in: {os.path.abspath(directory)}")  # DEBUG
    
    if not os.path.exists(directory):
        print(f"8a. ERROR: Directory not found: {directory}")  # DEBUG
        return None
    
    print("8b. Directory exists, checking contents")  # DEBUG
    all_files = os.listdir(directory)
    print(f"9. All files in directory: {all_files}")  # DEBUG
    
    files = [f for f in all_files if 'pv_clustered_poses' in f and f.endswith('.maegz')]
    print(f"10. Matching files found: {files}")  # DEBUG

    if not files:
        print("11a. No matching files found")  # DEBUG
        return None

    selected_file = os.path.join(directory, files[0])
    print(f"11b. Selected file: {selected_file}")  # DEBUG
    return selected_file

def extract_major_representatives(input_file):
    print(f"12. Entering extract_major_representatives with file: {input_file}")
    logger = logging.getLogger(__name__)

    try:
        print("13. Checking file existence")
        if not os.path.exists(input_file):
            print(f"14a. ERROR: Input file not found: {input_file}")
            return False

        print("14b. File exists, attempting to read")
        output_file = input_file.replace('.maegz', '_major_clusters.maegz')
        print(f"15. Will write output to: {output_file}")

        structures_to_write = []
        compounds = {}
        
        with StructureReader(input_file) as reader:
            print("16. Successfully opened file with StructureReader")
            for structure in reader:
                base_title = structure.property.get('s_m_title', '')
                
                if base_title.lower().startswith('receptor'):
                    structures_to_write.append(structure)
                    continue
                
                if base_title not in compounds:
                    compounds[base_title] = []
                compounds[base_title].append(structure)

        total_structures = 0
        processed_compounds = 0
        
        for base_title, structures in compounds.items():
            # Find largest cluster and best score for this compound
            max_cluster_size = max(st.property.get('i_cluster_size', 0) for st in structures)
            best_score = min(st.property.get('r_i_glide_gscore', float('inf')) for st in structures)
            
            # Find structure with largest cluster
            largest_cluster_structure = None
            largest_cluster_score = float('inf')
            
            for st in structures:
                if st.property.get('i_cluster_size', 0) == max_cluster_size:
                    score = st.property.get('r_i_glide_gscore', float('inf'))
                    if largest_cluster_structure is None or score < largest_cluster_score:
                        largest_cluster_structure = st
                        largest_cluster_score = score
            
            # Check if largest cluster is also best scoring
            if abs(largest_cluster_score - best_score) < 1e-6:  # Using small threshold for float comparison
                # Case 2: Largest cluster is also best scoring
                largest_cluster_structure.property['s_m_title'] = f"{base_title}_size-major_score-best"
                structures_to_write.append(largest_cluster_structure)
                print(f"Added combined major/best for {base_title} (size: {max_cluster_size}, score: {best_score})")
            else:
                # Case 1: Largest cluster is not best scoring
                largest_cluster_structure.property['s_m_title'] = f"{base_title}_size-major"
                structures_to_write.append(largest_cluster_structure)
                print(f"Added largest cluster for {base_title} (size: {max_cluster_size})")
                
                # Find and add best scoring structure
                for st in structures:
                    if abs(st.property.get('r_i_glide_gscore', float('inf')) - best_score) < 1e-6:
                        st.property['s_m_title'] = f"{base_title}_score-best"
                        structures_to_write.append(st)
                        print(f"Added best scoring structure for {base_title} (score: {best_score})")
                        break
            
            total_structures += len(structures)
            processed_compounds += 1

        print(f"17. Processed {processed_compounds} compounds with {total_structures} total structures")

        if structures_to_write:
            print(f"18. Writing {len(structures_to_write)} structures to {output_file}")
            with StructureWriter(output_file) as writer:
                for structure in structures_to_write:
                    writer.append(structure)
            print(f"19. Successfully wrote {len(structures_to_write)} structures to output file")
            return True
        else:
            print("18. No structures to write!")
            return False

    except Exception as e:
        print(f"ERROR in extract_major_representatives: {e}")
        traceback.print_exc()
        return False
        
def main():
    print("17. Entering main function")  # DEBUG
    try:
        logger = setup_logging()
        print("18. Logger setup complete")  # DEBUG

        # Check if directory argument is provided
        if len(sys.argv) > 1:
            directory = sys.argv[1]
            print(f"Using directory from command line: {directory}")
        else:
            directory = '.'
            print("No directory specified, using current directory")

        input_file = find_clustered_file(directory)
        print(f"19. Found input file: {input_file}")  # DEBUG

        if input_file:
            success = extract_major_representatives(input_file)
            print(f"20. Processing complete, success={success}")  # DEBUG
        else:
            print("21. No input file found")  # DEBUG
            sys.exit(1)

    except Exception as e:
        print(f"ERROR in main: {e}")  # DEBUG
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("=== Script Starting ===")  # DEBUG
    main()
    print("=== Script Finished ===")  # DEBUG
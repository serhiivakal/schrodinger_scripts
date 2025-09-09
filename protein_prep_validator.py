#!/usr/bin/env python3
"""
Protein Structure Preparation Validator
======================================

A comprehensive validation tool to check protein structures for potential issues
that could cause grid generation failures, particularly Lewis structure problems.

Author: Dr. Serhii Vakal & Anthropic Claude-Sonnet-4
Organization: Orion Pharma, Turku, Finland
Date: September 2025

This tool performs detailed validation of prepared protein structures to identify
potential issues before attempting grid generation with Glide.

Usage:
$SCHRODINGER/run python3 protein_structure_validator.py XXX_pdbs_for_test_aligned_proteins_prep/proteins_prep/ XXX_validation_output/
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Set
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from collections import defaultdict, Counter

# Try to import Schrodinger modules
try:
    from schrodinger.structure import StructureReader, Structure
    from schrodinger.structutils import analyze, transform
    from schrodinger.infra import mm
    SCHRODINGER_AVAILABLE = True
except ImportError:
    SCHRODINGER_AVAILABLE = False
    print("Warning: Schrodinger modules not available. Structure validation will be limited.")

# Try to import Glide modules for compatibility testing
try:
    from schrodinger import glide
    from schrodinger.application.glide import glide as glide_input
    GLIDE_AVAILABLE = True
except ImportError:
    GLIDE_AVAILABLE = False
    print("Warning: Glide modules not available. Glide-specific validation will be disabled.")

@dataclass
class AtomIssue:
    """Data structure for individual atom problems."""
    atom_index: int
    atom_name: str
    residue: str
    chain: str
    residue_number: int
    element: str
    issue_type: str
    description: str
    severity: str  # "critical", "warning", "info"
    suggestion: str

@dataclass
class BondIssue:
    """Data structure for bond-related problems."""
    atom1_index: int
    atom2_index: int
    atom1_name: str
    atom2_name: str
    residue1: str
    residue2: str
    bond_order: float
    expected_order: Optional[float]
    issue_type: str
    description: str
    severity: str
    suggestion: str

@dataclass
class StructureValidationResult:
    """Complete validation result for a structure."""
    file_path: str
    structure_name: str
    is_valid: bool
    overall_score: float  # 0-100, higher is better
    atom_issues: List[AtomIssue]
    bond_issues: List[BondIssue]
    general_issues: List[str]
    recommendations: List[str]
    statistics: Dict
    grid_generation_likely_to_succeed: bool

class ProteinStructureValidator:
    """Main validator class for protein structure preparation issues."""
    
    def __init__(self, output_dir: str, debug: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        # Create subdirectories
        self.reports_dir = self.output_dir / "validation_reports"
        self.logs_dir = self.output_dir / "logs"
        self.fixed_structures_dir = self.output_dir / "suggested_fixes"
        
        for directory in [self.reports_dir, self.logs_dir, self.fixed_structures_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        
        # Validation results storage
        self.validation_results: Dict[str, StructureValidationResult] = {}
        
        # Known problematic patterns
        self.problematic_residues = {
            'MSE', 'SEP', 'TPO', 'PTR', 'TYS', 'CSD', 'HYP', 'M3L'
        }
        
        self.problematic_elements = {
            'SE', 'AS', 'BR', 'I', 'AL', 'SI'
        }
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        self.logger = logging.getLogger('ProteinValidator')
        self.logger.setLevel(log_level)
        
        # Create file handler
        log_file = self.logs_dir / 'validation.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger (avoid duplicates)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def validate_structures(self, input_dir: str):
        """Main method to validate all structures in a directory."""
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        self.logger.info(f"Starting protein structure validation in {input_dir}")
        
        # Find structure files
        structure_files = []
        patterns = ["*.mae", "*.maegz", "*.pdb", "*.PDB"]
        
        for pattern in patterns:
            files = list(input_path.glob(pattern))
            structure_files.extend(files)
            if files:
                self.logger.info(f"Found {len(files)} {pattern} files")
        
        if not structure_files:
            raise ValueError(f"No structure files found in {input_dir}")
        
        self.logger.info(f"Total structure files to validate: {len(structure_files)}")
        
        # Validate each structure
        for structure_file in structure_files:
            try:
                self.logger.info(f"Validating {structure_file.name}...")
                result = self._validate_single_structure(structure_file)
                self.validation_results[result.structure_name] = result
                
                if result.is_valid:
                    self.logger.info(f"✅ {structure_file.name}: PASSED (score: {result.overall_score:.1f})")
                else:
                    self.logger.warning(f"❌ {structure_file.name}: FAILED (score: {result.overall_score:.1f})")
                    
            except Exception as e:
                self.logger.error(f"Error validating {structure_file.name}: {str(e)}")
                if self.debug:
                    self.logger.error(traceback.format_exc())
        
        # Generate reports
        self._generate_reports()
        self._print_summary()
    
    def _validate_single_structure(self, structure_file: Path) -> StructureValidationResult:
        """Validate a single structure file comprehensively."""
        
        structure_name = structure_file.stem
        
        # Initialize result
        result = StructureValidationResult(
            file_path=str(structure_file),
            structure_name=structure_name,
            is_valid=True,
            overall_score=100.0,
            atom_issues=[],
            bond_issues=[],
            general_issues=[],
            recommendations=[],
            statistics={},
            grid_generation_likely_to_succeed=True
        )
        
        if not SCHRODINGER_AVAILABLE:
            result.general_issues.append("Schrodinger modules not available - limited validation")
            result.overall_score = 50.0
            result.is_valid = False
            return result
        
        try:
            # Load structure
            if structure_file.suffix.lower() in ['.mae', '.maegz']:
                structure = StructureReader.read(str(structure_file))
            elif structure_file.suffix.lower() in ['.pdb']:
                structure = StructureReader.read(str(structure_file))
            else:
                result.general_issues.append(f"Unsupported file format: {structure_file.suffix}")
                result.is_valid = False
                result.overall_score = 0.0
                return result
            
            self.logger.info(f"  Structure loaded: {structure.atom_total} atoms, {len(list(structure.residue))} residues")
            
            # Perform comprehensive validation
            self._validate_basic_properties(structure, result)
            self._validate_atom_properties(structure, result)
            self._validate_bond_orders(structure, result)
            self._validate_residue_integrity(structure, result)
            self._validate_preparation_state(structure, result)
            self._validate_lewis_structure_compatibility(structure, result)
            self._validate_glide_compatibility(structure, result)
            
            # Calculate final score and validity
            self._calculate_final_score(result)
            
        except Exception as e:
            result.general_issues.append(f"Critical error during validation: {str(e)}")
            result.is_valid = False
            result.overall_score = 0.0
            result.grid_generation_likely_to_succeed = False
            
            if self.debug:
                result.general_issues.append(f"Debug trace: {traceback.format_exc()}")
        
        return result
    
    def _validate_basic_properties(self, structure, result: StructureValidationResult):
        """Validate basic structural properties."""
        
        stats = {
            'total_atoms': structure.atom_total,
            'total_residues': len(list(structure.residue)),
            'total_chains': len(list(structure.chain)),
            'total_bonds': len(list(structure.bond))  # FIX: Use len(list(structure.bond))
        }
        
        # Count elements
        element_counts = defaultdict(int)
        for atom in structure.atom:
            element_counts[atom.element] += 1
        
        stats['element_counts'] = dict(element_counts)
        
        # Check for reasonable structure size
        if stats['total_atoms'] < 100:
            result.general_issues.append("Structure seems too small (< 100 atoms)")
            result.overall_score -= 10
        
        if stats['total_atoms'] > 50000:
            result.general_issues.append("Structure is very large (> 50,000 atoms) - may cause performance issues")
            result.overall_score -= 5
        
        # Check hydrogen content
        hydrogen_count = element_counts.get('H', 0)
        hydrogen_ratio = hydrogen_count / stats['total_atoms'] if stats['total_atoms'] > 0 else 0
        
        stats['hydrogen_ratio'] = hydrogen_ratio
        
        if hydrogen_ratio < 0.3:
            result.general_issues.append(f"Low hydrogen content ({hydrogen_ratio:.1%}) - structure may not be properly prepared")
            result.overall_score -= 15
            result.recommendations.append("Re-prepare structure with proper hydrogen addition")
        elif hydrogen_ratio > 0.7:
            result.general_issues.append(f"Very high hydrogen content ({hydrogen_ratio:.1%}) - may indicate over-protonation")
            result.overall_score -= 5
        
        # Check for zero-order bonds
        zero_order_bonds = 0
        for bond in structure.bond:
            if bond.order == 0:
                zero_order_bonds += 1
        
        stats['zero_order_bonds'] = zero_order_bonds
        stats['zero_order_bond_ratio'] = zero_order_bonds / stats['total_bonds'] if stats['total_bonds'] > 0 else 0
        
        if stats['zero_order_bond_ratio'] > 0.1:
            result.general_issues.append(f"High proportion of zero-order bonds ({stats['zero_order_bond_ratio']:.1%})")
            result.overall_score -= 20
            result.recommendations.append("Check bond orders - may need structure preparation/minimization")
        
        result.statistics.update(stats)
    
    def _validate_atom_properties(self, structure, result: StructureValidationResult):
        """Validate individual atom properties for potential issues."""
        
        critical_atoms = []
        problematic_atoms = []
        
        for atom in structure.atom:
            issues = []
            
            # Check for problematic elements
            if atom.element in self.problematic_elements:
                issues.append(f"Contains potentially problematic element: {atom.element}")
                
            # Check formal charges (if available)
            try:
                if hasattr(atom, 'formal_charge') and abs(atom.formal_charge) > 2:
                    issues.append(f"High formal charge: {atom.formal_charge}")
            except:
                pass
            
            # Check partial charges (if available)
            try:
                if hasattr(atom, 'partial_charge') and abs(atom.partial_charge) > 3.0:
                    issues.append(f"Extreme partial charge: {atom.partial_charge:.2f}")
            except:
                pass
            
            # Check connectivity
            atom_bonds = list(atom.bond)
            connected_atoms = len(atom_bonds)
            expected_connectivity = self._get_expected_connectivity(atom.element)
            
            if connected_atoms > expected_connectivity + 2:
                issues.append(f"Over-connected atom: {connected_atoms} bonds (expected ~{expected_connectivity})")
            elif connected_atoms == 0 and atom.element not in ['NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE']:
                issues.append("Isolated atom (no bonds)")
            
            # IMPROVED: Better histidine hydrogen detection
            if atom.element == 'N' and connected_atoms < expected_connectivity:
                residue = atom.getResidue()
                res_name = getattr(residue, 'pdbres', 'UNK').strip()
                atom_name = getattr(atom, 'pdbname', f"{atom.element}{atom.index}").strip()
                
                # Skip histidine nitrogens - they have complex protonation patterns
                if res_name in ['HIS', 'HIE', 'HID', 'HIP'] and atom_name in ['ND1', 'NE2']:
                    # Histidines can have different protonation states - don't flag as missing H
                    pass
                else:
                    # Count actual hydrogen bonds
                    h_count = 0
                    for bond in atom_bonds:
                        bonded_atom = bond.atom1 if bond.atom2 == atom else bond.atom2
                        if bonded_atom.element == 'H':
                            h_count += 1
                    
                    # Only flag if it's clearly missing hydrogens (not histidine edge cases)
                    if h_count == 0 and atom.element == 'N' and res_name not in ['HIS', 'HIE', 'HID', 'HIP']:
                        # Also check if it's backbone nitrogen (which may not need extra H)
                        if atom_name not in ['N']:  # Backbone N is usually fine
                            issues.append("Nitrogen atom without hydrogens - may need protonation")
            
            # Record issues
            if issues:
                residue = atom.getResidue()
                atom_issue = AtomIssue(
                    atom_index=atom.index,
                    atom_name=getattr(atom, 'pdbname', f"{atom.element}{atom.index}").strip(),
                    residue=getattr(residue, 'pdbres', 'UNK').strip(),
                    chain=residue.chain,
                    residue_number=residue.resnum,
                    element=atom.element,
                    issue_type="atom_property",
                    description="; ".join(issues),
                    severity="warning" if len(issues) == 1 else "critical",
                    suggestion=self._get_atom_fix_suggestion(atom, issues)
                )
                
                result.atom_issues.append(atom_issue)
                
                if atom_issue.severity == "critical":
                    critical_atoms.append(atom.index)
                    result.overall_score -= 5
                else:
                    problematic_atoms.append(atom.index)
                    result.overall_score -= 1  # REDUCED penalty for warnings
        
        # Update statistics
        result.statistics['critical_atoms'] = len(critical_atoms)
        result.statistics['problematic_atoms'] = len(problematic_atoms)
        
        if critical_atoms:
            result.recommendations.append(f"Fix {len(critical_atoms)} critical atom issues before grid generation")
        
        if len(critical_atoms) > 10:
            result.grid_generation_likely_to_succeed = False
    
    def _validate_bond_orders(self, structure, result: StructureValidationResult):
        """Validate bond orders for Lewis structure compatibility."""
        
        suspicious_bonds = []
        fractional_bonds = []
        
        for bond in structure.bond:
            atom1 = bond.atom1
            atom2 = bond.atom2
            
            # Check for fractional bond orders
            if bond.order != int(bond.order) and bond.order != 0:
                fractional_bonds.append(bond)
            
            # Check for unrealistic bond orders
            max_reasonable_order = self._get_max_bond_order(atom1.element, atom2.element)
            
            if bond.order > max_reasonable_order:
                residue1 = atom1.getResidue()
                residue2 = atom2.getResidue()
                
                bond_issue = BondIssue(
                    atom1_index=atom1.index,
                    atom2_index=atom2.index,
                    atom1_name=getattr(atom1, 'pdbname', f"{atom1.element}{atom1.index}").strip(),  # FIX: Use getattr
                    atom2_name=getattr(atom2, 'pdbname', f"{atom2.element}{atom2.index}").strip(),  # FIX: Use getattr
                    residue1=getattr(residue1, 'pdbres', 'UNK').strip(),  # FIX: Use getattr
                    residue2=getattr(residue2, 'pdbres', 'UNK').strip(),  # FIX: Use getattr
                    bond_order=bond.order,
                    expected_order=max_reasonable_order,
                    issue_type="unrealistic_bond_order",
                    description=f"Bond order {bond.order} exceeds reasonable maximum {max_reasonable_order}",
                    severity="critical",
                    suggestion="Reset bond order to reasonable value and re-minimize structure"
                )
                
                result.bond_issues.append(bond_issue)
                suspicious_bonds.append(bond)
                result.overall_score -= 10
        
        # Handle fractional bond orders
        if fractional_bonds:
            result.general_issues.append(f"Found {len(fractional_bonds)} fractional bond orders")
            result.overall_score -= len(fractional_bonds) * 2
            result.recommendations.append("Convert fractional bond orders to integer values")
        
        result.statistics['suspicious_bonds'] = len(suspicious_bonds)
        result.statistics['fractional_bonds'] = len(fractional_bonds)
        
        if len(suspicious_bonds) > 5:
            result.grid_generation_likely_to_succeed = False
    
    def _validate_residue_integrity(self, structure, result: StructureValidationResult):
        """Validate residue-level integrity with less harsh penalties."""
        
        problematic_residues = []
        
        for residue in structure.residue:
            res_name = getattr(residue, 'pdbres', 'UNK').strip()
            
            # Check for known problematic residues - but be less harsh
            if res_name in self.problematic_residues:
                if res_name == 'TPO':  # Phosphorylated threonine is common and usually fine
                    result.general_issues.append(f"Contains phosphorylated residue: {res_name} at {residue.resnum}:{residue.chain} (may be fine)")
                    result.overall_score -= 2  # Reduced penalty
                else:
                    result.general_issues.append(f"Contains problematic residue: {res_name} at {residue.resnum}:{residue.chain}")
                    result.overall_score -= 5
                    
                problematic_residues.append(res_name)
                
                if res_name == 'MSE':
                    result.recommendations.append(f"Consider converting MSE (selenomethionine) to MET at position {residue.resnum}:{residue.chain}")
            
            # Check residue completeness
            atoms_in_residue = list(residue.atom)
            if len(atoms_in_residue) < 3 and res_name not in ['HOH', 'WAT', 'NA', 'CL', 'MG', 'CA', 'ZN']:
                result.general_issues.append(f"Incomplete residue: {res_name} at {residue.resnum}:{residue.chain} has only {len(atoms_in_residue)} atoms")
                result.overall_score -= 3
        
        result.statistics['problematic_residue_types'] = list(set(problematic_residues))
    
    def _validate_preparation_state(self, structure, result: StructureValidationResult):
        """Validate the preparation state of the structure."""
        
        # Check for preparation properties
        prep_properties = []
        all_properties = list(structure.property.keys()) if hasattr(structure, 'property') else []
        
        preparation_indicators = [
            'b_ppw_prepared', 's_ppw_prepared', 's_ppw_prepared_with_version',
            's_ppw_het_states', 'r_ffld_Potential_Energy', 'r_psp_Potential_Energy'
        ]
        
        for prop in preparation_indicators:
            if prop in all_properties:
                prep_properties.append(prop)
        
        result.statistics['preparation_properties'] = prep_properties
        
        if not prep_properties:
            result.general_issues.append("No preparation properties found - structure may not be prepared")
            result.overall_score -= 20
            result.recommendations.append("Prepare structure using Protein Preparation Wizard")
        
        # Check for minimization
        energy_properties = [prop for prop in all_properties if 'Energy' in prop]
        if not energy_properties:
            result.general_issues.append("No energy properties found - structure may not be minimized")
            result.overall_score -= 10
            result.recommendations.append("Minimize structure after preparation")
    
    def _validate_lewis_structure_compatibility(self, structure, result: StructureValidationResult):
        """Validate compatibility with Lewis structure algorithms."""
        
        # This is the most critical validation for the mmlewis errors you're seeing
        lewis_issues = []
        
        for atom in structure.atom:
            # Check valence consistency
            element = atom.element
            connected_bonds = list(atom.bond)
            
            # Calculate total bond order
            total_bond_order = sum(bond.order for bond in connected_bonds)
            
            # Get expected valence
            expected_valences = self._get_possible_valences(element)
            
            if expected_valences and total_bond_order not in expected_valences:
                # Check if this could be fixed by adjusting bond orders
                if abs(total_bond_order - min(expected_valences)) > 2:
                    residue = atom.getResidue()
                    
                    lewis_issue = AtomIssue(
                        atom_index=atom.index,
                        atom_name=getattr(atom, 'pdbname', f"{atom.element}{atom.index}").strip(),  # FIX: Use getattr
                        residue=getattr(residue, 'pdbres', 'UNK').strip(),  # FIX: Use getattr
                        chain=residue.chain,
                        residue_number=residue.resnum,
                        element=element,
                        issue_type="lewis_structure",
                        description=f"Valence {total_bond_order} inconsistent with expected {expected_valences}",
                        severity="critical",
                        suggestion=f"Adjust bond orders to achieve valence in {expected_valences}"
                    )
                    
                    result.atom_issues.append(lewis_issue)
                    lewis_issues.append(atom.index)
                    result.overall_score -= 8
        
        # Check for problematic functional groups
        self._check_problematic_functional_groups(structure, result)
        
        result.statistics['lewis_structure_issues'] = len(lewis_issues)
        
        if lewis_issues:
            result.recommendations.append(f"Fix {len(lewis_issues)} Lewis structure issues - these will cause mmlewis errors in Glide")
            
        if len(lewis_issues) > 3:
            result.grid_generation_likely_to_succeed = False
            result.general_issues.append("CRITICAL: Multiple Lewis structure issues detected - grid generation will likely fail")
    
    def _validate_glide_compatibility(self, structure, result: StructureValidationResult):
        """Validate compatibility with Glide grid generation."""
        
        issues = []
        
        # Find disconnected fragments
        fragments = self._find_disconnected_fragments(structure)
        
        # IMPROVED: More realistic fragment analysis
        large_fragments = [f for f in fragments if len(f) > 50]  # Only count significant fragments
        
        if len(large_fragments) > 3:  # Allow for 2 chains + ligand
            issues.append(f"Multiple large disconnected fragments: {len(large_fragments)}")
            # But don't penalize heavily - this is often normal
            result.overall_score -= 2  # Reduced penalty
        
        # Only worry about fragmentation if it's extreme
        if len(fragments) > 20:  # Many tiny fragments
            issues.append(f"Excessive fragmentation: {len(fragments)} total fragments")
            result.overall_score -= 5
            result.recommendations.append("Consider removing isolated atoms/residues")
        
        # Check for missing crucial protein parts (very conservative)
        total_protein_atoms = sum(len(f) for f in large_fragments)
        if total_protein_atoms < 1000:  # Very small protein
            issues.append("Structure may be too small or highly fragmented")
            result.overall_score -= 10
        
        result.statistics['glide_compatibility_issues'] = issues
    
    def _check_problematic_functional_groups(self, structure, result: StructureValidationResult):
        """Check for functional groups that commonly cause Lewis structure issues."""
        
        # Define problematic patterns
        problematic_patterns = [
            # Sulfur compounds
            ('SO4', ['S', 'O', 'O', 'O', 'O']),
            ('SO3', ['S', 'O', 'O', 'O']),
            # Phosphate compounds
            ('PO4', ['P', 'O', 'O', 'O', 'O']),
            # Metal coordination
            ('METAL_COORD', ['ZN', 'FE', 'MG', 'CA', 'MN']),
        ]
        
        functional_group_issues = []
        
        for residue in structure.residue:
            res_name = residue.pdbres.strip() if hasattr(residue, 'pdbres') else 'UNK'
            atoms_in_residue = list(residue.atom)
            
            # Check for sulfur-containing residues with potential issues
            if any(atom.element == 'S' for atom in atoms_in_residue):
                sulfur_atoms = [atom for atom in atoms_in_residue if atom.element == 'S']
                for s_atom in sulfur_atoms:
                    # Check sulfur valence
                    s_bonds = list(s_atom.bond)
                    total_s_order = sum(bond.order for bond in s_bonds)
                    
                    if total_s_order > 6:  # Sulfur can have up to 6 bonds
                        functional_group_issues.append(f"Sulfur over-valence in {res_name}:{residue.resnum}:{residue.chain}")
            
            # Check for phosphorus issues
            if any(atom.element == 'P' for atom in atoms_in_residue):
                phosphorus_atoms = [atom for atom in atoms_in_residue if atom.element == 'P']
                for p_atom in phosphorus_atoms:
                    p_bonds = list(p_atom.bond)
                    total_p_order = sum(bond.order for bond in p_bonds)
                    
                    if total_p_order > 5:  # Phosphorus can have up to 5 bonds
                        functional_group_issues.append(f"Phosphorus over-valence in {res_name}:{residue.resnum}:{residue.chain}")
        
        if functional_group_issues:
            result.general_issues.extend(functional_group_issues)
            result.overall_score -= len(functional_group_issues) * 5
            result.recommendations.append("Review functional groups with over-valence atoms")
    
    def _calculate_final_score(self, result: StructureValidationResult):
        """Calculate final validation score and determine validity."""
        
        # Ensure score doesn't go below 0
        result.overall_score = max(0.0, result.overall_score)
        
        # Determine validity based on score and critical issues
        critical_issues = len([issue for issue in result.atom_issues if issue.severity == "critical"])
        critical_issues += len([issue for issue in result.bond_issues if issue.severity == "critical"])
        
        if result.overall_score < 30:
            result.is_valid = False
            result.grid_generation_likely_to_succeed = False
        elif critical_issues > 5:
            result.is_valid = False
            result.grid_generation_likely_to_succeed = False
        elif result.overall_score < 60:
            result.is_valid = False  # Needs work but might succeed
        else:
            result.is_valid = True
        
        # Grid generation prediction
        if result.overall_score < 50 or critical_issues > 2:
            result.grid_generation_likely_to_succeed = False
        
        # Add final recommendations
        if not result.is_valid:
            if result.overall_score < 30:
                result.recommendations.append("CRITICAL: Structure needs complete re-preparation")
            elif critical_issues > 0:
                result.recommendations.append("Address critical issues before attempting grid generation")
            else:
                result.recommendations.append("Structure needs improvement but may work with modifications")
    
    # Helper methods
    def _get_expected_connectivity(self, element: str) -> int:
        """Get expected connectivity for an element."""
        connectivity_map = {
            'H': 1, 'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3,
            'F': 1, 'CL': 1, 'BR': 1, 'I': 1,
            'NA': 0, 'MG': 0, 'CA': 0, 'ZN': 0, 'FE': 0
        }
        return connectivity_map.get(element.upper(), 4)
    
    def _get_max_bond_order(self, element1: str, element2: str) -> float:
        """Get maximum reasonable bond order between two elements."""
        # Simplified rules
        if 'H' in [element1, element2]:
            return 1.0
        if element1 == 'C' and element2 == 'C':
            return 3.0
        if element1 == 'C' and element2 in ['N', 'O']:
            return 3.0
        if element1 == 'N' and element2 == 'N':
            return 3.0
        return 2.0
    
    def _get_possible_valences(self, element: str) -> List[int]:
        """Get possible valences for an element."""
        valence_map = {
            'H': [1],
            'C': [4],
            'N': [3, 4, 5],  # Can be protonated or have different oxidation states
            'O': [2],
            'S': [2, 4, 6],
            'P': [3, 5],
            'F': [1],
            'CL': [1],
            'BR': [1],
            'I': [1]
        }
        return valence_map.get(element.upper(), [])
    
    def _get_atom_fix_suggestion(self, atom, issues: List[str]) -> str:
        """Generate fix suggestion for atom issues."""
        suggestions = []
        
        for issue in issues:
            if "formal charge" in issue:
                suggestions.append("Check formal charge assignment")
            elif "over-connected" in issue:
                suggestions.append("Remove extra bonds or check connectivity")
            elif "isolated" in issue:
                suggestions.append("Check if atom should be connected or removed")
            elif "hydrogens" in issue:
                suggestions.append("Add missing hydrogens")
            elif "problematic element" in issue:
                suggestions.append("Consider replacing with more common element")
        
        return "; ".join(suggestions) if suggestions else "Manual review required"
    
    def _find_disconnected_fragments(self, structure) -> List[List[int]]:
        """Find disconnected molecular fragments with better classification."""
        visited = set()
        fragments = []
        
        def dfs(atom_index, current_fragment):
            if atom_index in visited:
                return
            visited.add(atom_index)
            current_fragment.append(atom_index)
            
            atom = structure.atom[atom_index]
            for bond in atom.bond:
                neighbor = bond.atom1 if bond.atom2 == atom else bond.atom2
                if neighbor.index not in visited:
                    dfs(neighbor.index, current_fragment)
        
        for atom in structure.atom:
            if atom.index not in visited:
                fragment = []
                dfs(atom.index, fragment)
                fragments.append(fragment)
        
        # IMPROVED: Better fragment classification with context
        protein_fragments = []
        ligand_fragments = []
        ion_fragments = []
        water_fragments = []
        small_fragments = []
        
        for fragment in fragments:
            # Analyze fragment composition
            fragment_residues = set()
            protein_residues = 0
            
            for atom_idx in fragment:
                atom = structure.atom[atom_idx]
                residue = atom.getResidue()
                res_name = getattr(residue, 'pdbres', 'UNK').strip()
                fragment_residues.add(res_name)
                
                # Count protein residues
                if res_name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                               'HIS', 'HIE', 'HID', 'HIP', 'ILE', 'LEU', 'LYS', 'MET', 
                               'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'TPO', 'SEP']:
                    protein_residues += 1
            
            # Classify fragment based on size and composition
            if len(fragment) > 100 or protein_residues > 10:  # Large protein fragment
                protein_fragments.append(fragment)
            elif any(res in fragment_residues for res in ['HOH', 'WAT', 'H2O']):
                water_fragments.append(fragment)
            elif any(res in fragment_residues for res in ['NA', 'CL', 'MG', 'CA', 'ZN', 'FE', 'MN']):
                ion_fragments.append(fragment)
            elif len(fragment) < 5:  # Very small fragments (probably artifacts)
                small_fragments.append(fragment)
            else:
                ligand_fragments.append(fragment)

        # Log fragment info for context (but don't over-penalize)
        self.logger.debug(f"Fragment analysis: {len(protein_fragments)} protein, "
                         f"{len(ligand_fragments)} ligand, {len(water_fragments)} water, "
                         f"{len(ion_fragments)} ion, {len(small_fragments)} small")

        # Return only significant fragments for counting
        significant_fragments = protein_fragments + ligand_fragments
        
        return significant_fragments
    
    def _calculate_final_score(self, result: StructureValidationResult):
        """Calculate final validation score with more realistic thresholds."""
        
        # Ensure score doesn't go below 0
        result.overall_score = max(0.0, result.overall_score)
        
        # Count only critical issues for validity
        critical_issues = len([issue for issue in result.atom_issues if issue.severity == "critical"])
        critical_issues += len([issue for issue in result.bond_issues if issue.severity == "critical"])
        
        # More lenient thresholds
        if result.overall_score < 10:  # Only extremely broken structures
            result.is_valid = False
            result.grid_generation_likely_to_succeed = False
        elif critical_issues > 10:  # Many critical issues
            result.is_valid = False
            result.grid_generation_likely_to_succeed = False
        elif result.overall_score < 40:  # Moderate issues
            result.is_valid = False  # Needs work but might succeed
            result.grid_generation_likely_to_succeed = True  # Still might work
        else:
            result.is_valid = True
        
        # Grid generation prediction - be more optimistic
        if result.overall_score < 20 or critical_issues > 5:
            result.grid_generation_likely_to_succeed = False
        else:
            result.grid_generation_likely_to_succeed = True
        
        # Add final recommendations
        if not result.is_valid:
            if result.overall_score < 20:
                result.recommendations.append("CRITICAL: Structure needs significant work")
            elif critical_issues > 0:
                result.recommendations.append("Address critical issues before attempting grid generation")
            else:
                result.recommendations.append("Structure has minor issues but should work for grid generation")    
    
    def _generate_reports(self):
        """Generate comprehensive validation reports."""
        
        self.logger.info("Generating validation reports...")
        
        # Summary report
        self._generate_summary_report()
        
        # Detailed individual reports
        self._generate_individual_reports()
        
        # Problematic structures report
        self._generate_problematic_structures_report()
        
        # Recommendations report
        self._generate_recommendations_report()
    
    def _generate_summary_report(self):
        """Generate summary CSV report."""
        
        summary_data = []
        
        for name, result in self.validation_results.items():
            row = {
                'structure_name': name,
                'file_path': result.file_path,
                'is_valid': result.is_valid,
                'overall_score': result.overall_score,
                'grid_generation_likely': result.grid_generation_likely_to_succeed,
                'critical_atom_issues': len([issue for issue in result.atom_issues if issue.severity == "critical"]),
                'total_atom_issues': len(result.atom_issues),
                'critical_bond_issues': len([issue for issue in result.bond_issues if issue.severity == "critical"]),
                'total_bond_issues': len(result.bond_issues),
                'general_issues_count': len(result.general_issues),
                'recommendations_count': len(result.recommendations),
                'lewis_structure_issues': result.statistics.get('lewis_structure_issues', 0),
                'hydrogen_ratio': result.statistics.get('hydrogen_ratio', 0),
                'zero_order_bond_ratio': result.statistics.get('zero_order_bond_ratio', 0)
            }
            summary_data.append(row)
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.reports_dir / "validation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Summary report saved to {summary_file}")
    
    def _generate_individual_reports(self):
        """Generate detailed reports for each structure."""
        
        individual_dir = self.reports_dir / "individual_structures"
        individual_dir.mkdir(exist_ok=True)
        
        for name, result in self.validation_results.items():
            report_file = individual_dir / f"{name}_validation_report.json"
            
            # Convert to serializable format
            report_data = asdict(result)
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
    
    def _generate_problematic_structures_report(self):
        """Generate focused report on structures with issues."""
        
        problematic = [result for result in self.validation_results.values() 
                      if not result.is_valid or not result.grid_generation_likely_to_succeed]
        
        if not problematic:
            self.logger.info("No problematic structures found!")
            return
        
        report_file = self.reports_dir / "problematic_structures_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("PROBLEMATIC STRUCTURES REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Found {len(problematic)} structures with issues:\n\n")
            
            # Sort by severity (lowest score first)
            problematic.sort(key=lambda x: x.overall_score)
            
            for result in problematic:
                f.write(f"Structure: {result.structure_name}\n")
                f.write(f"Score: {result.overall_score:.1f}/100\n")
                f.write(f"Grid generation likely: {result.grid_generation_likely_to_succeed}\n")
                f.write(f"File: {result.file_path}\n\n")
                
                if result.general_issues:
                    f.write("General Issues:\n")
                    for issue in result.general_issues:
                        f.write(f"  - {issue}\n")
                    f.write("\n")
                
                if result.atom_issues:
                    critical_atom_issues = [issue for issue in result.atom_issues if issue.severity == "critical"]
                    if critical_atom_issues:
                        f.write("Critical Atom Issues:\n")
                        for issue in critical_atom_issues[:5]:  # Show first 5
                            f.write(f"  - {issue.residue}:{issue.residue_number}:{issue.chain} "
                                   f"{issue.atom_name} ({issue.element}): {issue.description}\n")
                        if len(critical_atom_issues) > 5:
                            f.write(f"  ... and {len(critical_atom_issues) - 5} more\n")
                        f.write("\n")
                
                if result.recommendations:
                    f.write("Recommendations:\n")
                    for rec in result.recommendations:
                        f.write(f"  - {rec}\n")
                    f.write("\n")
                
                f.write("-" * 40 + "\n\n")
        
        self.logger.info(f"Problematic structures report saved to {report_file}")
    
    def _generate_recommendations_report(self):
        """Generate actionable recommendations report."""
        
        # Collect all recommendations
        all_recommendations = defaultdict(list)
        
        for result in self.validation_results.values():
            for rec in result.recommendations:
                all_recommendations[rec].append(result.structure_name)
        
        report_file = self.reports_dir / "actionable_recommendations.txt"
        
        with open(report_file, 'w') as f:
            f.write("ACTIONABLE RECOMMENDATIONS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Sort by frequency
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: len(x[1]), reverse=True)
            
            for recommendation, structures in sorted_recs:
                f.write(f"Recommendation: {recommendation}\n")
                f.write(f"Affects {len(structures)} structure(s): {', '.join(structures[:5])}")
                if len(structures) > 5:
                    f.write(f" and {len(structures) - 5} more")
                f.write("\n\n")
        
        self.logger.info(f"Recommendations report saved to {report_file}")

    def _generate_final_recommendations(self, result: StructureValidationResult):
        """Generate final recommendations based on all validation results."""
        
        # Add context about fragmentations
        if any("disconnected fragments" in issue for issue in result.general_issues):
            result.recommendations.insert(0, 
                "NOTE: Disconnected fragments are normal for crystal structures with missing residues "
                "and do not prevent grid generation")
        
        # Existing recommendation logic...
        if result.overall_score < 50:
            result.recommendations.append("Consider re-preparing structure")
        elif len(result.atom_issues) > 10:
            result.recommendations.append("Review atom-level issues for accuracy")
    
    def _print_summary(self):
        """Print validation summary to console."""
        
        total = len(self.validation_results)
        valid = len([r for r in self.validation_results.values() if r.is_valid])
        grid_ready = len([r for r in self.validation_results.values() if r.grid_generation_likely_to_succeed])
        
        print("\n" + "=" * 70)
        print("PROTEIN STRUCTURE VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total structures validated: {total}")
        print(f"Valid structures: {valid} ({valid/total*100:.1f}%)")
        print(f"Grid-generation ready: {grid_ready} ({grid_ready/total*100:.1f}%)")
        print(f"Structures needing work: {total - valid}")
        print(f"Structures likely to fail grid generation: {total - grid_ready}")
        
        if total > 0:
            avg_score = sum(r.overall_score for r in self.validation_results.values()) / total
            print(f"Average validation score: {avg_score:.1f}/100")
        
        print(f"\nReports saved to: {self.reports_dir}")
        print("=" * 70)
        
        # Show top issues
        if total - grid_ready > 0:
            print("\nTOP ISSUES TO ADDRESS:")
            
            # Collect most common issues
            issue_counts = defaultdict(int)
            for result in self.validation_results.values():
                if not result.grid_generation_likely_to_succeed:
                    for issue in result.general_issues:
                        issue_counts[issue] += 1
            
            # Show top 5 issues
            top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for issue, count in top_issues:
                print(f"  - {issue} ({count} structures)")
        
        print()

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Validate protein structures for preparation issues before grid generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python protein_structure_validator.py proteins/ validation_output/
    python protein_structure_validator.py --debug proteins/ validation_output/
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Directory containing protein structure files (.mae, .maegz, .pdb)'
    )
    
    parser.add_argument(
        'output_dir',
        help='Directory to save validation reports'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        validator = ProteinStructureValidator(
            output_dir=str(output_dir),
            debug=args.debug
        )
        
        validator.validate_structures(str(input_dir))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
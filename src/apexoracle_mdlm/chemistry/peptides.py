"""Recognize supported linear and head-to-tail cyclic peptide structures."""

from __future__ import annotations

import re
from typing import Optional

import selfies
from rdkit import Chem


def _flip_stereo(smarts: str) -> str:
    if "@" not in smarts:
        return smarts
    placeholder = smarts.replace("@@", "§§")
    return placeholder.replace("@", "@@").replace("§§", "@")


PEPTIDE_BOND_PATTERN = Chem.MolFromSmarts(
    "[C;X3,X4:1](=O)[N;!a;X2,X3,X4:2][C;X4;H1,H2:3]"
)
CA_PATTERN = Chem.MolFromSmarts("[N][C;X4:1][C](=O)")

AA_INNER_L = {
    "A": "[*:1]N[C@@H](C)C(=O)[*:2]",
    "R": "[*:1]N[C@@H](CCCNC(N)=N)C(=O)[*:2]",
    "B": "[*:1]N[C@@H](CCCN=C(N)N)C(=O)[*:2]",
    "N": "[*:1]N[C@@H](CC(=O)N)C(=O)[*:2]",
    "D": "[*:1]N[C@@H](CC(=O)O)C(=O)[*:2]",
    "C": "[*:1]N[C@@H](CS)C(=O)[*:2]",
    "E": "[*:1]N[C@@H](CCC(=O)O)C(=O)[*:2]",
    "Q": "[*:1]N[C@@H](CCC(=O)N)C(=O)[*:2]",
    "G": "[*:1]NC(C(=O)[*:2])",
    "H": "[*:1]N[C@@H](Cc1c[nH]cn1)C(=O)[*:2]",
    "L": "[*:1]N[C@@H](CC(C)C)C(=O)[*:2]",
    "I": "[*:1]N[C@@H](C(C)CC)C(=O)[*:2]",
    "K": "[*:1]N[C@@H](CCCCN)C(=O)[*:2]",
    "M": "[*:1]N[C@@H](CCSC)C(=O)[*:2]",
    "F": "[*:1]N[C@@H](Cc1ccccc1)C(=O)[*:2]",
    "P": "[*:1]N1CCC[C@@H]1C(=O)[*:2]",
    "S": "[*:1]N[C@@H](CO)C(=O)[*:2]",
    "T": "[*:1]N[C@@H](C(O)C)C(=O)[*:2]",
    "W": "[*:1]N[C@@H](Cc1c[nH]c2ccccc12)C(=O)[*:2]",
    "Y": "[*:1]N[C@@H](Cc1ccc(O)cc1)C(=O)[*:2]",
    "V": "[*:1]N[C@@H](C(C)C)C(=O)[*:2]",
}


def _make_nterm(smarts: str) -> str:
    return re.sub(r"\[\*\:1\]N", "[N,N+;H1,H2]", smarts, count=1)


def _make_cterm(smarts: str) -> str:
    return smarts.replace("[*:2]", "[O;H1,H0-]")


def _make_d_patterns(source: dict[str, str]) -> dict[str, str]:
    return {
        amino_acid.lower(): _flip_stereo(smarts)
        for amino_acid, smarts in source.items()
    }


def _compile_patterns(source: dict[str, str]):
    return [
        (amino_acid, pattern, pattern.GetNumAtoms())
        for amino_acid, smarts in source.items()
        if (pattern := Chem.MolFromSmarts(smarts)) is not None
    ]


AA_NTERM_L = {
    amino_acid: _make_nterm(smarts) for amino_acid, smarts in AA_INNER_L.items()
}
AA_NTERM_L["G"] = "[N,N+;H1,H2]C(C(=O)[*:2])"
AA_CTERM_L = {
    amino_acid: _make_cterm(smarts) for amino_acid, smarts in AA_INNER_L.items()
}
AA_CTERM_L["G"] = "[*:1]NC(C(=O)[O;H1,H0-])"
RESIDUE_PATTERNS = [
    _compile_patterns(AA_INNER_L),
    _compile_patterns(_make_d_patterns(AA_INNER_L)),
    _compile_patterns(AA_NTERM_L),
    _compile_patterns(_make_d_patterns(AA_NTERM_L)),
    _compile_patterns(AA_CTERM_L),
    _compile_patterns(_make_d_patterns(AA_CTERM_L)),
]


def _peptide_bonds(molecule: Chem.Mol) -> list[tuple[int, int, int]]:
    result = []
    for carbon, _, nitrogen, _ in molecule.GetSubstructMatches(
        PEPTIDE_BOND_PATTERN, useChirality=False
    ):
        bond = molecule.GetBondBetweenAtoms(carbon, nitrogen)
        if bond and bond.GetBondType() == Chem.BondType.SINGLE:
            result.append((bond.GetIdx(), carbon, nitrogen))
    return result


def _find_ca_index(fragment: Chem.Mol) -> Optional[int]:
    matches = fragment.GetSubstructMatches(CA_PATTERN, useChirality=False)
    return matches[0][1] if matches else None


def _ca_chirality(fragment: Chem.Mol, ca_index: int, amino_acid: str) -> Optional[str]:
    Chem.AssignStereochemistry(fragment, cleanIt=True, force=True)
    atom = fragment.GetAtomWithIdx(ca_index)
    if not atom.HasProp("_CIPCode"):
        return None
    cip = atom.GetProp("_CIPCode").upper()
    if amino_acid in {"C", "U"}:
        cip = "S" if cip == "R" else "R"
    return "L" if cip == "S" else "D"


def _fragment_to_amino_acid(fragment: Chem.Mol) -> str:
    amino_acid = None
    for pattern_group in RESIDUE_PATTERNS:
        for name, pattern, atom_count in sorted(
            pattern_group, key=lambda item: -item[2]
        ):
            if fragment.GetNumAtoms() == atom_count and fragment.HasSubstructMatch(
                pattern, useChirality=False
            ):
                amino_acid = name.upper()
                break
        if amino_acid:
            break
    if amino_acid is None:
        return "X"
    ca_index = _find_ca_index(fragment)
    if ca_index is None:
        return amino_acid
    chirality = _ca_chirality(fragment, ca_index, amino_acid)
    return amino_acid.lower() if chirality == "D" else amino_acid


def smiles_to_peptide_sequence(structure: str) -> tuple[str, Optional[str]]:
    """Return canonical SMILES and a case-sensitive peptide sequence.

    The parser accepts SMILES or SELFIES. Lower-case residue letters denote D
    stereochemistry and head-to-tail cyclic sequences use a ``cyclo-`` prefix.
    Unsupported, branched, fragmented, or unparsable structures return a
    ``None`` sequence without raising.
    """

    smiles = structure
    if "[C]" in smiles:
        try:
            smiles = selfies.decoder(smiles)
        except Exception:
            return structure, None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles, None
    peptide_bonds = list(set(_peptide_bonds(molecule)))
    if not peptide_bonds:
        return smiles, _fragment_to_amino_acid(molecule)

    fragmented = Chem.FragmentOnBonds(
        molecule, {bond[0] for bond in peptide_bonds}, addDummies=True
    )
    fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
    atom_groups = Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False)
    atom_to_fragment = {
        atom: fragment_index
        for fragment_index, atoms in enumerate(atom_groups)
        for atom in atoms
    }
    successors: dict[int, int] = {}
    indegree: dict[int, int] = {}
    for _, carbon, nitrogen in peptide_bonds:
        carbon_fragment = atom_to_fragment[carbon]
        nitrogen_fragment = atom_to_fragment[nitrogen]
        successors[carbon_fragment] = nitrogen_fragment
        indegree[nitrogen_fragment] = indegree.get(nitrogen_fragment, 0) + 1

    starts = [index for index in range(len(fragments)) if indegree.get(index, 0) == 0]
    last_to_first = None
    if starts:
        start = starts[0]
        cyclic = False
    else:
        cyclic = True
        if len(indegree) != len(fragments) or any(
            count != 1 for count in indegree.values()
        ):
            return smiles, None
        start = min(range(len(fragments)), key=lambda index: min(atom_groups[index]))
        for _, carbon, nitrogen in peptide_bonds:
            carbon_fragment = atom_to_fragment[carbon]
            nitrogen_fragment = atom_to_fragment[nitrogen]
            if nitrogen_fragment == start:
                last_to_first = carbon_fragment
                break
        if last_to_first is not None:
            successors[last_to_first] = start

    ordered = []
    visited = set()
    current = start
    while current not in visited:
        ordered.append(current)
        visited.add(current)
        current = successors.get(current)
        if current is None:
            break
    if cyclic and len(ordered) == len(fragments) - 1 and last_to_first is not None:
        if last_to_first not in visited:
            ordered.append(last_to_first)
    if len(ordered) != len(fragments):
        return smiles, None

    sequence = "".join(_fragment_to_amino_acid(fragments[index]) for index in ordered)
    if cyclic:
        sequence = f"cyclo-{sequence}"
    return Chem.MolToSmiles(molecule), sequence

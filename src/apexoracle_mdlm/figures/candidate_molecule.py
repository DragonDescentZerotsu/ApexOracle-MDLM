"""Reusable annotated 2D molecule rendering for screened candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_annotated_candidate(
    smiles: str,
    *,
    predicted_mic_umol: float,
    peptide_sequence: str,
    size: tuple[int, int] = (1500, 1500),
    font_size: int = 48,
) -> Any:
    """Return a PIL image with the historical MIC/sequence annotation."""

    import PIL
    from PIL import ImageDraw, ImageFont
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import MolDrawOptions

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Could not parse candidate SMILES.")
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(molecule)
    options = MolDrawOptions()
    options.reduceOverlap = True
    image = Draw.MolToImage(molecule, size=size, options=options)

    bundled_font = Path(PIL.__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(bundled_font), font_size)
    except OSError:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    text = f"MIC: {predicted_mic_umol:.2f}\nSeq: {peptide_sequence}"
    x, y = 20, 20
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [x - 5, y - 5, x + (right - left) + 5, y + (bottom - top) + 5],
        fill=(255, 255, 255, 200),
    )
    draw.text((x, y), text, fill="black", font=font)
    return image

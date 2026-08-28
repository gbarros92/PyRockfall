import os
import argparse
import numpy as np

from typing import List, Tuple, Dict, Optional

import pyrockfall as pr
from pyrockfall.scripts._data import material_lib_path
from pyrockfall.scripts._style import get_material_colors


DESCRIPTION = "Create  input files for rockfall simulations from profiles."
HELP = "Create input files for rockfall simulations from profiles with given initial height from the floor, height increment between seeders, and initial drop height to initiate fall."


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("filename", help="Input profiles file (.xyz with material and profile attributes).")
    ap.add_argument("-m", "--material-lib", default=None, help="Material library file. If omitted, uses default.")
    ap.add_argument("-n", "--material-name", default="Material", help="Attribute name for material IDs in profiles PLY.")
    ap.add_argument("-p", "--profile-name", default="Profile", help="Attribute name for profile IDs in profiles PLY.")
    ap.add_argument("-H", "--height-start", type=float, default=1.0, help="Starting height for simulations.")
    ap.add_argument("-D", "--height-delta", type=float, default=1.0, help="Delta height for simulations.")
    ap.add_argument("-i", "--drop-init", type=float, default=0.5, help="Initial drop height for simulations.")
    ap.add_argument("-N", "--number-of-rocks", type=int, default=1000, help="Number of rocks per seeder.")
    return ap


def main(argv: list[str] | None = None) -> int:
    # ---- CLI -----------------------------------------------------------------
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap = add_arguments(ap)
    args = ap.parse_args(argv)
    return main_from_namespace(args)


def main_from_namespace(args: argparse.Namespace) -> int:
    filename       = args.filename
    material_lib   = args.material_lib
    material_name  = args.material_name
    profile_name   = args.profile_name
    height_start   = args.height_start
    height_delta   = args.height_delta
    drop_init      = args.drop_init
    number_of_rocks = args.number_of_rocks

    # ---- Load profiles and create simulations ----------------------------------
    print("🔄 Loading profiles...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file '{filename}' does not exist")
    profiles = pr.PointCloud.load(filename=os.path.abspath(filename), attributes=[material_name, profile_name])

    run(
        os.path.dirname(filename),
        material_name,
        profile_name,
        height_start,
        height_delta,
        drop_init,
        number_of_rocks,
        profiles,
        material_lib=material_lib,
    )

    return 0


def loadMaterials(user_path: str | None) -> Dict[int, pr.Material]:
    if user_path:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Material library file '{user_path}' does not exist")
        print(f"🔄 Loading material properties from: {os.path.abspath(user_path)}")
        return pr.readMaterials(os.path.abspath(user_path))
    else:
        # Bundled default inside pyrockfall/data/material_lib.csv
        with material_lib_path() as p:
            print(f"🔄 Loading default material properties: {p}")
            return pr.readMaterials(p)


def colorMaterials(materials: Dict[int, pr.Material]) -> None:
    def hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.strip().lstrip("#")
        if len(h) == 3:  # e.g. #abc
            h = "".join(ch*2 for ch in h)
        if len(h) != 6:
            raise ValueError("Use #RGB or #RRGGBB")
        i=0
        r = int(h[i:i+2], 16); i+=2
        g = int(h[i:i+2], 16); i+=2
        b = int(h[i:i+2], 16)
        return (r, g, b)
    colors = get_material_colors(latex_escape=False)
    for m in materials.values():
        m.color = hex_to_rgb(colors.get(m.name, "#808080"))


def config_simulation(
        init_height: float,
        delta_height: float,
        init_drop: float,
        analysis: pr.Analysis,
        profile: pr.Slope,
        rock: pr.Rock,
        materials: Dict[int, pr.Material],
    ) -> None:
    _, height_layer_start = pr.materialLayers(profile)
    height_layer_start = height_layer_start[height_layer_start > init_height]
    layer_edges = np.append(height_layer_start, init_height)
    n = np.maximum(3, np.ceil((layer_edges[:-1] - layer_edges[1:]) / delta_height).astype(int) + 1)
    heights = np.concatenate([
        np.linspace(hi, lo, n_i)[:-1] for lo, hi, n_i in zip(layer_edges[1:], layer_edges[:-1], n)
    ])
    heights = np.append(heights, layer_edges[-1])

    seeders = []
    positions = []
    for block_height in heights:
        x_block, _ = pr.findClosest(profile, block_height)
        seeder = pr.Seeder(points=np.array([x_block, block_height + init_drop]), rocks=[rock])
        seeder.translationalVelocity = [0.0, 0.0]
        seeders.append(seeder)
        positions.append((x_block, block_height))

    # Add a simple floor segment
    floor_coords = np.array([
        [profile.nodes[-1, 0] + 50.0, profile.nodes[-1, 1]],
    ])
    coords = np.vstack((profile.nodes, floor_coords))
    slope = pr.Slope(
        pr.Geometry(nodes=np.vstack((profile.nodes, floor_coords))),
        materialIDs=np.append(profile.materialIDs, len(profile.materialTable)-1),
        materials=profile.materialTable
    )

    for seeder in seeders:
        seeder.rockThrowMode = pr.SeederRocksThrown.Overall
        seeder.numberOfRocks = analysis.numberOfRocks
    analysis.seeders = seeders
    analysis.slope = slope


def readProfiles(
        mdl: pr.Model,
        material_name: str,
        profile_name: str,
        materials: Dict[int, pr.Material]
    ) -> Dict[int, pr.Slope]:
    mat_ids = list(materials)
    idx = {k:i for i, k in enumerate(mat_ids)}
    mats = [materials[m] for m in mat_ids]

    prof_ids = np.unique(mdl.get_attr(profile_name)).astype(int)
    profiles = {}
    for p_id in prof_ids:
        is_in_profile = mdl.get_attr(profile_name).astype(int)==p_id
        points = mdl.points[is_in_profile, :3].copy()
        points -= points[0]
        dist = np.linalg.norm(points[:,:2], axis=1)
        z = points[:,2]
        z -= z.min()
        points = np.vstack((dist, z)).T
        prof_mat_ids = mdl.get_attr(material_name)[is_in_profile].astype(int)
        prof_mat_ids = prof_mat_ids[:-1]
        mats_in_slope = [idx[m] for m in prof_mat_ids]
        profiles[p_id] = pr.Slope(pr.Geometry(nodes=points), materialIDs=mats_in_slope, materials=mats)
    return profiles

def run(
        output_dir: str,
        material_name: str,
        profile_name: str,
        init_height: float,
        delta_height: float,
        init_drop: float,
        number_of_rocks: int,
        model: pr.Model,
        material_lib: Optional[str] = None,
    ):
    # Read materials
    materials = loadMaterials(material_lib)

    # Read profiles
    profiles = readProfiles(model, material_name, profile_name, materials)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export materials
    colorMaterials(materials)
    pr.exportMaterials(os.path.join(output_dir, "materials.ini"), list(materials.values()))

    # Define unit rock
    unit_rock = pr.Rock("unit_rock")
    unit_rock.mass = 1.0
    unit_rock.density = 1.0
    unit_rock.color = (0, 0, 255)

    # Export rocks
    pr.exportRocks(os.path.join(output_dir, "rocks.ini"), [unit_rock])

    # Define analysis
    analysis = pr.Analysis()
    analysis.rockThrowMode = pr.AnalysisRocksThrown.IndividuallyPerSeeder
    analysis.numberOfRocks = number_of_rocks
    analysis.scaleByVelocity = True
    analysis.considerRotationalVelocity = True

    # Export project settings
    pr.exportSettings(os.path.join(output_dir, f"settings.ini"), analysis)

    for i, profile in profiles.items():
        config_simulation(init_height, delta_height, init_drop, analysis, profile, unit_rock, materials)
        pr.exportSlope(os.path.join(output_dir, f"profile_{i}.txt"), analysis.slope)
        pr.exportSeeders(os.path.join(output_dir, f"seeders_{i}.txt"), analysis.seeders, [unit_rock])

    print(f"✅ Done.")


def run_from_arrays(
        output_dir,
        material_name,
        profile_name,
        init_height,
        delta_height,
        init_drop,
        number_of_rocks: int,
        points,
        attrs=None,
        material_lib: Optional[str] = None,
    ):
    points = np.asarray(points, dtype=float)
    attrs = {} if attrs is None else attrs
    model = pr.PointCloud(points, attrs=attrs)

    return run(
        os.path.expanduser(output_dir),
        material_name,
        profile_name,
        init_height,
        delta_height,
        init_drop,
        number_of_rocks,
        model,
        material_lib=material_lib,
    )



if __name__ == "__main__":
    raise SystemExit(main())

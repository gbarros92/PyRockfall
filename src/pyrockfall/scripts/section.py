import os
import argparse
import numpy as np

import pyrockfall as pr


DESCRIPTION = "Extract profiles from 3D model by tracing sections along strike."
HELP = "Extract profiles from 3D model using PointCloud.section (no segmentation/alignment needed)."


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("filename", help="Input point cloud file (.ply with material attributes).")
    ap.add_argument("-S", "--profile-spacing", type=float, default=1.0, help="Spacing between profiles/sections (m).")
    ap.add_argument("-n", "--material-name", default="Material", help="Attribute name for material IDs in point cloud PLY.")

    ap.add_argument(
        "-r", "--transverse-radius",
        type=float,
        default=None,
        help="Half-width of the strip used to assign points to each section (m). Defaults to half the point cloud's vertical extent."
    )

    ap.add_argument("--min-points", type=int, default=20, help="Minimum number of points required for a section to be valid.")
    ap.add_argument("--max-turn-angle", type=float, default=45.0, help="Maximum change in marching direction between consecutive nodes (degrees).")

    ap.add_argument(
        "--save-profiles",
        choices=["yes", "no", "separate"],
        default="yes",
        help="Save profiles: yes (combined), no, or separate."
    )

    return ap


def main(argv: list[str] | None = None) -> int:
    # ---- CLI -----------------------------------------------------------------
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap = add_arguments(ap)
    args = ap.parse_args(argv)
    return main_from_namespace(args)


def main_from_namespace(args: argparse.Namespace) -> int:
    filename           = args.filename
    profile_spacing    = args.profile_spacing
    material_name      = args.material_name
    transverse_radius  = args.transverse_radius
    min_points         = args.min_points
    max_turn_angle     = args.max_turn_angle
    do_save_profiles   = args.save_profiles

    # ---- Load 3D model and extract sections -----------------------------------
    print("🔄 Loading 3D model...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file '{filename}' does not exist")
    directory = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    wall = pr.PointCloud.load(filename=os.path.abspath(filename), attributes=[material_name])

    output_dir = os.path.join(directory, f"{base_name}_profiles")

    run(
        output_dir,
        profile_spacing,
        material_name,
        transverse_radius,
        min_points,
        max_turn_angle,
        do_save_profiles,
        wall,
    )

    return 0


def run(
        output_dir,
        profile_spacing,
        material_name,
        transverse_radius,
        min_points,
        max_turn_angle,
        do_save_profiles,
        model,
    ):
    print("🔄 Extracting sections...")
    max_turn_angle_rad = np.deg2rad(max_turn_angle) if max_turn_angle is not None else None
    profiles, sections_pc = model.section(
        increment=profile_spacing,
        label=material_name,
        transverse_radius=transverse_radius,
        min_points=min_points,
        max_turn_angle=max_turn_angle_rad,
    )
    print(f"Extracted {len(profiles)} sections.")

    if do_save_profiles.lower() != "no":
        os.makedirs(output_dir, exist_ok=True)

        if do_save_profiles.lower() == "separate":
            profile_ids = sections_pc.get_attr("Profile").astype(int)
            for p_id in np.unique(profile_ids):
                mask = profile_ids == p_id
                section_pc = pr.PointCloud(sections_pc.points[mask])
                section_pc.set_attr(material_name, sections_pc.get_attr(material_name)[mask])
                section_pc.set_attr("Profile", sections_pc.get_attr("Profile")[mask])
                section_pc.save(os.path.join(output_dir, f"profile_{p_id}.ply"), attributes="*", overwrite=True)
            print("Profiles saved separately")
        else:
            sections_pc.save(os.path.join(output_dir, "profiles.ply"), attributes="*", overwrite=True)
            print("Profiles saved")

    print(f"✅ Done.")


def run_from_arrays(
        output_dir,
        profile_spacing,
        material_name,
        transverse_radius,
        min_points,
        max_turn_angle,
        do_save_profiles,
        points,
        attrs=None,
    ):
    points = np.asarray(points, dtype=float)
    attrs = {} if attrs is None else attrs
    model = pr.PointCloud(points, attrs=attrs)

    return run(
        os.path.expanduser(output_dir),
        profile_spacing,
        material_name,
        transverse_radius,
        min_points,
        max_turn_angle,
        do_save_profiles,
        model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

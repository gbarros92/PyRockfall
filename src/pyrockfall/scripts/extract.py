import os
import argparse
import numpy as np
from dataclasses import dataclass

from typing import List

import pyrockfall as pr


DESCRIPTION = "Extract profiles from 3D model (point cloud or mesh)."
HELP = "Extract profiles from 3D model."


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("filename", help="Input point cloud file (.ply with material attributes).")
    ap.add_argument("-L", "--segment-length", type=float, default=10.0, help="Segment length along the strike (m).")
    ap.add_argument("-S", "--profile-spacing", type=float, default=1.0, help="Spacing between profiles (m).")
    ap.add_argument("-q", "--profile-resolution", type=float, default=0.15, help="Profile resolution (m).")
    ap.add_argument("-n", "--material-name", default="Material", help="Attribute name for material IDs in point cloud PLY.")

    ap.add_argument(
        "-R", "--remove-materials",
        dest="remove_ids",
        metavar="ID",
        type=int,
        nargs="+",
        default=None,
        help="Remove materials by IDs, e.g. -R 6 7"
    )

    ap.add_argument(
        "--save-profiles",
        choices=["yes", "no", "separate"],
        default="yes",
        help="Save profiles: yes, no, or separate."
    )

    ap.add_argument(
        "--save-segments",
        choices=["yes", "no", "aligned"],
        default="no",
        help="Save segments: yes, no, or aligned (unrolled strike, dip-direction, elevation coordinates)."
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
    segment_length     = args.segment_length
    profile_spacing    = args.profile_spacing
    profile_resolution = args.profile_resolution
    material_name      = args.material_name
    remove_ids         = []
    if args.remove_ids is not None:
        remove_ids = set(args.remove_ids)
    do_save_profiles   = args.save_profiles
    do_save_segments   = args.save_segments

    # ---- Load 3D model and extract profiles ----------------------------------
    print("🔄 Loading 3D model...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file '{filename}' does not exist")
    directory = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    wall = pr.PointCloud.load(filename=os.path.abspath(filename), attributes=[material_name])

    output_dir = os.path.join(directory, f"{base_name}_profiles")

    run(
        output_dir,
        segment_length,
        profile_spacing,
        profile_resolution,
        material_name,
        remove_ids,
        do_save_profiles,
        do_save_segments,
        wall
    )

    return 0


@dataclass
class ProcessResult:
    min_z: float
    frame: pr.CurvilinearFrame


def process_model(
        mdl: pr.Model,
        material_name: str,
        profile_spacing: float,
        remove_ids: List[int] = [],
    ) -> ProcessResult:
    # Fit the curvilinear frame: s along the strike, d down-dip, e elevation.
    # Slicing along s needs no alignment of the model with an axis.
    frame = mdl.fitFrame()
    print(f'Wall length: {frame.length:.2f} in {frame.n_segments} polyline segments')
    info = frame.diagnostics(mdl.points)
    if info["n_nan"] or info["n_ambiguous"]:
        print(f'⚠️ {info["n_nan"]} points outside the frame, {info["n_ambiguous"]} ambiguous')

    # Common floor for all profiles: lowest elevation reached by every complete
    # slice, excluding the specified materials
    min_z = pr.wallFloorElevation(
        mdl, profile_spacing, label=material_name, ignore_ids=list(remove_ids), frame=frame
    )
    mdl.apply_point_mask_(mdl.points[:, 2] >= min_z)  # Remove points below the common floor

    return ProcessResult(min_z, frame)


def extract_profiles(
        mdl: pr.Model,
        processed: ProcessResult,
        material_name: str,
        segment_length: float,
        profile_spacing: float,
        profile_resolution: float,
        remove_ids: List[int] = [],
        save_segments: str = '',
        do_save_segments_aligned: bool = False,
        save_coords: str = '',
        do_save_coords_separate: bool = False
    ) -> None:
    # Get segments from model with specified segment length
    coords_list: List[np.ndarray] = []
    frame = processed.frame

    def save_seg(seg_id, seg):
        if not save_segments:
            return
        if do_save_segments_aligned:
            # Unrolled coordinates (s, d, e) instead of (x, y, z)
            sde = np.nan_to_num(frame.xyz_to_sde(seg.points))
            n = seg.points.shape[0]
            attrs = {k: v for k, v in seg.attrs.items() if v.shape[0] == n}
            if isinstance(seg, pr.Mesh):
                seg = pr.Mesh(points=sde, triangles=seg.triangles, attrs=attrs)
            else:
                seg = pr.PointCloud(points=sde, attrs=attrs)
        seg.save(save_segments+f'_{seg_id}.ply', attributes=material_name, overwrite=True)

    profile_id = -1
    for seg_id, seg in enumerate(mdl.segment(segment_length)):
        save_seg(seg_id, seg)

        # Extract profiles from segment: slices of constant s, nodes in (d, e)
        slices, positions = seg.slice(
            None,
            increment=profile_spacing,
            slice_height_increment=profile_resolution,
            label=material_name,
            frame=frame,
        )
        for profile, s_k in zip(slices, positions):
            # Find beginning of the floor
            profile.nodes[-1][1] = processed.min_z  # Ensure profile ends at the floor

            # Find mean segment length in the profile for tolerance
            len_seg_mean = 0.0
            for i in range(len(profile.nodes)-1):
                xi, yi = profile.nodes[i]
                xj, yj = profile.nodes[i+1]
                len_seg_mean += np.sqrt((xj-xi)**2 + (yj-yi)**2)
            len_seg_mean /= len(profile.nodes)-1

            max_y_profile = profile.nodes[:,1].max()
            min_y_profile = profile.nodes[:,1].min()

            for m_id in remove_ids:
                mask = profile.attributes == m_id
                mask_augmented = np.r_[False, mask, False]
                d = np.diff(mask_augmented.astype(int))
                starts = np.where(d == 1)[0]
                ends   = np.where(d == -1)[0] - 1
                
                # Check if material is only on top or bottom of the profile
                elem_keep = np.ones(len(profile.elements), dtype=bool)
                for s, e in zip(starts, ends):
                    elems = profile.elements[s:e+1]
                    y_m = profile.nodes[elems][:, :, 1]

                    max_y_m = y_m.max()
                    min_y_m = y_m.min()

                    if (
                        abs(max_y_m - max_y_profile) < len_seg_mean / 2
                        or abs(min_y_m - min_y_profile) < len_seg_mean / 2
                    ):
                        elem_keep[s:e+1] = False

                new_elements = profile.elements[elem_keep]
                new_attributes = profile.attributes[elem_keep] if profile.attributes is not None else None

                used_nodes = np.unique(new_elements.ravel())
                new_nodes = profile.nodes[used_nodes]
                if profile.hasUncertainty:
                    new_nodes_std = profile.nodes_std[used_nodes]
                else:
                    new_nodes_std = None

                old_to_new = -np.ones(len(profile.nodes), dtype=int)
                old_to_new[used_nodes] = np.arange(len(used_nodes))

                new_elements = old_to_new[new_elements]
                profile = pr.Geometry(new_nodes, new_nodes_std, new_elements, new_attributes)
            
            if save_coords:
                # Back to world coordinates through the frame
                sde = np.column_stack((np.full(len(profile.nodes), s_k), profile.nodes))
                coords = frame.sde_to_xyz(sde).T
                mat_list = profile.attributes
                if mat_list is None:
                    mat_list = np.full_like(profile.elements, -1)
                mat_list = np.append(mat_list, mat_list[-1])  # Add last material again
                profile_id += 1
                coords = np.vstack((coords, mat_list, np.full_like(mat_list, profile_id)))
                coords_list.append(coords.T)

    # Save profiles
    if save_coords:
        if do_save_coords_separate:
            for i in range(len(coords_list)):
                prf = pr.PointCloud(coords_list[i][:,:3])
                prf.set_attr(material_name, coords_list[i][:,3])
                prf.set_attr('Profile', coords_list[i][:,4])
                prf.save(save_coords + f"_{i}.ply", attributes='*', overwrite=True)
        else:
            coords = np.vstack(coords_list)
            prf = pr.PointCloud(coords[:,:3])
            prf.set_attr(material_name, coords[:,3])
            prf.set_attr('Profile', coords[:,4])
            prf.save(save_coords + ".ply", attributes='*', overwrite=True)


def run(
        output_dir,
        segment_length,
        profile_spacing,
        profile_resolution,
        material_name,
        remove_ids,
        do_save_profiles,
        do_save_segments,
        model
    ):
    print("🔄 Processing 3D model...")
    processed = process_model(model, material_name, profile_spacing, remove_ids)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    segments_out = ''
    do_save_aligned = False
    if do_save_segments.lower() != 'no':
        segments_out = os.path.join(output_dir, 'segment')
        if do_save_segments.lower() == 'aligned':
            do_save_aligned = True
    print('Segments will ' + ('' if segments_out else 'not ') + 'be saved' + (' aligned' if do_save_aligned else ''))

    profiles_out = ''
    do_save_separate = False
    if do_save_profiles.lower() != 'no':
        profiles_out = os.path.join(output_dir, 'profile')
        if do_save_profiles.lower() == 'yes':
            profiles_out += 's'
        else:
            do_save_separate = True
    print('Profiles will ' + ('' if profiles_out else 'not ') + 'be saved' + (' separately' if do_save_separate else ''))

    print("🔄 Extracting profiles...")
    extract_profiles(
        model,
        processed,
        material_name,
        segment_length,
        profile_spacing,
        profile_resolution,
        remove_ids,
        save_segments=segments_out,
        do_save_segments_aligned=do_save_aligned,
        save_coords=profiles_out,
        do_save_coords_separate=do_save_separate,
    )

    print(f"✅ Done.")


def run_from_arrays(
        output_dir,
        segment_length,
        profile_spacing,
        profile_resolution,
        material_name,
        remove_ids,
        do_save_profiles,
        do_save_segments,
        points,
        triangles=None,
        attrs=None,
    ):
    points = np.asarray(points, dtype=float)

    if triangles is None:
        triangles = np.empty((0, 3), dtype=int)
    else:
        triangles = np.asarray(triangles, dtype=int)

    attrs = {} if attrs is None else attrs
    remove_ids = [] if remove_ids is None else remove_ids

    if triangles.size == 0:
        model = pr.PointCloud(points, attrs=attrs)
    else:
        model = pr.Mesh(points=points, triangles=triangles, attrs=attrs)

    return run(
        os.path.expanduser(output_dir),
        segment_length,
        profile_spacing,
        profile_resolution,
        material_name,
        remove_ids,
        do_save_profiles,
        do_save_segments,
        model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

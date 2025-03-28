import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import map_coordinates

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)


def get_checkpoint_file(latest=False):
    if latest:
        # Get the latest checkpoint file
        CHECKPOINTS_FOLDER = "./training_output/checkpoints"
        files = os.listdir(CHECKPOINTS_FOLDER)
        files = sorted(
            files,
            key=lambda x: os.path.getmtime(os.path.join(CHECKPOINTS_FOLDER, x)),
            reverse=True,
        )
        if not files:
            raise FileNotFoundError(
                "No checkpoint files found in the specified folder."
            )
        ckpt_path = os.path.join(CHECKPOINTS_FOLDER, files[0])
    else:
        ckpt_path = "/content/membrain_tutorial_scripts/membrain_pick_ckpt/tutorial_0-epoch=199-val_loss=1.05.ckpt"
    return ckpt_path


def create_membrain_pick_training_data():
    os.system("mkdir training_data")
    os.system("mkdir training_data/train")
    os.system("mkdir training_data/val")
    os.system("scp mesh_data/Tomo0001_T1S1M12.h5 ./training_data/train/")
    os.system("scp mesh_data/Tomo0001_T1S1M14.h5 ./training_data/train/")
    os.system("scp mesh_data/Tomo0001_T1S1M16.h5 ./training_data/train/")
    os.system("scp mesh_data/Tomo0001_T1S1M17.h5 ./training_data/val/")
    os.system("scp mesh_data/Tomo0001_T1S1M19.h5 ./training_data/val/")
    os.system("scp positions/Tomo0001_T1S1M12.star ./training_data/train/")
    os.system("scp positions/Tomo0001_T1S1M14.star ./training_data/train/")
    os.system("scp positions/Tomo0001_T1S1M16.star ./training_data/train/")
    os.system("scp positions/Tomo0001_T1S1M17.star ./training_data/val/")
    os.system("scp positions/Tomo0001_T1S1M19.star ./training_data/val/")
    print("Training data created.")


def download_membrain_model():
    import gdown

    print(
        "Downloading MemBrain model from Google Drive. This should be faster, but can still take few minutes."
    )
    # File ID and destination path
    file_id = "1tSQIz_UCsQZNfyHg0RxD-4meFgolszo8"
    destination = "./membrain_v10_alpha.ckpt"  # Replace with your desired file name

    # Download file from Google Drive
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", destination, quiet=False
    )
    print("Checkpointt file is stored here:", destination)


def load_tutorial_data():
    # Load the data
    print("Downloading data from Zenodo. This can take few minutes.")
    os.system(
        "curl https://zenodo.org/api/records/14610597/files/membrain_tutorial.zip/content > membrain_tutorial.zip"
    )
    print("Unzipping downloaded data.")
    os.system("unzip ./membrain_tutorial.zip")
    print("")
    print("Done. Files in the tutorial folder:")
    for filename in os.listdir("./data5mbs"):
        print(filename)


def load_membrane_data_raw(membrane_file):
    from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5, read_star_file

    assert membrane_file in membrane_files, f"Invalid membrane file: {membrane_file}"
    # Load and prepare data for plotly:
    mesh_path = f"./mesh_data/Tomo0001_{membrane_file}.h5"
    mesh_data = load_mesh_from_hdf5(mesh_path)
    tomo_path = "./Tomo0001.mrc"
    tomo = load_tomogram(tomo_path).data
    star_file = f"./positions/Tomo0001_{membrane_file}.star"
    positions = read_star_file(star_file)
    positions = np.array(positions)
    points = mesh_data["points"]
    tomo_values = map_coordinates(tomo, points.T)
    # if "scores" in the keys: also load that
    out_dict = {
        "points": points,
        "tomo_values": tomo_values,
        "positions": positions,
    }
    if "scores" in mesh_data.keys():
        scores = mesh_data["scores"]
        out_dict["scores"] = scores
    return out_dict


def load_membrane_data_pred(membrane_file):
    from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5, read_star_file

    assert membrane_file in membrane_files, f"Invalid membrane file: {membrane_file}"
    # Load and prepare data for plotly:
    mesh_path = f"./predict_output/Tomo0001_{membrane_file}.h5"
    mesh_data = load_mesh_from_hdf5(mesh_path)
    tomo_path = "./Tomo0001.mrc"
    tomo = load_tomogram(tomo_path).data
    positions = mesh_data["cluster_centers"]
    positions = np.array(positions) / 14.08
    points = np.array(mesh_data["points"]) / 14.08
    tomo_values = map_coordinates(tomo, points.T)
    # if "scores" in the keys: also load that
    out_dict = {
        "points": points,
        "tomo_values": tomo_values,
        "positions": positions,
    }
    if "scores" in mesh_data.keys():
        scores = mesh_data["scores"]
        out_dict["scores"] = scores
    return out_dict


def generate_sphere(center, radius=1, resolution=10):
    phi, theta = np.linspace(0, np.pi, resolution), np.linspace(
        0, 2 * np.pi, resolution
    )
    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    return x, y, z


def crop_tomogram(tomo_file, out_file, extents=[(100, 200), (100, 200), (100, 200)]):
    tomo = load_tomogram(tomo_file).data
    tomo_cropped = tomo[
        extents[0][0] : extents[0][1],
        extents[1][0] : extents[1][1],
        extents[2][0] : extents[2][1],
    ]
    store_tomogram(out_file, tomo_cropped)
    return tomo_cropped


def visualize_membranes(points, positions, colors, color_scales, z_shifts):
    import plotly.graph_objects as go

    data = []

    for pointset, color, cscale, z_shift in zip(points, colors, color_scales, z_shifts):
        data.append(
            go.Scatter3d(
                x=pointset[:, 0],
                y=pointset[:, 1],
                z=pointset[:, 2] + z_shift,
                mode="markers",
                marker=dict(
                    size=3.5,
                    color=color,
                    colorscale=cscale,
                ),
            )
        )

    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        ),
    )

    if positions is not None:
        for point in positions:
            x, y, z = generate_sphere(point, radius=4)
            fig.add_trace(
                go.Surface(
                    x=x, y=y, z=z, colorscale="Viridis", opacity=0.9, showscale=False
                )
            )
    fig.show()


membrane_files = [
    "T1S1M12",
    "T1S1M14",
    "T1S1M16",
    "T1S1M17",
    "T1S1M19",
]

import warnings
warnings.simplefilter("ignore", UserWarning)
import socket
from mpi4py import MPI
import numpy as np
import sys
from PIL import Image

# Linear interpolation function for color and opacity
def linear_interpolate(value, table, is_color=False):
    """
    Linearly interpolates a value based on a provided table of mappings.
    If is_color is True, interpolates in RGB space for color mapping;
    otherwise, interpolates opacity values.
    """
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= value <= x1:
            # Calculate the interpolation factor t
            t = (value - x0) / (x1 - x0)
            # Interpolate colors if is_color, else interpolate single opacity value
            return [(1 - t) * c0 + t * c1 for c0, c1 in zip(y0, y1)] if is_color else (1 - t) * y0 + t * y1
    # Return default values if out of range
    return [0, 0, 0] if is_color else 0

def main():
    """
    Main function for MPI-based volume rendering.
    Parses arguments, distributes sub-volumes to each process, performs rendering,
    and gathers the results to construct the final image.
    """
    # Initialize MPI and retrieve the process rank and number of processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    print(f"Rank {rank} running on {socket.gethostname()}")

    # Start timing the entire program execution
    start_time_total = MPI.Wtime()
    send_time, recv_time = 0, 0

    # Verify that sufficient command-line arguments are provided
    if len(sys.argv) < 8:
        if rank == 0:
            print("Usage: mpirun -np <num_procs> python file.py <dataset_name> <X_parts> <Y_parts> <Z_parts> <step_size> <opacity_tf> <color_tf>")
        sys.exit()

    # Parse input arguments
    dataset_name, X_parts, Y_parts, Z_parts, step_size, opacity_file, color_file = sys.argv[1:8]
    X_parts, Y_parts, Z_parts = int(X_parts), int(Y_parts), int(Z_parts)
    step_size = float(step_size)

    # Define mapping for known dataset dimensions
    volume_dims_map = {
        "1000x1000x200": (1000, 1000, 200),
        "2000x2000x400": (2000, 2000, 400),
    }
    # Determine the volume dimensions based on the dataset name
    for key in volume_dims_map:
        if key in dataset_name:
            volume_dims = volume_dims_map[key]
            break

    # Load opacity and color transfer functions from files
    opacity_map = []
    with open(opacity_file, 'r') as file:
        values = [float(v) for line in file for v in line.replace(',', '').strip().split()]
        opacity_map = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

    color_map = []
    with open(color_file, 'r') as file:
        values = [float(v) for line in file for v in line.replace(',', '').strip().split()]
        color_map = [(values[i], (values[i + 1], values[i + 2], values[i + 3])) for i in range(0, len(values), 4)]

    # Load the data volume and split it for distribution
    local_subvolume = None  # Initialize local subvolume for all ranks
    if rank == 0:
        # Load the full data volume from the file and reshape it
        data_volume = np.fromfile(dataset_name, dtype=np.float32).reshape(volume_dims, order='F')
        # Split data volume into sub-volumes across X and Y parts for distribution
        sub_volumes = [np.array_split(slice, Y_parts, axis=1) for slice in np.array_split(data_volume, X_parts, axis=0)]

        # Start sending sub-volumes to each process
        send_start = MPI.Wtime()
        for i in range(X_parts):
            for j in range(Y_parts):
                for k in range(Z_parts):
                    target_rank = i * Y_parts * Z_parts + j * Z_parts + k
                    # Select the portion of the volume to send to the target process
                    portion = sub_volumes[i][j][:, :, k::Z_parts]
                    if target_rank == 0:
                        local_subvolume = portion  # Assign the portion directly for rank 0
                    else:
                        comm.send(portion, dest=target_rank, tag=target_rank)
        send_time = MPI.Wtime() - send_start
    else:
        # Receive the sub-volume for this rank
        recv_start = MPI.Wtime()
        local_subvolume = comm.recv(source=0, tag=rank)
        recv_time = MPI.Wtime() - recv_start

    # Check if the subvolume is assigned correctly
    if local_subvolume is None:
        print(f"Rank {rank} received no data.")
        sys.exit()

    # Begin rendering each pixel of the sub-volume
    h, w, d = local_subvolume.shape
    local_img = np.zeros((h, w, 3))  # Initialize the local image for the sub-volume
    render_start = MPI.Wtime()

    # Raycasting rendering logic for each pixel in the sub-volume
    for col in range(w):
        for row in range(h):
            color_accum = np.zeros(3)  # Accumulated color for this pixel
            opacity_accum = 0  # Accumulated opacity
            z_pos = 0.0
            while z_pos < d:
                # Determine interpolated intensity and perform linear interpolation for color/opacity
                z_int = int(z_pos)
                z_int_next = min(z_int + 1, d - 1)
                ratio = z_pos - z_int
                value = (1 - ratio) * local_subvolume[row, col, z_int] + ratio * local_subvolume[row, col, z_int_next]

                # Get interpolated color and opacity based on the intensity value
                color = np.array(linear_interpolate(value, color_map, is_color=True))
                opacity = linear_interpolate(value, opacity_map)
                
                # Accumulate color and opacity, factoring in existing accumulation
                color_accum += (1 - opacity_accum) * color * opacity
                opacity_accum += (1 - opacity_accum) * opacity
                
                # Break early if opacity reaches threshold
                if opacity_accum >= 0.98:
                    break
                z_pos += step_size
            local_img[row, col, :] = color_accum  # Store the accumulated color for this pixel

    computation_time = MPI.Wtime() - render_start

    # Gather the rendered sub-images from all processes
    gather_start = MPI.Wtime()
    gathered_imgs = comm.gather(local_img, root=0)
    gather_time = MPI.Wtime() - gather_start

    # Gather max times across processes for reporting
    max_times = comm.reduce([computation_time, send_time, recv_time, gather_time, MPI.Wtime() - start_time_total], op=MPI.MAX)
    
    if rank == 0:
        # Assemble the final full image from gathered sub-images
        sub_h, sub_w, _ = gathered_imgs[0].shape
        final_img = np.zeros((sub_h * X_parts, sub_w * Y_parts, 3))  # Final image array
        for i in range(X_parts):
            for j in range(Y_parts):
                color_acc = np.zeros((sub_h, sub_w, 3))  # Accumulated color for each sub-image
                opacity_acc = np.zeros((sub_h, sub_w))   # Accumulated opacity for each sub-image
                for k in range(Z_parts):
                    idx = i * Y_parts * Z_parts + j * Z_parts + k
                    img = gathered_imgs[idx]
                    alpha = 1 - opacity_acc
                    color_acc += img * alpha[:, :, None]
                    opacity_acc += alpha
                final_img[i * sub_h:(i + 1) * sub_h, j * sub_w:(j + 1) * sub_w, :] = color_acc

        # Save the final assembled image to a file
        output_file = f"{X_parts}_{Y_parts}_{Z_parts}.png"
        final_image = np.flipud(final_img)  # Flip image vertically
        final_image = np.rot90(final_image)  # Rotate image 90 degrees clockwise
        Image.fromarray((final_image * 255).astype(np.uint8)).save(output_file)
        print(f"Image saved as {output_file}")

        # Print execution time summary for different parts of the process
        labels = ["Computation", "Communication (Send)", "Communication (Recv)", "Gather", "Total Execution"]
        print("\nExecution Times (seconds):\n" + "\n".join([f"{lbl:<25} | {t:.4f}" for lbl, t in zip(labels, max_times)]))

if __name__ == "__main__":
    main()

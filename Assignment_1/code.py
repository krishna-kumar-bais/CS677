import numpy as np
from mpi4py import MPI
from PIL import Image
 
# Set up MPI environment
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
 

def load_raw_data(filepath, dimensions):
    with open(filepath, 'rb') as file:
        raw_values = np.fromfile(file, dtype=np.float32)
    return np.reshape(raw_values, dimensions, order='F')
 


def parse_color_map(filepath):
    with open(filepath, 'r') as file:
        raw_lines = file.read().replace('\n', ',').split(',')
        parsed_values = [float(v) for v in raw_lines if v.strip()]
   
    color_map = {}
    for idx in range(0, len(parsed_values), 4):
        key = parsed_values[idx]
        color_map[key] = np.array(parsed_values[idx+1:idx+4])
   
    return color_map
 

def parse_opacity_map(filepath):
    with open(filepath, 'r') as file:
        raw_lines = file.read().replace('\n', ',').split(',')
        parsed_values = [float(v) for v in raw_lines if v.strip()]
 
    opacity_map = {}
    for idx in range(0, len(parsed_values), 2):
        opacity_map[parsed_values[idx]] = parsed_values[idx+1]
 
    return opacity_map

def interpolate_value(value, value_map):
    sorted_keys = sorted(value_map.keys())
    for i in range(len(sorted_keys) - 1):
        if sorted_keys[i] <= value <= sorted_keys[i + 1]:
            low_key, high_key = sorted_keys[i], sorted_keys[i + 1]
            t = (value - low_key) / (high_key - low_key)
            return value_map[low_key] * (1 - t) + value_map[high_key] * t
    return value_map[sorted_keys[-1]] if value >= sorted_keys[-1] else value_map[sorted_keys[0]]
 
def distribute_2d_array(data_array, process_id, total_processes):
    array_shape = data_array.shape
    grid_dim = int(np.sqrt(total_processes))
    block_height, block_width = array_shape[0] // grid_dim, array_shape[1] // grid_dim
 
    grid_x = process_id // grid_dim
    grid_y = process_id % grid_dim
 
    x_start = grid_x * block_height
    x_end = (grid_x + 1) * block_height if grid_x != grid_dim - 1 else array_shape[0]
 
    y_start = grid_y * block_width
    y_end = (grid_y + 1) * block_width if grid_y != grid_dim - 1 else array_shape[1]
 
    return data_array[x_start:x_end, y_start:y_end, :]
 

def volume_render(sub_block, color_map, opacity_map, step_size=1.0, termination_threshold=0.95):
    height, width, depth = sub_block.shape
    output_image = np.zeros((height, width, 3))
    terminated_rays = 0
    total_rays = height * width
 
    for i in range(height):
        for j in range(width):
            ray_color = np.zeros(3)
            ray_opacity = 0.0
            for k in range(0, depth, step_size):
                voxel_value = sub_block[i, j, k]
                alpha = interpolate_value(voxel_value, opacity_map)
                color = interpolate_value(voxel_value, color_map)
 
                ray_color += (1 - ray_opacity) * alpha * color
                ray_opacity += (1 - ray_opacity) * alpha
 
                if ray_opacity >= termination_threshold:
                    terminated_rays += 1
                    break
            output_image[i, j] = ray_color
 
    return output_image, terminated_rays / total_rays
 
def run_rendering_task():
    raw_data_path = 'Isabel_1000x1000x200_float32.raw'
    color_mapping_file = 'color_TF.txt'
    opacity_mapping_file = 'opacity_TF.txt'
    sampling_step = 1
    decomposition_mode = '2'
   
    data_dims = (1000, 1000, 200)
    dataset = load_raw_data(raw_data_path, data_dims)
    color_mapping = parse_color_map(color_mapping_file)
    opacity_mapping = parse_opacity_map(opacity_mapping_file)
 
    if decomposition_mode == '1':
        slice_height = data_dims[0] // mpi_size
        start_slice = mpi_rank * slice_height
        end_slice = (mpi_rank + 1) * slice_height if mpi_rank != mpi_size - 1 else data_dims[0]
        sub_data_block = dataset[start_slice:end_slice, :, :]
    else:
        sub_data_block = distribute_2d_array(dataset, mpi_rank, mpi_size)
 
    rendered_sub_image, termination_ratio = volume_render(sub_data_block, color_mapping, opacity_mapping, sampling_step)
 
    gathered_images = mpi_comm.gather(rendered_sub_image, root=0)
    termination_ratios = mpi_comm.gather(termination_ratio, root=0)
 
    if mpi_rank == 0:
        if decomposition_mode == '1':
            final_image = np.vstack(gathered_images)
        else:
            grid_size = int(np.sqrt(mpi_size))
            rows = []
            for row_index in range(grid_size):
                row_pieces = [gathered_images[row_index * grid_size + col_index] for col_index in range(grid_size)]
                rows.append(np.hstack(row_pieces))
            final_image = np.vstack(rows)
 
        avg_termination_ratio = np.mean(termination_ratios)
        image_output = Image.fromarray(np.uint8(final_image * 255))
        image_output.save('final_IMAGE.png')
 
        print(f"Mean early termination ratio: {avg_termination_ratio}")
 
if __name__ == "__main__":
    run_rendering_task()
 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import os
from ase.io import read, write
import sys
import argparse
import csv
import multiprocessing
import time

TIME_STEP = 100
TIME_CONVERSION = 0.00025

def read_lammps_dump(input_file):
    return read(input_file, format='lammps-dump-text', index=':')

def parse_args(): 
    parser = argparse.ArgumentParser(
        description="This is a code to calculate Mean Square Displacement from LAMMPS trajectory."
    )
    parser.add_argument(
        "input_file",
        help="Path of LAMMPS trajectory file"
    )
    parser.add_argument(
        'output_file',
        help='Filename for writing an output CSV file, end with file extension .csv'
    )
    parser.add_argument(
        '--num_processes', '-np',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of processes to use for parallel computation. Default is number of CPU cores.'
    )
    return parser.parse_args()

def calculate_displacement_vectors_single_interval(path, t, interval_size):
    displacement_vectors_t = []

    for i in range(0, len(path) - t, int(interval_size)):
        initial_position = np.array([path[i][n].position for n, element in enumerate(path[i].symbols) if element == 'H'])
        final_position = np.array([path[i + t][n].position for n, element in enumerate(path[i + t].symbols) if element == 'H'])

        # Displacement vectors
        r = np.abs(initial_position - final_position)

        displacement_vectors_t.append(r)

    return displacement_vectors_t

def calculate_displacement_vectors(path):
    displacement_vectors = []
    time_list = []

    for t in range(1, len(path)):
        time = t * TIME_STEP * TIME_CONVERSION
        time_list.append(time)

        base_interval_size = 0.1 * len(path)
        interval_size = max(1, base_interval_size - 0.1 * t)

        displacement_vectors_t = calculate_displacement_vectors_single_interval(path, t, interval_size)
        displacement_vectors.append(displacement_vectors_t)

    return time_list, displacement_vectors

def calculate_msd(displacement_vectors):
    average_msd_list = []

    for displacement_vectors_t in displacement_vectors:
        msd_list = []

        for r in displacement_vectors_t:
            r2 = np.square(r)
            msd = np.mean(r2, axis=0).sum()
            msd_list.append(msd)

        average_msd = np.mean(msd_list)
        average_msd_list.append(average_msd)

    return average_msd_list

def write_log(time_list, average_msd_list, output_file):
    rows = list(zip(time_list, average_msd_list))
    with open(output_file, 'w', newline='') as csvfile:
        # Create a CSV writer
        csvwriter = csv.writer(csvfile)

        # Write headers
        csvwriter.writerow(['Time', 'Average MSD'])

        # Write the data
        for time, average_msd in rows:
            formatted_time = '{:.3g}'.format(time)  # format to three significant figures
            formatted_average_msd = '{:.10g}'.format(average_msd)  # format to six significant figures
            csvwriter.writerow([formatted_time, formatted_average_msd])
    print(f"Data has been written to {output_file}")

def main():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    num_processes = args.num_processes

    path = read_lammps_dump(input_file)

    start_time = time.time()

    # Split the time steps into chunks for parallel processing
    chunk_size = (len(path) - 1) // num_processes
    chunks = [range(i * chunk_size + 1, (i + 1) * chunk_size + 1) for i in range(num_processes)]
    remaining_steps = (len(path) - 1) % num_processes
    if remaining_steps != 0:
        chunks[-1] = list(chunks[-1]) + list(range((num_processes - remaining_steps) * chunk_size + 1, len(path)))
        
    processes = []
    displacement_vectors = []

    for chunk in chunks:
        process = multiprocessing.Process(target=calculate_displacement_vectors_chunk, args=(path, chunk))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    time_list, displacement_vectors = calculate_displacement_vectors(path)
    average_msd_list = calculate_msd(displacement_vectors)

    write_log(time_list, average_msd_list, output_file)

    end_time = time.time()
    print(f"Script execution time: {end_time - start_time} seconds")

def calculate_displacement_vectors_chunk(path, chunk):
    displacement_vectors_chunk = []

    for t in chunk:
        base_interval_size = 0.1 * len(path)
        interval_size = max(1, base_interval_size - 0.1 * t)
        displacement_vectors_t = calculate_displacement_vectors_single_interval(path, t, interval_size)
        displacement_vectors_chunk.append(displacement_vectors_t)

    return displacement_vectors_chunk

if __name__ == "__main__":
    main()
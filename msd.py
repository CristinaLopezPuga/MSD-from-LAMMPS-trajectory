import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import os
from ase.io import read,write
import sys
import argparse
import csv

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
        help='Filename for writing a output CSV file, end with file extension .csv'
    )
    return parser.parse_args()


def calculate_displacement_vectors(path, a, b, c):
    displacement_vectors = []
    time_list = []

    for t in range(1, len(path)):
        time = t * TIME_STEP * TIME_CONVERSION
        time_list.append(time)

        base_interval_size = 0.1 * len(path)
        interval_size = max(1, base_interval_size - 0.1 * t)

        displacement_vectors_t = []

        for i in range(0, len(path) - t, int(interval_size)):
            initial_position = np.array([path[i][n].position for n, element in enumerate(path[i].symbols) if element == 'H'])
            final_position = np.array([path[i + t][n].position for n, element in enumerate(path[i + t].symbols) if element == 'H'])

            # Displacement vectors
            r = np.abs(initial_position - final_position)

            # Boundary conditions
            condition_1 = r[:, 0] > a / 2
            condition_2 = r[:, 1] > b / 2
            condition_3 = r[:, 2] > c / 2
            condition_4 = r[:, 0] < -a / 2
            condition_5 = r[:, 1] < -b / 2
            condition_6 = r[:, 2] < -c / 2

            r[condition_1, 0] -= a
            r[condition_2, 1] -= b
            r[condition_3, 2] -= c
            r[condition_4, 0] += a
            r[condition_5, 1] += b
            r[condition_6, 2] += c

            displacement_vectors_t.append(r)

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


def write_log(time_list,average_msd_list,output_file):
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

    path = read_lammps_dump(input_file)
    a = path[0].cell[0][0]
    b = path[0].cell[1][1]
    c = path[0].cell[2][2]

    time_list, displacement_vectors = calculate_displacement_vectors(path, a, b, c)
    average_msd_list = calculate_msd(displacement_vectors)

    write_log(time_list, average_msd_list, output_file)

if __name__ == "__main__":
    main()

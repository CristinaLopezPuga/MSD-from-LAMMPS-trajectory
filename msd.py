import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import os
from ase.io import read,write
import sys
import argparse


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

#This code calculates the MSD displacement from a LAMMPS trajectory using ASE to get atom positions 

# Constants
#To-do change time constants to user defined variables
THRESHOLD = 25
TIME_STEP = 100
TIME_CONVERSION = 0.00025


def get_msd(file):
    path = read(file,format='lammps-dump-text',index=':')
    # Lattice constants
    LATTICE_A = path[0].cell[0][0] 
    LATTICE_B = path[0].cell[1][1]
    LATTICE_C = path[0].cell[2][2]
    # Initialize arrays
    time_list = []
    initial_positions = []
    final_positions = []
    displacement = []
    msd_list = []
    average_msd_list  = []
    # Loop over each pair of consecutive paths
    for t in range(1, len(path)): 
        # Calculate time for each iteration
        time = t * TIME_STEP * TIME_CONVERSION
        time_list.append(time)
        for i in range(len(path)-t):
            # Extract and organize lithium positions for the current and previous paths
            initial_position = np.array([path[i][n].position for n, element in enumerate(path[i].symbols) if element == 'H']) 
            final_position = np.array([path[i + t][n].position for n, element in enumerate(path[i + t].symbols) if element == 'H']) 
            initial_positions.append(initial_position)
            final_positions.append(final_position)

            # Displacement vectors
            r = np.abs(initial_position - final_position)
            displacement.append(r)
    
            # Boundary conditions
            condition_1 = r[:, 0] > THRESHOLD
            condition_2 = r[:, 1] > THRESHOLD
            condition_3 = r[:, 2] > THRESHOLD
            condition_4 = r[:, 0] < -THRESHOLD 
            condition_5 = r[:, 1] < -THRESHOLD
            condition_6 = r[:, 2] < -THRESHOLD  
            r[condition_1, 0] -= LATTICE_A
            r[condition_2, 1] -= LATTICE_B
            r[condition_3, 2] -= LATTICE_C
            r[condition_4, 0] += LATTICE_A
            r[condition_5, 1] += LATTICE_B
            r[condition_6, 2] += LATTICE_C
            r2 = np.square(r) # (x(t)- x(t0))2 + (y(t)- y(t0))2 + (z(t)- z(t0))2
            msd = np.mean(r2, axis=0).sum()
            msd_list.append(msd)
            #print(t,msd_list)
            # Calculate the average MSD
            average_msd = np.mean(msd_list)
        average_msd_list.append(average_msd)
        #print(t,len(initial_positions))
        msd_list = []
    return(average_msd_list,time_list)

def write_log(average_msd_list,time_list,output_file):
    rows = list(zip(time_list, average_msd_list))
    with open(output_file, 'w', newline='') as csvfile:
    # Create a CSV writer
        csvwriter = csv.writer(csvfile)

    # Write headers
        csvwriter.writerow(['Time', 'Average MSD'])

    # Write the data
        for time, average_msd in rows:
            formatted_time = '{:.3g}'.format(time)  # format to three significant figures
            formatted_average_msd = '{:.6g}'.format(average_msd)  # format to six significant figures
            csvwriter.writerow([formatted_time, formatted_average_msd])
    print(f"Data has been written to {output_file}")


def main(): 
    args = parse_args()
    file = args.input_file
    output_file = args.output_file
    average_msd_list , time_list = get_msd(file)
    write_log(average_msd_list,time_list,output_file)

if __name__ == "__main__": 
    main()


    
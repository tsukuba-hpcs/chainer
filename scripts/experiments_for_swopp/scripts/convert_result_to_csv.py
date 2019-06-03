import argparse
import glob
import csv

ROOT = '/work/NBB/serihiro/src/chainer/scripts/experiments_for_swopp/resulsts'

def generate_csv(base_directory, column_prefix, output):
    raw_data = {}

    for file in glob.glob(f'{base_directory}/*'): 
        n_process = file.split('/')[-1]
        raw_data[n_process] = []            
        
        with open(file) as f:
            for line in f:
                raw_data[n_process].append(float(line.strip()))
    
    with open(output, mode='w') as f:
        writer = csv.writer(f)
        keys = sorted(map(lambda x: int(x), list(raw_data.keys())))
        
        writer.writerow(list(map(lambda x: f'{column_prefix}{x}', keys)))
        for i in range(10):
            contents = []
            for key in keys:
                contents.append(raw_data[str(key)][i])

            writer.writerow(contents)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['mpi_lustre', 'mpi_ssd', 'pmpi'])
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    if args.mode == 'mpi_lustre':
        generate_csv(f'{ROOT}/evaluate_10times_multiprocess_iterator', 'n_processes_', args.output)
    elif args.mode == 'mpi_ssd':
        generate_csv(f'{ROOT}/evaluate_10times_multiprocess_iterator_ssd', 'n_processes_', args.output)
    else:
        generate_csv(f'{ROOT}/evaluate_10times_prefetch_multiprocess_iterator', 'n_processes_', args.output)

if __name__ == '__main__':
    main()


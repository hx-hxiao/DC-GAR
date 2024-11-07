import os
import subprocess
import glob

def generate_dot_pdg(folder_path, output_folder, joern_path_1):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            file_path = os.path.join(folder_path, filename)
            print(f"Analyzing {file_path}...")

            parse_command = f"{joern_path_1}/joern-parse {file_path}"
            subprocess.run(parse_command, shell=True, check=True)
            joern_path = '/root/joern-cli'
            os.chdir(joern_path)
    
            bin_files_dir = '/main'
            output_dir = '/main/mydata/pkl'
    
            bin_files = glob.glob(os.path.join(bin_files_dir, '*.bin'))    
            for bin_file in bin_files:
                joern_generate_pdg(bin_file, output_dir)


def joern_generate_pdg(bin_file, outdir):
    name = os.path.basename(bin_file).split('.')[0]
    
    output_dir = os.path.join(outdir, name + '_pdg_dir')
    output_dot = os.path.join(outdir, name + '_pdg.dot')
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
    if os.path.exists(output_dot):
        print(f"PDG for {name} already exists, skipping.")
        return 
    # joern_command = f'joern-export {bin_file} --repr pdg --out {output_dir}'
    os.system(f'./joern-export {bin_file} --repr pdg --out {output_dir}')
    # subprocess.run(joern_command, shell=True, check=True)


if __name__ == '__main__':
    main()


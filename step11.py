# Step 11?. Create movement .par files from each run using mcflirt outputs
import os
import argparse 

# initialize parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run mcflirt on fMRI data to obtain motion correction parameters.")
    return parser.parse_args()
# specie will always be H and session 01

# run changes based on input

def main():
    args = parse_arguments()
    
    dataset = 'EmoB'
    task = 'EmoB'  # or 'EmoC'
    specie = 'H'  # 'H' for human, 'D' for dog
    session = '01'
    participants_possible = [range(1, 36)] + [range(37, 41)]
    runs_possible = range(1, 5)  # assuming 4 runs per task

    # define datafolder based on OS
    if os.name == 'nt':  # Windows
        datafolder = r"P:\userdata\raulh87\data"
    else:
        datafolder = os.path.join('/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data')
    # get raw_nifti path from bids folder 
    # #"P:\userdata\raulh87\data\EmoB\BIDS\H-sub-01\H-sub-01_ses-01_task-EmoB_run-01_bold.nii.gz"
    for sub_N in participants_possible:
        for run_N in runs_possible:
            print(f"Processing subject {sub_N}, run {run_N}")
            raw_nifti = os.path.join(
                    datafolder, dataset, 'BIDS',
                    f"{specie}-sub-{sub_N:02d}",
                    f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}_bold.nii.gz"
                )
            mc_path = os.path.join(
                    datafolder, dataset, 'preprocessing', 'H_mcflirt',
                    f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}"
                )
            ## run mcflirt with plots
            # build command
            command_mcflirt = f"mcflirt -in {raw_nifti} -out {mc_path} -plots"
            # run command
            os.system(command_mcflirt)
            print(f"Ran mcflirt on {raw_nifti}, output saved to {mc_path}")

#### run for all subjects and runs#####
if __name__ == "__main__":
    main()
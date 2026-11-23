# Joint leave-one-out RSA on Colab

From the repository root in PowerShell, package the existing pooled crossnobis maps for both configured cohorts. Both ZIPs include the dog and human stimulus classes and the analysis code. Step 1 is not rerun.

```powershell
$python = 'C:\ProgramData\anaconda3\python.exe'
$drive = 'G:\My Drive\rsa_colab'
& $python tools\create_package.py D --loo-group --all-participants --radius 4 --out $drive
& $python tools\create_package.py H --loo-group --all-participants --radius 4 --out $drive
```

Wait for Drive to upload both ZIPs. With the Colab CLI authenticated in WSL, run:

```powershell
.\tools\run_colab_loo.ps1
```

The launcher requests one T4, prompts for Drive access, rebuilds the observed maps and runs 1,000 joint draws per class. Keep the terminal open. Use `-ColabCli` for an explicit executable, `-Runtime CPU` for CPU or `-Reps 10` for a smoke run. The equivalent notebook is `tools/colab_gpu/colab_joint_loo.ipynb`, also inside each ZIP. Select a GPU runtime there and execute both cells. The runner expects the two ZIPs under My Drive/rsa_colab.

Verified batches of 20 draws are archived under `loo_results/<species>_<fingerprint>`. Restart the same command to resume. Each species directory corresponds to its input files, code, seed and requested draw count. After synchronization, use the two exact directories printed by the launcher with `tools/unpack_results.py --replace --dry-run`, then with `--replace`. The launcher also prints the four local inference commands. Import both observed and randomized files before rerunning steps 11.6 through 11.10. The toolbox reads the observed mean from the leave-one-out root and the randomized group draws from its mean subdirectory.

Step 11.4 writes a joint mapping JSON beside the randomized group stem. Step 11.5 uses one label mapping per participant throughout each draw, preserves the spatial relationship across searchlights, and rebuilds all peer means. Local commands use `--reps_group` and `--loo_seed` for both steps. To switch generations at existing local output paths, use `--replace_rnd_files`. Step 11.6 verifies the draw identities before consuming the ensemble. Step 11.15 is a separate class-difference procedure with its original participant-map input format; the joint group archives are for steps 11.6–11.10.


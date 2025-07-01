# Classification Pipeline
Implements AutoGluon and XGBoost for binary and multiclass classification.

## Setting up the python virtual environment
```bash
# Python 3.11.3
python3 -m venv autogluon_cpu_env
source autogluon_cpu_env/bin/activate # activate the virtual environment
pip install --upgrade pip
pip install -r autogluon_cpu_env_requirements.txt 
pip install imblearn
deactivate # deactivate the virtual environment

# If working on the HPCC, set these environment variables within the SLURM script
export LD_LIBRARY_PATH="/mnt/path/to/AG_pipeline/autogluon_cpu_env/lib:$LD_LIBRARY_PATH"
export PATH="/mnt/home/<user>/.local/bin:$PATH"

# If you see "module not found" errors on the HPCC, try specifying the full path to python, e.g.:
python_path="/mnt/path/to/AG_pipeline/autogluon_cpu_env/bin"
${python_path}/python a_classification.py \
    # ... etc
```



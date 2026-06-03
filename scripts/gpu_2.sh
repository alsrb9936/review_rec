
cd ../


export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OMP_WAIT_POLICY=PASSIVE

python run.py model=deepconn data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=2
python run.py model=deepconn data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=2
python run.py model=deepconn data.dataset=Amazon_Musical_Instruments_14 experiment.seed=57 experiment.device=2
python run.py model=deepconn data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2025 experiment.device=2
python run.py model=deepconn data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2026 experiment.device=2

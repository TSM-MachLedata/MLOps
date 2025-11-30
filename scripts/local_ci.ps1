python -m venv .ci-venv
.\.ci-venv\Scripts\activate

pip install -r requirements.txt
pip install "dvc[gcs]"
dvc pull

python src/retrain_if_drift.py
python src/select_champion.py

echo "install environment"

conda env create -f environment.yml

conda activate ntire

echo "Running HDR inference..."

python test_pl2.py

echo "Done."
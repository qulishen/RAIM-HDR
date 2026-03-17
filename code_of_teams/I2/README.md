#### dataset
```
  data
    └─ HDR
        ├─ testdata_phase3
        │   ├─ 001
        │   │   ├─0.jpg
        │   │   ├─1.jpg
        │   │   ├─2.jpg
        │   │   ├─3.jpg
        │   │   ├─4.jpg
        │   ├─002
        │   │   ├─0.jpg
        │   │   ├─1.jpg
        │   │   ├─2.jpg
        │   │   ├─3.jpg
        │   │   ├─4.jpg
        │   └─ ...
        └─ ... (testdata_phase2)
```
#### Install
```bash
conda env create -f environment.yml
```

#### Inference Usage

To run inference with option_3.py, you must manually modify several key arguments in the script. These parameters tell the program which dataset folder to use, which checkpoint to load, which GPU to run on, and where your data is located.

Required parameters to modify (inside option_3.py):

phase
Indicates which HDR dataset sub-folder you want to test.
Example: testdata_phase3

test_ckpt_path
Path to the model weights you want to load for inference.
Example: saved_models/HDR3/xxx.ckpt

gpu
GPU ID to use.
Example: 0 or 1 or 2

data_dir
The directory where your HDR dataset is stored.
Example: ./data/HDR/

dataset
Dataset name.
Example: HDR

Make sure these five arguments are correctly set before running inference.

Running inference

After modifying the above parameters in option_3.py, run the following command:

```bash
python test_pl2.py
```

Train

```bash	
python train_pl3.py
```


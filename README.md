# Antigen-Antibody-Binding-Site-Predictor
The Antigen-Antibody-Binding-Site-Predictor is an improved model based on PIsToN (evaluating Protein Binding Interfaces with Transformer Networks) for predicting antigen-antibody binding sites. It adopts PIsToN's method to generate two-dimensional images of both antigens and antibodies. Feature extraction is performed using antigen and antibody ViT (Vision Transformer) extractors, which are trained through CLIP (Contrastive Language-Image Pretraining). Finally, the model employs a k-NN (k-Nearest Neighbors) algorithm to match these extracted features, enabling accurate prediction of the binding sites.

## Installation

For an optimal setup, we suggest using a conda env ironment to avoid package conflicts. Set it up as follows:

    conda create -n piston python=3.7
    source activate piston
    
    
    pip install -r requirements.txt

You also need to install [MaSIF](https://github.com/LPDI-EPFL/masif) and [FireDock](http://bioinfo3d.cs.tau.ac.il/FireDock).

You can fetch our Singularity container with all pre-configured dependencies:

    wget https://users.cs.fiu.edu/~vsteb002/piston_sif/piston.sif

The details on the singularity definition can be found in the folder [env](./env).

| Note: The container was built with Singularity v3.5.3.

## Usage

PIsToN is designed to evaluate the interfaces of protein complexes. 
It offers two primary modules: "prepare" and "infer".

    piston -h
    usage: PIsToN [-h] [--config CONFIG] {prepare,infer} ...
    
    positional arguments:
      {prepare,infer}
        prepare        Data preparation module
        infer          Inference module
    
    optional arguments:
      -h, --help       show this help message and exit
      --config CONFIG  config file

### Inference module

The "infer" module computes PIsToN scores for protein complexes and visualizes the associated interface maps.

    usage: PIsToN infer [-h] --pdb_dir PDB_DIR [--list LIST] [--ppi PPI] --out_dir
                        OUT_DIR
    
    optional arguments:
      -h, --help         show this help message and exit
      --pdb_dir PDB_DIR  Path to the PDB file of the complex that we need to
                         score.
      --list LIST        Path to the list of protein complexes that we need to
                         score. The list should contain the PPIs in the following
                         format: PID_ch1_ch2, where PID is the name of PDB file,
                         ch1 is the first chain(s) of the protein complex , and
                         ch2 is the second chain(s). Ensure that PID does not
                         contain an underscore
      --ppi PPI          PPI in format PID_A_B (mutually exclusive with the --list
                         option)
      --out_dir OUT_DIR  Directory with output files.

The folder [example](./example) contains an example of running "infer" on two proteins: 6xe1AB-delRBD-100ns and 6xe1AB-wtRBD-100ns.
The proteins correspond to SARS-CoV-2 RBD/antibody complexes (delta variant and wild type) after 100ns of MD simulations
(see the study of [Baral et al.](https://doi.org/10.1016/j.bbrc.2021.08.036) for details).
The output is organized as follows:

- **PIsToN_scores.csv** - PIsToN scores in CSV format (Note: Lower scores indicate better binding).
- **gird_16R** - Directory containing interface maps in numpy format.
- **intermediate_files** - Intermediate files including proteins after prototantion, side-chain refinement, cropping, triangulation, and patch extraction.
- **patch_vis** - HTML files with interactive visualization of interface maps.

The "prepare" module facilitates the pre-computation of interface maps for extensive datasets.
```
    python3 piston.py prepare -h
    usage: piston.py prepare [-h] [--list LIST] [--ppi PPI] [--no_download]
                                [--download_only] [--prepare_docking]
    
    optional arguments:
      -h, --help         show this help message and exit
      --list LIST        List with PPIs in format PID_A_B
      --ppi PPI          PPI in format PID_A_B (mutually exclusive with the --list
                         option)
      --no_download      If set True, the pipeline will skip the download part.
      --download_only    If set True, the program will only download PDB
                         structures without processing them.
      --prepare_docking  If set True, re-dock native structures and pre-
                         process top 100 generated models
```
The script will automatically download complexes from [Protein Data Bank](https://www.rcsb.org/), transform them to a surface,
extract a pair of patches at the interface, compute all features, project it to images (interface maps), and save it as a numpy array.

If processing large datasets, we recommend running
We recommend using an HPC cluster for faster pre-processing.
An example of pre-processing on Slurm can be found in [piston/data/preprocessing_scripts/](../data/preprocessing_scripts/) of this repository.


## Benchmarks

The notebook [PiSToN_test.ipynb](PiSToN_test.ipynb) can be used to replicate the results reported in our paper.

## Training

[training_example](./training_example) provides an instructions of how to train PIsToN.

## Reference



## License





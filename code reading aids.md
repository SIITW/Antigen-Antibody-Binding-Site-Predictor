<h1 align="center"><b>Code Reading Aids</b></h1>

  The uploaded files can achieve two functions:

1. Training the Model

2. Use the model (this part of the code is incomplete and still being modified).

   

## 1 Model Training:

  There are two main blocks, one is to get the interface map from the PDB files, and the second is to train the vision transformer based on the interface map and real samples, the purpose of this part is to train a VIT model, so that the points that can be bound in the antigen and antibody are as similar as possible

### 1.1Get the interface map from the pdb file

If you want to run this part, you can enter the following command in the console in the example folder (or in another folder, just note the path):

```
python .. /piston infer --pdb_dir ./ --list ./training.txt --out_dir ./training
```

*(training.txt is the path to you pdb training lists, out_dir is the patch to your output)*

This part of the code is mainly under the folder data prepare, here is the order in which we read this part of the code and what the function of each part of the code is roughly accomplished (there will be some comments on the function inside the code, but it is not complete)

***.utils/infer*** : The beginning of the execution code

***data_prepare***: Batch processing of input PDB files, mainly read the get_process function, there are some functions we don't use, such as prepare_docking

***get\_ structure***: This code mainly implements the following main functions: protonate the PDB file (hydrogenation atoms), download the PDB file, get the atomic coordinates, and crop the PDB (we did not clip ，all the atoms are used in this step).

*triangulate:* This code uses triangulation to simulate the surface of a protein molecule, and uses triangular vertices to characterize the triangulation surface

***compute\_ patches**:* This is done by dividing the patches and calculating which points to include in the patch (in this case, we need to include all the points).

***map_to_atoms*** : In the previous use, the vertices of the triangle are used to represent the triangle to simulate the surface of the protein molecule, but the vertex is a series of virtual points, and does not have some physical and chemical properties that the protein atom should have, we map the characteristics of the heavy atom (non-hydrogen atom) closest to each vertex to the vertex, and after such an operation, the vertex will also have some physical and chemical properties like the atom

***convert_to_images*** : The main task is to map vertices in the 3D plane to the interface map of the 2D plane. The main idea is to use a multi-dimensional scaling algorithm to map points on a 3D plane to a 2D plane, and then use KD Tree in Python for search and feature fusion

### 1.2 Use the interface map and real-world samples to train the Vision Transformer(this part of the code is just a preliminary version).

***Definition of a real sample***: In a PDB complex, if the atom in an antigen is less than 4.5 away from an atom in the antibody, it is considered to be so

***The purpose of the training***: we want to train a network, input the antigen and antibody interface map separately, and get a vector representing the antigen and antibody respectively, in this vector, the more similar the features of the points that can be bound, the better

#### Code:

***training_example/self_train.ipynb*** : There are several parts to this code: load the data, calculate the mean and standard deviation of each eigenvalue, normalize and scale the data, define the data loading class, define the network model, and define each training process

***training_example/data_preparation/07-grid*:** stores the data of the training and validation sets



## 2 Using the model(this part of the code has not yet been integrated)

### 3 Some other codes

  ***env:*** configuration environment related code

  ***example:*** The first part runs the example

  ***get_label:*** code that calculates the true epitope

  ***networks**:* The definition of networks

  ***saved_models**:* Pre-trained parameters saved in the original paper

  ***utils**:* Loads some utility classes, such as dataloader and process


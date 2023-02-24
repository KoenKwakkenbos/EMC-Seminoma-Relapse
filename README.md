# Relapse Prediction in Clinical Stage I Seminomas
## TM2 Internship 2023 (Erasmus MC Department of Urology)

Koen Kwakkenbos, Feb 2022

This repository contains the scripts to reproduce the results in the report `Prediction of Metastatic Disease Relapse in
Clinical Stage I Seminoma Testis through Deep Learning`.

### Contents
- `helpers.py`: contains two functions used for creating and checking validity of directories.
- `tile_generation.py`: used for extracting and saving tiles from seminoma regions.
- `datagenerator.py`: defines the three different types of datagenerators used in the experiments.
- `models.py`: contains the three different types of models.
- `train_[autoencoder/classifier/MIL].py`: these scripts are used for actually running the experiments.

### Usage
All runnable scripts (`tile_generation` and the `train` scripts) can be used in combination with command line arguments. 
To reproduce the tiling, a specific folder structure should be used: 
```
├── Database
│   ├── Annotations
│   │   ├── Project 1 patient 1 tm 10
│   │   ├── Project 2 patient 11 tm 20
│   ├── Converted and anonymized
│   │   ├── Project 1 patient 1 tm 10
│   │   ├── Project 2 patient 11 tm 20
```

The structure in the example above reflects the way the database is set up currently (The patients are divided in smaller projects). The keywords 'Annotations' and 'Converted and anonymized' are hard-coded in the script. The root directory (Database) can be given through the command line input. The folders in 'Annotations' and 'Converted and anonymized' can have other names, as long as they are identical in both folders.
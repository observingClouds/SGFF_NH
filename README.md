# Neural network classifications of mesoscale patterns of shallow convection

## Download repository and data

This repository uses [dvc](dvc.org) to version the data. To work with the recent version of data please execute the following:
```shell
git clone git@github.com:observingClouds/SGFF_NH.git
dvc pull
```

To reproduce results the environment can be created with
```
mamba env install -n sgff -f environment.yml
```
and analysis steps rerun by simply calling
```
dvc repro
```
or
```
dvc repro patterns_along_traj
```
to only reproduce specific steps which are listed in `dvc.yaml`

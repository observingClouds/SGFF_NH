# Neural network classifications of mesoscale patterns of shallow convection

## Download repository and data

This repository uses [dvc](dvc.org) to version the data. To work with the recent version of data please execute the following:
```shell
git clone git@github.com:observingClouds/SGFF_NH.git
dvc pull
```

To reproduce results the environment can be created with
```
mamba env create -n sgff -f environment.yml
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


## Potential issues

### Error when running stage `stitch_images_TB`

```
convert-im6.q16: width or height exceeds limit `panorama-in.jpg' @ error/cache.c/OpenPixelCache/3802.
```

To solve it you will need to edit `/etc/ImageMagick-6/policy.xml` and increase the limit for memory, width, height and area. For example:

```
  <policy domain="resource" name="memory" value="8GiB"/>
  <policy domain="resource" name="width" value="128KB"/>
  <policy domain="resource" name="height" value="128KB"/>
  <policy domain="resource" name="area" value="8GB"/>
```

source: https://www.guyrutenberg.com/tag/imagemagick/


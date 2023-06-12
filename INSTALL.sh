curl micro.mamba.pm/install.sh | bash
source ~/.bashrc
micromamba create -f environment.yml
source /home/azureuser/micromamba/envs/schulz_et_al_2023/bin/activate

# Mount NFS storage
sudo apt-get -y update
sudo apt-get install nfs-common
sudo mkdir -p /mount/sgff/nfs
sudo mount -t nfs sgff.file.core.windows.net:/sgff/nfs /mount/sgff/nfs -o vers=4,minorversion=1,sec=sys

## Create Azure resources

### Create virtual machine
- Select subscription
- create new resource group, e.g. SGFF_MODIS_IO
- name virtual machine, e.g. download-machine
- otherwise all standard setup --> create

### Create NFS file storage for cheaper file storage
(https://learn.microsoft.com/en-us/azure/storage/files/storage-files-quick-create-use-linux)

### Create storage account

![](https://pad.gwdg.de/uploads/dbf40416-a82b-4136-97a4-bc2e7dfa9bd8.png)

- otherwise standard setup --> create

### Create the actual disk

![](https://pad.gwdg.de/uploads/c8c37f2b-0374-4d1c-977f-e5aa7d8081c5.png)

--> create

- disable secure transfer
![](https://pad.gwdg.de/uploads/df5c6ebd-2102-43f2-8eec-33b52bfeda17.png)

### Connect NFS with VM


- login to virutal machine
- follow instruction on how to connect
![](https://pad.gwdg.de/uploads/9bb18595-f988-41c0-bbcf-b525c749d9da.png)

### Deploy code
- generate ssh-key on vm and add it to github
- clone git repo to mount volume
- install miniforge to get mamba commands

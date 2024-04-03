import json

file_path = ".dvc/cache/files/md5/28/ecdbd59fcddc4de087cb9e49f5b7e5.dir"

with open(file_path) as file:
    data = json.load(file)

fsspec_ref_dict = {}
prefix_protocol = "s3://sgff"
prefix_path = "files/md5"

# Access the parsed JSON data
for item in data:
    md5 = item["md5"]
    relpath = item["relpath"]
    if ".zarr" in relpath:
        relpath = "/".join(relpath.split("/")[1:])
    fsspec_ref_dict[relpath] = [
        ("/").join([prefix_protocol, prefix_path, md5[:2], md5[2:]])
    ]

# Write fsspec_ref_dict to JSON file
output_file_path = "catalog/references/fsspec_ref_28ecdbd59fcddc4de087cb9e49f5b7e5.json"
with open(output_file_path, "w") as output_file:
    json.dump(fsspec_ref_dict, output_file)

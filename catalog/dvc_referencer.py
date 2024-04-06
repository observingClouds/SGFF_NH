import json

file_path = "../.dvc/cache/41/f03e31bf8d880a5a8723b33999db8e.dir"

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
output_file_path = "references/fsspec_ref_41f03e31bf8d880a5a8723b33999db8e.json"
with open(output_file_path, "w") as output_file:
    json.dump(fsspec_ref_dict, output_file)

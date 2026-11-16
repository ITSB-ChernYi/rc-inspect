"""Data root shared by the dataset builders and the training script.

Set the environment variable OBJ1DATA to use another location (e.g. C:/obj1data on a Windows
training PC); the default is the cloud workspace. Paths use forward slashes on every OS.
"""
import hashlib
import os

DATA = os.environ.get("OBJ1DATA", "/home/user/obj1data").replace("\\", "/").rstrip("/")


def label_digest(root, split):
    """sha256 over the sorted label files of one split (stem + content, line endings normalised).

    Complements the stem hash in split_meta.json: two builds agree on this digest only if every
    image of the split has byte-identical YOLO labels.
    """
    d = f"{root}/labels/{split}"
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        h.update(f.encode())
        h.update(open(f"{d}/{f}", "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def dataset_root(rec):
    """Folder of the dataset a run was trained on, under the current data root.

    run_record.json stores the data.yaml path of the machine and folder the run was trained in; only the
    dataset folder name (e.g. neu_s0) is taken from it, so records stay valid after the data root moves.
    """
    ds = rec["args"]["data"].replace("\\", "/").rstrip("/").split("/")[-2]
    root = f"{DATA}/{ds}"
    assert os.path.exists(f"{root}/data.yaml"), f"dataset {ds} not found under {DATA}"
    return root

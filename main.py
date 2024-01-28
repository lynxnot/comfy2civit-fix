"""ComfyUI metadata fixer from civit.ai"""

from dataclasses import dataclass, field
import hashlib
import json
from os import path

import pyjq
from PIL import Image

# from PIL.PngImagePlugin import PngInfo

TEST_FILE = "test-with-loras.png"

BASE_MODELS_FOLDERS = [
    "/mnt/d/StableDiffusion/instances/instance_1/ComfyUI/models",
    "/mnt/d/StableDiffusion/models/",
]


@dataclass
class Resource:
    """A ComfyUI resource, typically a file in one of the models directories"""

    type: str
    name: str
    path: str = field(default="", repr=False)
    hash: str = field(default="", repr=True)


def compute_sha256(file_path: str) -> str:
    """compute sha256 chunked"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_or_compute_hash(file_path: str) -> str:
    """get hash from cache or compute"""
    hash_cache_file = f"{path.splitext(file_path)[0]}.sha256sum"

    if path.isfile(hash_cache_file):
        with open(hash_cache_file, "r", encoding="utf-8") as f:
            sha256sum = f.readline()
    else:
        sha256sum = compute_sha256(file_path)
        with open(hash_cache_file, "w", encoding="utf-8") as f:
            f.write(sha256sum)

    return sha256sum


def resource_hash(res: Resource) -> Resource:
    """determine full path of a resource"""
    res.name = res.name.replace("\\", "/")
    for base_path in BASE_MODELS_FOLDERS:
        test_path = path.join(base_path, res.type, res.name)
        if path.isfile(test_path):
            res.path = test_path
            break

    res.hash = get_or_compute_hash(res.path)

    return res


def checkpoint_scan(prompt: dict) -> list[Resource]:
    """returns stuff"""
    # basic nodes
    jq = ".[].inputs.ckpt_name | values"
    checkpoints = [Resource("checkpoints", x) for x in pyjq.all(jq, prompt)]

    # full paths
    checkpoints = [resource_hash(x) for x in checkpoints]
    return checkpoints


def main():
    """main f"""
    with Image.open(TEST_FILE) as im:
        prompt = json.loads(im.info.get("prompt"))
        # workflow = json.loads(im.info.get("workflow"))
        print(checkpoint_scan(prompt))


if __name__ == "__main__":
    main()

# nodes = []
# if prompt is not None:
#    print(json.dumps(prompt, ensure_ascii=False, indent=4))
#    for idx, node in prompt.items():
#        o = {
#            'idx': idx,
#            'class_type': node.get("class_type"),
#            **node.get("inputs")
#        }
#        nodes.append(o)
#
# print(json.dumps(nodes, indent=4))


# if workflow is not None:
#    print(json.dumps(workflow, ensure_ascii=False, indent=4))

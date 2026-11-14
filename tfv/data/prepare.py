import io
import shutil
import time
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import requests
import torchvision.transforms as transforms
from datasets import Dataset, Features, Image, Sequence, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image as PILImage

from tfv.config import PrepareConfig

DATASETS_SERVER_SIZE_URL = "https://datasets-server.huggingface.co/size"

def resize_and_encode(image: dict[str, Any], image_size: int, jpeg_quality: int) -> bytes:
    """
    Decodes one stored image, converts it to RGB and resizes it exactly as training does at load
    time (VQADataset converts non-RGB to RGB, then transforms.Resize((size, size))), then encodes
    it as JPEG. Resizing again at train time to the same size is a no-op, so the only difference
    from the original images is the JPEG compression.

    Args:
        image: An undecoded datasets Image value, {"bytes": bytes | None, "path": str | None}.
        image_size: Output height and width in pixels.
        jpeg_quality: JPEG quality (1-95).
    Returns:
        bytes: The JPEG-encoded resized image.
    """
    source = io.BytesIO(image["bytes"]) if image["bytes"] is not None else image["path"]
    pil_image = PILImage.open(source)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    pil_image = transforms.Resize((image_size, image_size))(pil_image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=jpeg_quality)

    return buffer.getvalue()

def process_example(example: dict[str, Any], image_size: int, jpeg_quality: int) -> dict[str, Any]:
    """
    Keeps only what VQADataset uses: the first image (resized, JPEG) and the first QA pair.

    An image that fails to decode is dropped (images = []), which VQADataset turns into its
    all-zeros fallback image.

    Args:
        example: A Cauldron row with "images" (list of undecoded images) and "texts" (list of
            {"user", "assistant", "source"} dicts).
        image_size: Output height and width in pixels.
        jpeg_quality: JPEG quality (1-95).
    Returns:
        dict[str, Any]: {"images": [] or [{"bytes", "path"}], "texts": the first QA pair as a list}.
    """
    images = []
    if example["images"]:
        try:
            images = [{"bytes": resize_and_encode(example["images"][0], image_size, jpeg_quality), "path": None}]
        except Exception as error:
            print(f"Dropping undecodable image: {error}")

    return {"images": images, "texts": example["texts"][:1]}

def count_missing_images(dataset: Dataset) -> int:
    """
    Counts rows whose image list is empty, without decoding any image.

    Args:
        dataset: A processed subset.
    Returns:
        int: Number of rows with no image.
    """
    lengths = pc.list_value_length(dataset.data.column("images"))
    return int(pc.sum(pc.equal(lengths, 0)).as_py() or 0)

def finished_subsets(repo_id: str) -> set[str]:
    """
    Lists the subsets already pushed to the target dataset repo, so a rerun can skip them.

    Args:
        repo_id: Target HF dataset repo.
    Returns:
        set[str]: Subset names that already have data files in the repo (empty if no repo yet).
    """
    api = HfApi()
    if not api.repo_exists(repo_id, repo_type="dataset"):
        return set()
    
    files = api.list_repo_files(repo_id, repo_type="dataset")
    return {path.split("/")[0] for path in files if path.endswith(".parquet") and "/" in path}

def prepare_subset(name: str, cfg: PrepareConfig, num_proc: int) -> dict[str, Any]:
    """
    Downloads one Cauldron subset, processes every row, pushes it to the target repo as config
    `name` (so load_dataset(target_repo_id, name) works like loading the original), then
    deletes the local cache so disk usage peaks at about one subset.

    Args:
        name: Cauldron subset (config) name.
        cfg: Preprocessing settings.
        num_proc: Worker processes for the map.
    Returns:
        dict[str, Any]: {"rows", "missing_images", "seconds"} for this subset.
    """
    start = time.monotonic()
    cache_dir = Path(cfg.work_dir) / name

    dataset = load_dataset(cfg.source_path, name, split="train", cache_dir=str(cache_dir))
    features = Features({"images": Sequence(Image()), "texts": dataset.features["texts"]})
    dataset = dataset.cast_column("images", Sequence(Image(decode=False)))
    dataset = dataset.map(
        process_example,
        fn_kwargs={"image_size": cfg.image_size, "jpeg_quality": cfg.jpeg_quality},
        remove_columns=dataset.column_names,
        features=features,
        num_proc=num_proc,
        writer_batch_size=200,
        desc=name,
    )
    rows, missing = len(dataset), count_missing_images(dataset)

    dataset.push_to_hub(
        cfg.target_repo_id,
        config_name=name,
        split="train",
        private=cfg.private,
        max_shard_size=cfg.max_shard_size,
    )

    del dataset
    shutil.rmtree(cache_dir, ignore_errors=True)

    return {"rows": rows, "missing_images": missing, "seconds": time.monotonic() - start}

def fetch_subset_sizes(source_path: str) -> dict[str, dict[str, int]]:
    """
    Gets the row count and parquet size of every subset from the HF datasets server.

    Args:
        source_path: Source HF dataset repo.
    Returns:
        dict[str, dict[str, int]]: {subset: {"rows": int, "bytes": int}}.
    """
    response = requests.get(DATASETS_SERVER_SIZE_URL, params={"dataset": source_path}, timeout=30)
    response.raise_for_status()
    
    configs = response.json()["size"]["configs"]
    return {c["config"]: {"rows": c["num_rows"], "bytes": c["num_bytes_parquet_files"]} for c in configs}

def benchmark_processing(name: str, cfg: PrepareConfig, rows: int) -> dict[str, float]:
    """
    Times process_example on the first `rows` rows of a subset, on one process. Rows are
    streamed, so the subset is not downloaded, and fetching them is excluded from the timing.

    Args:
        name: Cauldron subset name.
        cfg: Preprocessing settings.
        rows: Number of rows to time.
    Returns:
        dict[str, float]: {"sec_per_row", "sec_per_input_mb", "output_bytes_per_row"}.
    """
    stream = load_dataset(cfg.source_path, name, split="train", streaming=True)
    stream = stream.cast_column("images", Sequence(Image(decode=False)))
    examples = list(stream.take(rows))

    input_bytes, output_bytes = 0, 0
    start = time.perf_counter()
    for example in examples:
        processed = process_example(example, cfg.image_size, cfg.jpeg_quality)
        input_bytes += sum(len(image["bytes"] or b"") for image in example["images"][:1])
        output_bytes += sum(len(image["bytes"]) for image in processed["images"])
    seconds = time.perf_counter() - start

    return {
        "sec_per_row": seconds / len(examples),
        "sec_per_input_mb": seconds / max(input_bytes / 1e6, 1e-9),
        "output_bytes_per_row": output_bytes / len(examples),
    }

def benchmark_download(source_path: str, subset: str, work_dir: str) -> float:
    """
    Measures download throughput by fetching the first parquet file of one subset.

    Args:
        source_path: Source HF dataset repo.
        subset: Subset whose first parquet file is downloaded.
        work_dir: Scratch directory; the file is deleted afterwards.
    Returns:
        float: Throughput in MB/s.
    """
    files = sorted(
        path for path in HfApi().list_repo_files(source_path, repo_type="dataset")
        if path.startswith(f"{subset}/") and path.endswith(".parquet")
    )
    target_dir = Path(work_dir) / "download_benchmark"
    start = time.perf_counter()
    local_path = hf_hub_download(source_path, files[0], repo_type="dataset", local_dir=str(target_dir))
    seconds = time.perf_counter() - start
    size_mb = Path(local_path).stat().st_size / 1e6
    shutil.rmtree(target_dir, ignore_errors=True)
    return size_mb / seconds

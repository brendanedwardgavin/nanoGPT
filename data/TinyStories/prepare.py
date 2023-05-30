import os
import contextlib
import itertools
from tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

# download the tiny shakespeare dataset
from pathlib import Path
cur_path = Path(__file__).parent.parent.parent.parent.parent.resolve()

tokenizer = Tokenizer.from_file("./tokenizer-TinyStories.json")
split_text = "<|endoftext|>"
def tokenize_datum(datum):
    return tokenizer.encode(datum).ids + tokenizer.encode(split_text).ids

def tokenize_text_file(filepath: Path, name=None):
    
    name = filepath.stem() if name is None else name
    print(f"reading {name}")
    data = filepath.read_text()
    data_list = data.split(split_text)
    print(f"tokenizing {name}")

    # about 4x faster:
    with tqdm_joblib(tqdm(desc=name, total=len(data_list))) as progress_bar:
        ids = Parallel(n_jobs=-1)(delayed(tokenize_datum)(datum) for datum in data_list)

    # slower:
    # ids = [tokenize_datum(datum) for datum in tqdm(data_list)]

    ids = list(itertools.chain.from_iterable(ids))
    print(f"{name}: {len(ids)} tokens")
    print(f"converting {name} to uint16")
    ids = np.array(ids, dtype = np.uint16)
    print(f"saving {name}")
    ids.tofile(str(Path(__file__).parent.resolve() / f"{name}.bin"))

train_input_file_path = cur_path / "datasets/TinyStories/TinyStories-train.txt"
val_input_file_path = cur_path / "datasets/TinyStories/TinyStories-valid.txt"

tokenize_text_file(val_input_file_path, name="val")
tokenize_text_file(train_input_file_path, name="train")
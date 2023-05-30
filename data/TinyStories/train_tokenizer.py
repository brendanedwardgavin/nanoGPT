import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# following https://huggingface.co/docs/tokenizers/quicktour
# and https://stackoverflow.com/a/76058017

# https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE

import pathlib
cur_path = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()

train_input_file_path = cur_path / "datasets/TinyStories/TinyStories-train.txt"
val_input_file_path = cur_path / "datasets/TinyStories/TinyStories-valid.txt"
files = [str(train_input_file_path), str(val_input_file_path)]

print(files)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "<|endoftext|>"])
tokenizer.train(files, trainer)

tokenizer.save("./tokenizer-TinyStories.json")
#tokenizer = Tokenizer.from_file("./tokenizer-TinyStories.json")
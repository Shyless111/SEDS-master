from transformers import AutoTokenizer


class HFTokenizerAdapter:
    """Adapter exposing the minimal tokenizer API expected by the dataloaders.

    The existing loaders manually prepend/append CLIP-style special tokens before
    calling `convert_tokens_to_ids`. This adapter remaps those placeholders to
    the underlying Hugging Face tokenizer special tokens so we can swap in a
    BERT-family tokenizer without rewriting every loader.
    """

    _SPECIAL_MAP = {
        "<|startoftext|>": "cls_token",
        "<|endoftext|>": "sep_token",
        "[PAD]": "pad_token",
        "[UNK]": "unk_token",
        "[MASK]": "mask_token",
    }

    def __init__(self, pretrained_path):
        self.base_tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=True)

    def tokenize(self, text):
        return self.base_tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        mapped_tokens = []
        for token in tokens:
            if token in self._SPECIAL_MAP:
                mapped = getattr(self.base_tokenizer, self._SPECIAL_MAP[token], token)
                mapped_tokens.append(mapped if mapped is not None else token)
            else:
                mapped_tokens.append(token)
        return self.base_tokenizer.convert_tokens_to_ids(mapped_tokens)

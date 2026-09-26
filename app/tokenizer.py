class CharacterTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))

        self.chars = chars
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens):
        return "".join(self.itos[i] for i in tokens)

    def save(self, path: str):
        import json

        data = {
            "chars": self.chars
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls("")

        tokenizer.chars = data["chars"]
        tokenizer.vocab_size = len(tokenizer.chars)
        tokenizer.stoi = {
            ch: i for i, ch in enumerate(tokenizer.chars)
        }
        tokenizer.itos = {
            i: ch for i, ch in enumerate(tokenizer.chars)
        }

        return tokenizer
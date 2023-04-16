import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade


class SpladeWrapper:
    def __init__(self, model_name: str, agg: str = "mean"):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.agg = agg

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_hidden_states = outputs[0]
            if self.agg == "mean":
                return torch.mean(last_hidden_states, dim=1).squeeze().numpy()
            elif self.agg == "max":
                return torch.max(last_hidden_states, dim=1).values.squeeze().numpy()
            else:
                raise ValueError(f"Aggregation method {self.agg} not supported")

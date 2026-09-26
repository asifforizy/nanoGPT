import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import BigramLanguageModel
from app.tokenizer import CharacterTokenizer


# -------------------------
# Configuration
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 32

n_embd = 64
n_head = 4
n_layer = 4

dropout = 0.0

MODEL_PATH = "checkpoints/nanoGPT.pth"

TOKENIZER_PATH = "checkpoints/tokenizer.json"


# -------------------------
# FastAPI
# -------------------------

app = FastAPI(
    title="NanoGPT API",
    description="API for generating text using NanoGPT",
    version="1.0.0"
)


# -------------------------
# Request / Response
# -------------------------

class GenerateRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        description="Text prompt"
    )

    max_new_tokens: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of new tokens"
    )


class GenerateResponse(BaseModel):

    prompt: str

    response: str

    max_new_tokens: int


# -------------------------
# Load tokenizer
# -------------------------

if not os.path.exists(TOKENIZER_PATH):

    raise RuntimeError(
        "Tokenizer not found. "
        "Run: python -m app.train"
    )


tokenizer = CharacterTokenizer.load(
    TOKENIZER_PATH
)


# -------------------------
# Load model
# -------------------------

if not os.path.exists(MODEL_PATH):

    raise RuntimeError(
        "Model checkpoint not found. "
        "Run: python -m app.train"
    )


model = BigramLanguageModel(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


print(f"NanoGPT loaded on: {device}")


# -------------------------
# Routes
# -------------------------

@app.get("/")
def root():

    return {
        "success": True,
        "message": "NanoGPT API is running"
    }


@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "device": device
    }


@app.post(
    "/generate",
    response_model=GenerateResponse
)
def generate(request: GenerateRequest):

    prompt = request.prompt.strip()

    if not prompt:

        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )

    # Keep only characters known by the tokenizer.
    unknown_characters = [
        character
        for character in prompt
        if character not in tokenizer.stoi
    ]

    if unknown_characters:

        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt contains characters "
                "that were not present in the "
                "training dataset: "
                + "".join(
                    sorted(
                        set(unknown_characters)
                    )
                )
            )
        )

    encoded_prompt = tokenizer.encode(
        prompt
    )

    if not encoded_prompt:

        raise HTTPException(
            status_code=400,
            detail="Prompt could not be encoded"
        )

    context = torch.tensor(
        [encoded_prompt],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        generated_tokens = model.generate(
            context,
            max_new_tokens=request.max_new_tokens
        )

    generated_text = tokenizer.decode(
        generated_tokens[0].tolist()
    )

    # Remove the original prompt from the response.
    response = generated_text[
        len(prompt):
    ]

    return GenerateResponse(
        prompt=prompt,
        response=response,
        max_new_tokens=request.max_new_tokens
    )
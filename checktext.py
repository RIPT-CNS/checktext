from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
from spellchecker import SpellChecker
import warnings
from gramformer import Gramformer
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from fastapi.middleware.cors import CORSMiddleware
warnings.filterwarnings('ignore', category=FutureWarning)

app = FastAPI()
origins = [
"*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Initialize the models
spell = SpellChecker()
gramformer = Gramformer(models=1, use_gpu=False)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str
    reference_text: str = None

class TextResponse(BaseModel):
    errors: List[Dict[str, Any]]
    corrections: List[str]
    comments: str
    similarity_score: float = None
    is_similar: bool = None

async def check_spelling(text: str):
    words = []
    for word in text.split():
        stripped_word = word.strip(string.punctuation)
        if stripped_word:
            words.append({
                'original': word,
                'stripped': stripped_word,
            })
    
    errors = []
    corrections = []
    
    for word_info in words:
        stripped_word = word_info['stripped']
        if stripped_word and stripped_word in spell.unknown([stripped_word]):
            correction = spell.correction(stripped_word)
            if correction and correction != stripped_word:
                original_word = word_info['original']
                prefix = ''
                suffix = ''
                
                while original_word and original_word[0] in string.punctuation:
                    prefix += original_word[0]
                    original_word = original_word[1:]
                while original_word and original_word[-1] in string.punctuation:
                    suffix = original_word[-1] + suffix
                    original_word = original_word[:-1]
                
                final_correction = prefix + correction + suffix
                
                errors.append({
                    "type": "spelling",
                    "original": word_info['original'],
                    "suggestion": final_correction
                })
                corrections.append(final_correction)
    
    return errors, corrections

def preprocess_text_for_grammar(text: str):
    """Remove final period if it exists and return both processed text and whether period was removed"""
    text = text.strip()
    had_final_period = text.endswith('.')
    if had_final_period:
        text = text[:-1]
    return text, had_final_period

def postprocess_grammar_correction(correction: str, had_final_period: bool):
    """Add back final period if it was in original text"""
    if had_final_period and not correction.endswith('.'):
        correction = correction + '.'
    return correction

async def check_grammar(text: str):
    # Preprocess text by removing final period if it exists
    processed_text, had_final_period = preprocess_text_for_grammar(text)
    
    corrections = gramformer.correct(processed_text, max_candidates=1)
    
    if not corrections:
        return [], []
    
    corrections_list = list(corrections)
    if not corrections_list:
        return [], []
        
    correction = corrections_list[0]
    
    # Add back period if it was in original text
    correction = postprocess_grammar_correction(correction, had_final_period)
    
    # Only return error if there are changes other than final punctuation
    if processed_text.lower() == correction.lower().rstrip('.'):
        return [], []
    
    errors = [{
        "type": "grammar",
        "original": text,
        "suggestion": correction
    }]
    
    return errors, [correction]

async def check_embedding_similarity(text: str, reference_text: str):
    # Generate embeddings
    text_embedding = embedding_model.encode([text])[0]
    ref_embedding = embedding_model.encode([reference_text])[0]
    
    # Calculate cosine similarity
    similarity = 1 - cosine(text_embedding, ref_embedding)
    
    # Check if similarity is above threshold (0.8)
    is_similar = similarity > 0.8
    
    return float(similarity), is_similar

@app.post("/checktext", response_model=TextResponse)
async def check_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    # Run checks concurrently
    tasks = [
        check_spelling(request.text),
        check_grammar(request.text)
    ]
    
    # Add embedding check if reference text is provided
    embedding_results = None
    if request.reference_text:
        similarity_score, is_similar = await check_embedding_similarity(
            request.text, 
            request.reference_text
        )
    
    # Gather results
    spell_results, gram_results = await asyncio.gather(*tasks)
    
    # Combine results
    all_errors = spell_results[0] + gram_results[0]
    all_corrections = spell_results[1] + gram_results[1]
    
    # Generate comment
    comment = ""
    if all_errors:
        comment = "Found the following issues:\n"
        for error in all_errors:
            if error["type"] == "spelling":
                comment += f"- Spelling error: '{error['original']}' should be '{error['suggestion']}'\n"
            else:
                comment += f"- Grammar error: Text should be: '{error['suggestion']}'\n"
    else:
        comment = "No spelling or grammar errors found."
    
    return TextResponse(
        errors=all_errors,
        corrections=all_corrections,
        comments=comment,
        similarity_score=similarity_score if request.reference_text else None,
        is_similar=is_similar if request.reference_text else None
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
import model_api

model, dataset = model_api.load_model()
app = FastAPI()


@app.get("/translate")
def translate(english_sentence: str = None):
    if english_sentence is None:
        return {"error": "Required query parameter 'english_sentence' is missing"}

    gloss_translation = model_api.translate(model, dataset, english_sentence)

    return {"english_sentence": english_sentence, "gloss_translation": gloss_translation}

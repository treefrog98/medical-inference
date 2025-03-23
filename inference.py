from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Define request model with updated input schema
class TextRequest(BaseModel):
    clinical_note: str

# Define output model for response schema
class TextResponse(BaseModel):
    diagnoses: list[str]

MODEL_NAME = "microsoft/BioGPT-Large"
inference_pipeline = pipeline("text-generation", model=MODEL_NAME, temperature=0.1)

# Initialize FastAPI
app = FastAPI()

@app.post("/predict", response_model=TextResponse)
def predict(request: TextRequest):
    try:
        response = inference_pipeline(f"You are a diagnostic assistant trying to figure out what all the problems a patient is having. Please extract diagnoses as a comma-separated list from this clinical note: {request.clinical_note}.", max_length=1000)
        # Split the comma-separated list into a list of strings
        diagnoses_list = [diagnosis.strip() for diagnosis in response[0]['generated_text'].split(',')]
        return TextResponse(diagnoses=diagnoses_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from langchain import PromptTemplate, OpenAI, LLMChain
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import json

load_dotenv()

# build fastapi backend
app = FastAPI()
origins = ['https://learn.canvas.net']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Request(BaseModel):
    question: str
    answers: List[str]

@app.get("/")
async def health_check():
    return {"status": "success"}

@app.post("/api")
async def ask_question(request: Request):
    question = request.question
    answers = request.answers

    template = """You're a helpful assistant for customers to make decisions. Once problem is provided, you need to select correct answers.
    Question: {question}
    Possible Answers: {answers}
    Return in json format with indexes of "answers" with array of numbers(starting from 1, in order they've provided) and "reason" with string
    """
    prompt = PromptTemplate(template=template, input_variables=["question", "answers"])
    llm_chain = LLMChain(prompt=prompt, llm = OpenAI(temperature=0,model_name="gpt-3.5-turbo"))

    print(question)
    print(','.join(answers))

    result = llm_chain.predict(question=question, answers=','.join(answers))

    return json.loads(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
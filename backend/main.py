from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from backend.predict_model import predict

app = FastAPI()

templates = Jinja2Templates(directory="backend/templates")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(
    request: Request,
    age: int = Form(...),
    job: str = Form(...),
    marital: str = Form(...),
    education: str = Form(...),
    default: str = Form(...),
    balance: int = Form(...),
    housing: str = Form(...),
    loan: str = Form(...),
    contact: str = Form(...),
    day: int = Form(...),
    month: str = Form(...),
    duration: int = Form(...),
    campaign: int = Form(...),
    pdays: int = Form(...),
    previous: int = Form(...),
    poutcome: str = Form(...),
):
    data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }

    result = predict(data)

    return templates.TemplateResponse("form.html", {"request": request, "result": result})
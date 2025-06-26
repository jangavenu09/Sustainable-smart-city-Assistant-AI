pip install fastapi uvicorn gradio transformers torch pandas numpy scikit-learn python-multipart nest-asyncio pyngrok python-dotenv pydantic

# Sustainable Smart City Assistant AI
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import nest_asyncio
from datetime import datetime
import io
import base64

# FastAPI and related imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Gradio for UI
import gradio as gr

# ML and AI imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Enable nested asyncio for Colab
nest_asyncio.apply()

# Initialize the IBM Granite model
class GraniteModel:
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.3-2b-instruct"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.load_model()

    def load_model(self):
        try:
            print("Loading IBM Granite model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a smaller model if Granite fails
            self.pipeline = pipeline("text-generation", model="gpt2", max_length=256)

    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        try:
            if self.pipeline:
                response = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
                return response[0]['generated_text'][len(prompt):].strip()
            return "Model not available"
        except Exception as e:
            return f"Error generating text: {str(e)}"

# Initialize model
granite_model = GraniteModel()

# Pydantic models for API
class CitizenFeedback(BaseModel):
    name: str
    email: str
    category: str
    message: str
    location: Optional[str] = None

class ChatMessage(BaseModel):
    message: str

class EcoTipsRequest(BaseModel):
    keywords: str

# Smart City Assistant Class
class SmartCityAssistant:
    def __init__(self):
        self.feedback_data = []
        self.kpi_data = {}
        self.policies = []

    def summarize_document(self, text: str) -> str:
        """Summarize policy documents using Granite LLM"""
        prompt = f"""Please provide a concise, citizen-friendly summary of the following policy document:

{text}

Summary:"""
        return granite_model.generate_text(prompt, max_length=300)

    def process_citizen_feedback(self, feedback: CitizenFeedback) -> Dict:
        """Process and categorize citizen feedback"""
        feedback_entry = {
            "id": len(self.feedback_data) + 1,
            "timestamp": datetime.now().isoformat(),
            "name": feedback.name,
            "email": feedback.email,
            "category": feedback.category,
            "message": feedback.message,
            "location": feedback.location,
            "status": "Open"
        }
        self.feedback_data.append(feedback_entry)
        return {"status": "success", "message": "Feedback submitted successfully", "id": feedback_entry["id"]}

    def generate_eco_tips(self, keywords: str) -> str:
        """Generate eco-friendly tips based on keywords"""
        prompt = f"""Generate 3-5 practical, actionable eco-friendly tips related to: {keywords}

Focus on sustainable living practices that citizens can implement in their daily lives.

Eco Tips:"""
        return granite_model.generate_text(prompt, max_length=250)

    def forecast_kpis(self, data: List[float]) -> Dict:
        """Forecast KPIs using linear regression"""
        try:
            if len(data) < 4:
                return {"error": "Need at least 4 data points for forecasting"}

            # Prepare data
            X = np.array(range(len(data))).reshape(-1, 1)
            y = np.array(data)

            # Train model
            model = LinearRegression()
            model.fit(X, y)

            # Forecast next 3 periods
            future_periods = np.array(range(len(data), len(data) + 3)).reshape(-1, 1)
            predictions = model.predict(future_periods)

            return {
                "historical_data": data,
                "forecasted_values": predictions.tolist(),
                "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
                "r_squared": model.score(X, y)
            }
        except Exception as e:
            return {"error": str(e)}

    def detect_anomalies(self, data: List[float], threshold: float = 2.0) -> Dict:
        """Simple anomaly detection using z-score"""
        try:
            data_array = np.array(data)
            mean = np.mean(data_array)
            std = np.std(data_array)

            z_scores = np.abs((data_array - mean) / std)
            anomalies = []

            for i, z_score in enumerate(z_scores):
                if z_score > threshold:
                    anomalies.append({
                        "index": i,
                        "value": data[i],
                        "z_score": float(z_score),
                        "severity": "high" if z_score > 3.0 else "medium"
                    })

            return {
                "total_points": len(data),
                "anomalies_found": len(anomalies),
                "anomalies": anomalies,
                "mean": float(mean),
                "std": float(std)
            }
        except Exception as e:
            return {"error": str(e)}

    def chat_response(self, message: str) -> str:
        """Generate chat responses for citizen queries"""
        prompt = f"""You are a helpful Smart City Assistant AI. Answer the following question about urban sustainability, governance, and city services:

Question: {message}

Provide a helpful, informative response:"""
        return granite_model.generate_text(prompt, max_length=300)

# Initialize assistant
assistant = SmartCityAssistant()

# FastAPI app
app = FastAPI(title="Sustainable Smart City Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def root():
    return {"message": "Sustainable Smart City Assistant API"}

@app.post("/summarize")
async def summarize_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode('utf-8')
        summary = assistant.summarize_document(text)
        return {"summary": summary, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: CitizenFeedback):
    result = assistant.process_citizen_feedback(feedback)
    return result

@app.get("/feedback")
async def get_feedback():
    return {"feedback": assistant.feedback_data}

@app.post("/eco-tips")
async def get_eco_tips(request: EcoTipsRequest):
    tips = assistant.generate_eco_tips(request.keywords)
    return {"tips": tips, "keywords": request.keywords}

@app.post("/forecast")
async def forecast_kpis(data: List[float]):
    result = assistant.forecast_kpis(data)
    return result

@app.post("/anomaly-detection")
async def detect_anomalies(data: List[float], threshold: float = 2.0):
    result = assistant.detect_anomalies(data, threshold)
    return result

@app.post("/chat")
async def chat(message: ChatMessage):
    response = assistant.chat_response(message.message)
    return {"response": response}

# Gradio Interface Functions
def gradio_summarize(file):
    if file is None:
        return "Please upload a file"
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        summary = assistant.summarize_document(content)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

def gradio_feedback(name, email, category, message, location):
    feedback = CitizenFeedback(
        name=name, email=email, category=category,
        message=message, location=location
    )
    result = assistant.process_citizen_feedback(feedback)
    return f"Feedback submitted successfully! ID: {result['id']}"

def gradio_eco_tips(keywords):
    tips = assistant.generate_eco_tips(keywords)
    return tips

def gradio_forecast(data_text):
    try:
        data = [float(x.strip()) for x in data_text.split(',')]
        result = assistant.forecast_kpis(data)
        if 'error' in result:
            return f"Error: {result['error']}"
        return f"Forecasted values: {result['forecasted_values']}\nTrend: {result['trend']}\nR-squared: {result['r_squared']:.3f}"
    except Exception as e:
        return f"Error: {str(e)}"

def gradio_anomaly(data_text, threshold):
    try:
        data = [float(x.strip()) for x in data_text.split(',')]
        result = assistant.detect_anomalies(data, threshold)
        if 'error' in result:
            return f"Error: {result['error']}"
        return f"Anomalies found: {result['anomalies_found']}\nDetails: {result['anomalies']}"
    except Exception as e:
        return f"Error: {str(e)}"

def gradio_chat(message, history):
    response = assistant.chat_response(message)
    history.append((message, response))
    return "", history

# Create Gradio Interface
def create_gradio_interface():
    with gr.Blocks(title="Sustainable Smart City Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏙️ Sustainable Smart City Assistant AI")
        gr.Markdown("Powered by IBM Granite LLM for urban sustainability and governance")

        with gr.Tabs():
            # Document Summarization Tab
            with gr.TabItem("📄 Document Summarization"):
                gr.Markdown("### Upload policy documents for AI-powered summarization")
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload Document", file_types=[".txt", ".pdf"])
                        summarize_btn = gr.Button("Summarize", variant="primary")
                    with gr.Column():
                        summary_output = gr.Textbox(label="Summary", lines=8)

                summarize_btn.click(gradio_summarize, inputs=[file_input], outputs=[summary_output])

            # Citizen Feedback Tab
            with gr.TabItem("📝 Citizen Feedback"):
                gr.Markdown("### Submit feedback and reports to city administration")
                with gr.Row():
                    with gr.Column():
                        name_input = gr.Textbox(label="Name")
                        email_input = gr.Textbox(label="Email")
                        category_input = gr.Dropdown(
                            choices=["Water", "Electricity", "Transportation", "Waste", "Environment", "Other"],
                            label="Category"
                        )
                        message_input = gr.Textbox(label="Message", lines=4)
                        location_input = gr.Textbox(label="Location (Optional)")
                        feedback_btn = gr.Button("Submit Feedback", variant="primary")
                    with gr.Column():
                        feedback_output = gr.Textbox(label="Status", lines=3)

                feedback_btn.click(
                    gradio_feedback,
                    inputs=[name_input, email_input, category_input, message_input, location_input],
                    outputs=[feedback_output]
                )

            # Eco Tips Tab
            with gr.TabItem("🌱 Eco Tips Generator"):
                gr.Markdown("### Get AI-generated eco-friendly tips")
                with gr.Row():
                    with gr.Column():
                        keywords_input = gr.Textbox(label="Keywords (e.g., plastic, solar, water)")
                        tips_btn = gr.Button("Generate Tips", variant="primary")
                    with gr.Column():
                        tips_output = gr.Textbox(label="Eco Tips", lines=8)

                tips_btn.click(gradio_eco_tips, inputs=[keywords_input], outputs=[tips_output])

            # KPI Forecasting Tab
            with gr.TabItem("📊 KPI Forecasting"):
                gr.Markdown("### Forecast city KPIs using machine learning")
                with gr.Row():
                    with gr.Column():
                        data_input = gr.Textbox(
                            label="Historical Data (comma-separated)",
                            placeholder="100, 120, 130, 140, 150"
                        )
                        forecast_btn = gr.Button("Generate Forecast", variant="primary")
                    with gr.Column():
                        forecast_output = gr.Textbox(label="Forecast Results", lines=6)

                forecast_btn.click(gradio_forecast, inputs=[data_input], outputs=[forecast_output])

            # Anomaly Detection Tab
            with gr.TabItem("🔍 Anomaly Detection"):
                gr.Markdown("### Detect anomalies in city data")
                with gr.Row():
                    with gr.Column():
                        anomaly_data_input = gr.Textbox(
                            label="Data Points (comma-separated)",
                            placeholder="10, 12, 11, 50, 13, 12"
                        )
                        threshold_input = gr.Slider(
                            minimum=1.0, maximum=4.0, value=2.0, step=0.1,
                            label="Anomaly Threshold (Z-score)"
                        )
                        anomaly_btn = gr.Button("Detect Anomalies", variant="primary")
                    with gr.Column():
                        anomaly_output = gr.Textbox(label="Anomaly Results", lines=8)

                anomaly_btn.click(
                    gradio_anomaly,
                    inputs=[anomaly_data_input, threshold_input],
                    outputs=[anomaly_output]
                )

            # Chat Assistant Tab
            with gr.TabItem("💬 Chat Assistant"):
                gr.Markdown("### Ask questions about urban sustainability and city services")
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(label="Your Message", placeholder="How can my city reduce carbon emissions?")
                clear = gr.Button("Clear Chat")

                msg.submit(gradio_chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
                clear.click(fn=lambda: ([], ""), outputs=[chatbot, msg])

        gr.Markdown("---")
        gr.Markdown("*Powered by IBM Granite 3.3-2B-Instruct and modern AI technologies*")

    return demo

# Main execution function
async def run_servers():
    """Run both FastAPI and Gradio servers"""

    # Create Gradio interface
    demo = create_gradio_interface()

    # Start FastAPI server in background
    import threading
    import time

    def start_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()

    # Wait a moment for FastAPI to start
    time.sleep(2)

    print("🚀 Sustainable Smart City Assistant is starting...")
    print("📡 FastAPI server: http://localhost:8000")
    print("🎨 Gradio interface: Starting...")

    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link for Colab
        debug=True
    )

# Run the application
if __name__ == "__main__":
    print("🏙️ Initializing Sustainable Smart City Assistant...")
    print("🤖 Loading IBM Granite model...")

    # Run the servers
    asyncio.run(run_servers())
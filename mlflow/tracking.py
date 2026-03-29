import mlflow
import os

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("VisionAgent")

def log_run(question: str, answer: str, duration: float):
    with mlflow.start_run():
        mlflow.log_param("question", question[:200])
        mlflow.log_metric("duration_seconds", duration)
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_text(answer, "answer.txt")
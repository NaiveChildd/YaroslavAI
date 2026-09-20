from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
VIDEO_FOLDER = "videos"
DB_PATH = "database/history.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)

model = YOLO("best.pt")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            result TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)

    results = model(img)

    output = results[0].plot()

    result_path = os.path.join(STATIC_FOLDER, "result.jpg")
    cv2.imwrite(result_path, output)

    hanger_found = False
    count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["hanger", "hangers"]:
            count += 1

    if count > 0:
        hanger_found = True

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO requests (timestamp, result) VALUES (?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(hanger_found))
    )
    conn.commit()
    conn.close()

    return jsonify({
        "hanger_present": hanger_found,
        "count": count,
        "image_url": "/static/result.jpg"
    })

@app.route("/process_video", methods=["POST"])
def process_video():
    file = request.files["video"]

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(VIDEO_FOLDER, "result.mp4")

    file.save(input_path)

    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    hanger_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["hanger", "hangers"]:
                hanger_detected = True

        out.write(annotated)

    cap.release()
    out.release()

    return jsonify({
        "video_url": "/videos/result.mp4",
        "hanger_detected": hanger_detected
    })

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)

if __name__ == "__main__":
    app.run(debug=True)
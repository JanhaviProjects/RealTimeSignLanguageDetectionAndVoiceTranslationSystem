# 🤟 Sign to Speech: Real-Time Sign Language Detection and Voice Translation System

This project is a real-time web-based system that detects **sign language gestures** using a webcam and translates them into **audible speech**. Built using **Python**, **Flask**, **OpenCV**, and **deep learning (CNN)**, this system is designed to bridge the communication gap between the hearing-impaired and the general public.

---

## 🚀 Features

- 📷 Real-time webcam-based hand gesture detection
- 🧠 CNN-based sign recognition (static gestures)
- 🔊 Converts recognized signs to speech using TTS
- 🌐 Web-based interface built with Flask
- 📱 Responsive design with a simple UI
- 🧪 Fully tested on a local Flask development server

---

## 🛠️ Tools & Technologies Used

| Category         | Technology             |
|------------------|------------------------|
| Programming      | Python                 |
| Web Framework    | Flask                  |
| Computer Vision  | OpenCV, MediaPipe (optional) |
| Deep Learning    | TensorFlow / Keras     |
| Text-to-Speech   | gTTS / pyttsx3         |
| Frontend         | HTML5, CSS3, JavaScript |
| Deployment       | Flask Localhost (`127.0.0.1:5000`) |

---

## 📦 Installation

1. **Clone the repository**

bash
git clone https://github.com/your-username/sign-to-speech.git
cd sign-to-speech
`

2. **Create a virtual environment (optional but recommended)**

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. **Install dependencies**

bash
pip install -r requirements.txt


4. **Run the Flask app**

bash
python app.py
```

5. *Open in browser*

Visit http://127.0.0.1:5000 to start using the system.

---

## 🧠 How It Works

1. *User clicks "Start Detection"* on the web page.
2. Webcam activates and captures hand gestures in real time.
3. Frames are processed using OpenCV and passed to a trained *CNN model*.
4. The recognized gesture is translated into text.
5. Text is converted into speech using *gTTS* or *pyttsx3*.
6. Output is played through the browser or system speaker.

## 📚 References

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [MediaPipe](https://developers.google.com/mediapipe/)
* [gTTS Documentation](https://pypi.org/project/gTTS/)
* [Flask](https://flask.palletsprojects.com/)
* Research papers on sign language recognition (see report for citations)
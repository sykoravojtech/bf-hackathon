# bf-hackathon

## Demo Video

[▶️ Watch the Sewwy Robot demo](bf-hackathon_sewwy-robot.mp4)

## run project
```
poetry run python main.py
```

Mediapipe hands works for palm, thumbs up, peace, four
```
poetry run python models/mediapipe_hands.py -d -i data/four2.png
```


# Human-Robot Interaction Made Natural
Built by humor.io at Black Florest Hackathon, 9-11 May 2025

## 🚀 Overview
A proof-of-concept system that enables mobile robots to interact naturally with human workers using voice commands, hand gestures, and a simple user interface. Designed for real-time responsiveness and ease of use — no technical skills required.

## 🎯 Problem Statement
In manufacturing environments, mobile robots are often rigidly controlled from centralized systems. This creates bottlenecks when tasks shift, leading to:

- Frequent interruptions in human workflow

- A frustrating user experience

- Decreased productivity

- Underutilization of robotic potential

## 💡 Our Solution
- A multi-modal human-robot interaction interface that:

    - Uses voice recognition, hand gestures, and a web-based UI

    - Allows workers to guide robots naturally, on the spot

    - Requires zero programming or robotics knowledge

    - Runs in real-time and is fully demoable on a notebook


## 🧠 Core Features
✅ Voice-controlled commands (e.g., "bring me beer")

✋ Gesture recognition for contextual input

🖥️ Lightweight, web UI

🔄 Real-time feedback and status monitoring

🤝 Seamless integration with ROS2 and REST APIs

## 📦 Models & 🛠️ Tech Stack
Our system integrates a variety of cutting-edge models and open-source tools for real-time human-robot interaction:

**YOLOv8** – for fast and accurate hand gesture detection

**Vosk** – lightweight offline speech recognition (multilingual support)

**MediaPipe** – for hand landmarks and gesture analysis

**ROS2** – robot control and sensor communication

**OpenCV** – for video processing and computer vision

**REST API** – communication between modules and the robot

**Hardware**: SEW mobile robot (with RGB camera, laser scanner)


## 🌱 Future Development
- Self-learning
- 3D Recognition
- Natural Communication
- Transition to Multiple Scenarios
- Simultaneous Multi-model Interaction

We believe robots should feel like helpful teammates — not more work. 🙌

## Black Forest Hackathon: Data Decoded (May 2025, Freiburg)
[Hackathon web](https://www.blackforesthackathon.de/may/)

[Challenges: Data Decoded](https://www.blackforesthackathon.de/challenges-data-decoded/)



🧠 OAIX: Open Artificial Intelligence eXperiment
An Emergent Self-Learning AI Framework Powered by Real-Time Sensory Feedback

Overview
OAIX is a real-time adaptive AI engine built from the ground up to simulate early-stage emergent intelligence. Unlike models trained on static datasets or tuned through reward shaping, OAIX learns organically through continuous multi-sensory input, recursive LSTM feedback loops, and intrinsic motivational systems.

It is not pre-trained, not rule-based, and not dependent on reinforcement signals. It grows by living in a simulated world and making sense of it.

🔧 Architecture
🧠 Dual-LSTM core (Pattern LSTM + Central LSTM)

🪞 Predictive forward model for novelty detection

♻️ Continuous self-supervised feedback loop

🦠 Real-time sensory data input (vision, smell, hearing, touch, digestion, internal states)

📊 Live dashboard with HUD, action tracking, sensory panel, and agent stats

Commands
=========================

# Run with full render
python main.py --render

# Headless speed mode
python main.py

# Enable heatmap tracking
python main.py --heatmap

# Full screen with enemies
python main.py --render --maximize --enemies
=============================


🚨 New Features (v2025.04)
✔ Intrinsic Motivation System

Entropy, boredom detection, and novelty-based exploration

Diversity penalties for repetitive behavior

Softmax temperature modulation via prediction error

✔ Predictive Forward Modeling

Predicts next input state using LSTM forward model

Weights losses by prediction error (novelty = more learning)

✔ RAM-Only Memory Model

⛔ No SQL.

Memory is encoded as pattern weights and LSTM state history.

✔ Visual HUD

Health, energy, digestion, death/failure logs, time-of-day

Action history, food stats, survival trends

Sensory panels w/ tokens and stimulus indicators

✔ Day/Night Vision Mechanics

5 tile range during day

2 tile range at night

✔ Hearing System

Detects directional enemy noise within 3-tile radius

Translates proximity into "enemy:direction" token

✔ Agent Behavior

Movement, sprinting, energy drain

Healing, regeneration, digestion-based energy gain

Starvation + health decay after 12 hrs with no food

✔ Safe Zone (Red Zone)

Enemies avoid it

Agent may strategically return when threatened

✔ Checkpoints & Persistence

Automatically saves after configurable death interval

Recovers model weights, exploration state, and prediction history

✔ Full Heatmap Engine (Live Visualization)

Raw LSTM hidden state activity

Agent exploration across the grid

Live tracking of agent/enemy/food/zone positions

🧠 Core Components
Module	Description
pattern_lstm	Learns token relationships and basic environmental patterns
central_lstm	Core intelligence layer—makes movement decisions
forward_model	Predicts next state → computes prediction error
intrinsic_reward	Adjusts exploration via boredom, entropy, novelty
tokenizer	Encodes real-time stimuli into structured inputs
decoder + reverse	Translates AI output into environment actions
Pygame visualizer	Tracks live stats, direction, stimuli, digestion, and more
🎯 Project Goals
Reproduce intelligence not by brute force, but by sensory learning

Create agents that survive, learn, and adapt without reward functions

Visualize how behavior and pattern recognition emerge from sensory complexity

Push AI closer to organic learning—not anthropomorphism, not AGI hype

📦 Requirements
bash
Copy
Edit
Python 3.11+
torch
pygame
numpy
matplotlib
psutil
scipy
🧪 Execution
bash
Copy
Edit
# Run with full render
python main.py --render

# Headless speed mode
python main.py

# Enable heatmap tracking
python main.py --heatmap

# Full screen with enemies
python main.py --render --maximize --enemies
📈 Data Tracked
Digestion rate

Death counts & causes

Survival duration (logged to CSV)

Food consumed

Health decay sources (enemy vs starvation)


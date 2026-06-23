# NVIDIA Jetson Safe EQ Analysis and Machine Learning Code

This folder contains the NVIDIA Jetson-side code for the Safe EQ project. The Jetson Orin Nano served as the system’s main audio analysis and control-parameter generation platform.

## Purpose

The Jetson received real-time stereo microphone audio from the ESP32-S3 USB audio interface and used that audio to evaluate listening conditions. It performed loudness analysis, machine-learning-based audio descriptor prediction, deterministic tonal analysis, and control-parameter generation for the Raspberry Pi audio playback system.

## Main Functions

* Read microphone audio captured through the ESP32-S3 interface
* Estimate sound pressure level (SPL) from in-ear microphone input
* Run the Context Analysis Model (CAM) for machine-learning-based audio descriptor prediction
* Support Deterministic Tonal Analysis (DTA) for audio feature analysis
* Generate loudness and EQ control parameters for the Raspberry Pi DSP system
* Communicate control parameters to the Raspberry Pi over UART
* Support system latency testing and runtime validation

## CAM Machine Learning Model

The Context Analysis Model was used to predict audio descriptors from captured music input. These descriptors helped characterize the audio content so the system could make more informed DSP adjustments.

CAM predicted descriptors such as:

* brightness
* fullness
* harshness
* tonal balance
* intensity
* electric distortion
* instrumentation density
* percussiveness
* bass presence
* vocal presence

In system testing, CAM achieved prediction accuracy above 90%, and ran on the Jetson with processing time below the required 10-second window.

## Project Context

Safe EQ is a safe-listening audio system designed to reduce the risk of noise-induced hearing loss while preserving audio quality at lower listening levels. The full system used three main platforms:

* ESP32-S3 for stereo MEMS microphone capture and USB audio input
* NVIDIA Jetson Orin Nano for audio analysis, machine learning, and control-parameter generation
* Raspberry Pi 4 for playback, GUI, DSP equalization, limiting, and audio output

## My Work

My work on this portion focused on the Jetson audio analysis pipeline, including SPL reading, CAM evaluation, control-parameter generation, system latency testing, and integration with the ESP32 microphone input and Raspberry Pi control interface.

## Notes

This folder is part of a public portfolio version of the Safe EQ project. It may not include every development or experimental file from the original project.

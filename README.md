Inspiration
To support a vulnerable population through a simple device that, unlike other technologies, does not inhibit their senses. Our goal is to empower users through an intuitive, non-intrusive solution.

What it does
Our project is an intelligent, voice-controlled assistive wearable that helps visually impaired users navigate their surroundings independently. By combining a camera-equipped pair of glasses with a smartphone app, the system processes visual data 100% locally to assist the user in real time.

How we built it
Using a lightweight pair of glasses equipped with an ESP32-CAM module with hardware configured to capture real-time video frames to the user's smartphone over a localized, low-latency connection, preserving frame rate.

The control center is a native Android application built in Java. It acts as a central state machine managing application modes (System Active, Reading Mode, and Restroom Mode) seamlessly depending on user behavior and real-time environment requirements.

We trained a custom object detection model using YOLO11 to recognize key environmental elements. To achieve 100% local execution we optimized and deployed the model on the device using TensorFlow Lite integrated directly into Android Studio. For reading emergency and navigational signs, we integrated Google ML Kit. We designed a custom image processing pipeline that crops the camera feed to a central 50% boundary box, forcing the system to focus exactly on what the user is pointing at. Navigation is entirely hands-free via Android's native SpeechRecognizer configured for Mexican Spanish

Challenges we ran into
Java Model Compatibility: Ensuring our AI model integrated smoothly with native Android Java code.

Hardware-Software Sync: Establishing a stable, low-latency communication link between the ESP32 hardware and the smartphone application.

Model Optimization: Training, refining, and squeezing a computer vision model to run efficiently on mobile hardware.

Edge AI Deployment: Implementing a local model inference on a smartphone without relying on cloud servers.

Data Stream Quality: Balancing high video quality with a consistent frame rate while streaming data from the ESP32.

Voice Command Management: Handling continuous background voice recognition without picking up audio feedback from the device's own speakers.

Accomplishments that we're proud of
Social Impact: Building a viable, high-quality solution engineered specifically to empower a vulnerable population.

Model Precision: Achieving high prototype accuracy and reliable detection despite working under limited hardware and budget constraints.

100% Local Execution: Keeping all data processing on the device to guarantee absolute user privacy and offline functionality.

What we learned
How to train advanced computer vision models using YOLO11 and optimize them for local edge deployment.

Mastering the integration of Java and TensorFlow Lite inside Android Studio for real-time inference.

Designing robust software architectures that handle sensor fusion (camera + microphone) efficiently on Android.

What's next for Viccio
Expand Detection: Train the model to recognize a wider variety of everyday objects and potential hazards.

UX Refinement: Continuously improve the user experience based on direct feedback from visually impaired users.

Cost Optimization: Increase video and processing quality without driving up hardware and manufacturing costs.

Mass Production: Create a scalable manufacturing and hardware assembly strategy.

Strategic Partnerships: Forge alliances with NGOs, health organizations, and strategic allies to scale our growth and reach those who need it most.

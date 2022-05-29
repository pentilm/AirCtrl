# AirCtrl: Control Your Mouse with Hand Gestures

AirCtrl is an innovative Python project that enables you to control your mouse using your hand gestures. The project uses your webcam to capture and interpret your hand gestures and translate them into mouse movements and actions. Built with the power of OpenCV and TensorFlow, the project provides a new, interactive, and intuitive way to control your computer.

The project is developed by Zhiwei Fang and is licensed under the MIT License.

## Concept

AirCtrl uses the principles of computer vision and deep learning to interpret hand gestures. The primary technology behind the project includes OpenCV and TensorFlow, two powerful and widely used libraries in Python.

- **OpenCV (Open Source Computer Vision Library)**: OpenCV is leveraged to handle image and video operations. It is responsible for capturing video input, processing the images, and detecting and tracking hand movements in video frames.

- **TensorFlow**: TensorFlow, a leading machine learning framework, is used to train and run the deep learning model that recognizes hand gestures and translates them into mouse actions.

## Algorithm

The core algorithm of the project involves the detection and interpretation of hand gestures in real-time:

1. **Hand Detection**: The algorithm first applies a skin color detection technique to isolate the hand in video frames. This technique relies on color thresholding to identify pixels that match the color range of human skin.

2. **Hand Tracking**: Once the hand is detected, the algorithm identifies the contours of the hand. It then calculates the centroid, which is the center point of the contour.

3. **Gesture Recognition**: The TensorFlow model, trained on a variety of hand gestures, recognizes the specific gesture being made.

4. **Mouse Control**: The movement of the centroid and the recognized gesture are translated into corresponding mouse actions. For example, moving the hand to the right may translate to moving the mouse cursor to the right, while a specific gesture could be interpreted as a mouse click.

## Dataset

The model is trained on a custom dataset that comprises various hand gestures that mimic mouse controls, such as moving the mouse, clicking, and scrolling. The dataset includes a diverse range of hand shapes, sizes, and movements to ensure that the model is robust and can adapt to different users and conditions.

## Setup

To run this project, your system should meet the following requirements:

- Ubuntu 20.04 with Desktop GUI
- NVIDIA GPU with CUDA Support

To get AirCtrl working on your system, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/pentilm/AirCtrl.git
```

2. Navigate into the cloned repository:

```bash
cd AirCtrl
```

3. Install the necessary Python dependencies manually:

For TensorFlow:

```bash
pip install tensorflow==2.6.0
```

For OpenCV:

```bash
pip install opencv-python==4.5.3
```

For NumPy:

```bash
pip install numpy==1.21.0
```

For PyAutoGUI:

```bash
pyautogui==0.9.50
```

For MediaPipe:

```bash
pip install mediapipe==0.8.6
```

4. The pre-trained model weights are stored in the repository using Git Large File Storage (LFS). Make sure you have Git LFS installed on your machine. If not, you can install it by following the instructions on the [Git LFS website](https://git-lfs.github.com/). Once Git LFS is installed, pull the model weights using the following command:

```bash
git lfs pull
```

5. Run the main Python script:

```bash
python airctrl.py
```

## Usage

After starting the script, the application will open your webcam and start tracking your hand movements. You can perform different gestures to control your mouse:

- Move your index finger to move the cursor.
- Pinch your index finger and thumb together to click.
- Swipe your hand left or right to scroll horizontally.
- Swipe your hand up or down to scroll vertically.

Please note that these are just examples and the actual gestures may vary. You can customize the gestures according to your preference by modifying the training data.

## Demo Video

To get a more detailed understanding of the design and to see AirCtrl in action, check out this [full demo video](demo/Demo-Full.mp4).

Here's a short clip:

https://github.com/pentilm/AirCtrl/assets/51684958/162b8e76-54c5-43a8-b156-465dc491c277

## Contact

If you have any questions or suggestions, feel free to open an issue on this repository. We welcome all feedback and contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please note that AirCtrl is a research project and is not intended for commercial use. The authors are not responsible for any damage or harm caused by the use of this software.

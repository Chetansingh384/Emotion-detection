# Emotion Detection Using CNN

![Emotion Detection](https://github.com/Chetansingh384/Emotion_detection_cnn/blob/main/PROJECT_IMAGE.jpg)

## 📝 Project Overview

This project is an implementation of emotion detection using Convolutional Neural Networks (CNNs). The model is trained to recognize various human emotions from facial expressions. The goal is to provide a system that can accurately classify emotions like happiness, sadness, anger, surprise, etc., based on input images.

## 📚 Features

- **Real-time Emotion Detection:** The model can predict emotions from live video feeds or static images.
- **Multi-Emotion Classification:** It can classify emotions into multiple categories like happy, sad, angry, surprised, neutral, etc.
- **User-Friendly Interface:** The project includes an easy-to-use interface for uploading images or using a webcam to detect emotions.
- **Scalable and Customizable:** The model can be fine-tuned or expanded to include more emotion categories or datasets.

## 🛠️ Technologies Used

- **Python**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**

## 🎯 How It Works

1. **Data Preprocessing:** Facial expression images are preprocessed (resizing, normalization).
2. **Model Training:** A CNN model is trained on a labeled dataset to recognize different emotions.
3. **Prediction:** The trained model predicts the emotion for a given image or video frame.
4. **Display:** The predicted emotion is displayed with the corresponding probability.

## 🚀 Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chetansingh384/Emotion_detection_cnn
   ```
2. **Run the application:**
   ```bash
   python project1.ipynb
   ```

## 💡 Usage
- **Use Webcam:** Use the webcam to capture live video and detect emotions in real-time.

## 🧠 Model Architecture

The CNN model used in this project consists of multiple convolutional layers followed by pooling and fully connected layers. The architecture is designed to capture and learn facial features effectively.

- **Input Layer:** 48x48 grayscale images
- **Convolutional Layers:** Extract features using filters
- **Pooling Layers:** Downsample feature maps
- **Fully Connected Layers:** Classify the extracted features into emotion categories
- **Output Layer:** Softmax activation for multi-class classification

## 📁 Dataset

The model was trained on the [FER 2013] dataset, which contains thousands of labeled images across different emotion categories.
- **Number of Images:**
- **Train** = 28709
- **Test** = 7178
- **Image Size:** 48x48 pixels

 You can download the dataset from the following Google Drive link:

[Download Dataset](https://drive.google.com/drive/folders/11MouWOiB3WompIieCSnW2voIUk55Bv9Z?usp=drive_link)

## 🤖 Future Work

- **Expand the Dataset:** Add more images and categories for better generalization.
- **Model Optimization:** Experiment with different architectures and hyperparameters to improve accuracy.
- **Real-Time Deployment:** Deploy the model in a real-time application using TensorFlow Lite or similar.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- Special thanks to the creators of the [FER 2013].
- Inspiration and guidance from [MR. Ranjit Singh /Upflair Pvt Ltd].

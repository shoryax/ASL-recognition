# ASL Recognition

Machine Learning model for recognizing **American Sign Language (ASL) letters**

---

## 📦 Dataset

We used a public ASL dataset provided by [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters), containing labeled images for each letter in the ASL alphabet.

- Format: Object Detection (converted for classification)
- Classes: A–Z (26 total)
- Source: Roboflow Public Datasets

---

## 🧠 Model

The model is based on **MobileNetV2**, a lightweight convolutional neural network:

- ✅ Pretrained on **ImageNet**
- 🔁 Fine-tuned on the ASL dataset
- 🔧 Optimized for **mobile performance** (ideal for Snap AR)

---

## 📊 Accuracy

| Dataset       | Accuracy   |
|---------------|------------|
| ✅ Validation | **86.9%%** |
| ✅ Test       | **86%**    |
| ✅ Training:  |  **100%**  |

Status: Ready to use! 🎉

The model generalizes well, maintaining high performance on unseen ASL hand signs.

---

## 🧪 Image Augmentation

To improve generalization and robustness, the following augmentations were applied during training:

- 🔄 **Rotation**
- ↔️ **Width & Height Shift**
- 🔍 **Zoom**
- 📐 **Shear Transformation**
- 💡 **Brightness Adjustment**

These augmentations simulate real-world variations like lighting and camera angles.

---

## 🚀 How to Use

1. Clone this repo:

   ```bash
   git clone https://github.com/your-username/asl-recognition.git
   cd asl-recognition

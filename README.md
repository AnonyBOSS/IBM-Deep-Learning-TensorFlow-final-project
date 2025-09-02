# Project Overview: Waste Classification Using Transfer Learning

This project aims to build an automated waste classification model to distinguish between **recyclable** and **organic** waste from images. The goal is to address the inefficiencies and errors of manual waste sorting by leveraging machine learning and computer vision. This notebook walks through the process of using a pre-trained VGG16 model and applying transfer learning techniques to achieve accurate image classification.

---

### Learning Objectives 📚

By completing this project, you will be able to:

* Apply transfer learning with the **VGG16 model** for image classification tasks.
* Prepare and preprocess image data for machine learning.
* Fine-tune a pre-trained model to enhance classification accuracy.
* Evaluate model performance using key metrics.
* Visualize model predictions on test data.

---

### Methodology ⚙️

1.  **Setup**: The project begins by installing and importing necessary libraries, including **TensorFlow, Keras, NumPy, Scikit-learn, and Matplotlib**.

2.  **Data Preparation**: The Waste Classification Dataset is downloaded and prepared. The images are divided into training and testing sets, with a 20% validation split. `ImageDataGenerator` is used for real-time data augmentation, which includes rescaling, width and height shifts, and horizontal flips.

3.  **Model Building (Extract Features)**:
    * A base model is created using the **VGG16 architecture** with pre-trained ImageNet weights, excluding the top classification layer.
    * The layers of the base model are "frozen" so their weights are not updated during initial training.
    * A new classification model is built on top of the base model, consisting of `Dense` and `Dropout` layers to prevent overfitting.

4.  **Training and Evaluation**:
    * The model is compiled with the `binary_crossentropy` loss function and the `RMSprop` optimizer.
    * **Early stopping** is used to prevent over-training.
    * The model is trained, and the loss and accuracy curves for both training and validation sets are plotted to visualize performance.

5.  **Fine-Tuning**:
    * To further improve performance, the model is fine-tuned by unfreezing the last convolutional block of the VGG16 base model (`block5_conv3` and subsequent layers).
    * The model is then re-compiled and trained with a lower learning rate.
    * Loss and accuracy curves for the fine-tuned model are also plotted.

6.  **Final Evaluation**: Both the initial "extract features" model and the "fine-tuned" model are evaluated on the test dataset. The notebook includes code to visualize predictions on test images, comparing the actual labels with the predicted labels from both models.

---

### How to Use This Notebook 🚀

1.  **Run the Setup Cells**: Execute the cells under the "Setup" section to install and import the required libraries.
2.  **Follow the Tasks**: The notebook is divided into 10 tasks. Follow the instructions in each markdown cell and execute the corresponding code cells to progress through the project.
3.  **Review the Results**: After training and fine-tuning, the notebook provides classification reports and visualizations to evaluate the models' performance.

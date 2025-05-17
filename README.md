Audio DeepFake Detection using LMDC and STCA

This project aims to detect spam (fake) audio messages using a combination of LMDC (Multi-Scale Dynamic Learning) for feature extraction 
and STCA (Spectro-Temporal Cross Aggregation) for attention-based temporal modeling. The model processes audio data to classify whether 
a voice message is real (Real voice) or fake (Fake voice).

Dataset :

The dataset contains 54,000 labeled audio samples, divided as follows:
27,000 real audio messages (labeled as 1, representing real voice)
27,000 fake audio messages (labeled as 0, representing fake voice)
Each audio file was preprocessed and transformed into relevant features for input into the model.

Preprocessing :

Before feeding the data into the model, the following preprocessing steps were applied :
Label Encoding : All labels (real/fake) were encoded using LabelEncoder to numerical format.
Standardization : Features were standardized using StandardScaler to ensure equal weight and improve convergence.

Feature Extraction :

Audio messages were converted into numerical representations using a set of extracted features. These include:
MFCCs (Mel-Frequency Cepstral Coefficients)
Spectrograms
Chroma Features
Zero-Crossing Rate
Spectral Contrast
These features help the model understand the structure and quality of the audio message.

Model Architecture :

The model is composed of the following main blocks:
1- LMDC (Local Multi-scale Dilated Convolutions)
Three parallel Conv1D branches with kernel sizes (3, 5, 7).
Each branch extracts features at different temporal resolutions.
Max-pooling and dropout applied after convolution.

2- STCA (Spectro-Temporal Cross Aggregation)
An attention block that focuses dynamically on the most important time steps of the extracted features.
Improves the model’s ability to interpret short-term dependencies in the audio signal.

3- BiLSTM Layers
Bidirectional LSTM layers capture sequential patterns and contextual information from both past and future time steps.

4- Dense Layers

Fully connected layers with dropout regularization reduce overfitting and produce the final classification.

Libraries Used :

numpy
pandas
matplotlib, seaborn
sklearn
tensorflow / keras
librosa

Training Configuration : 

Optimizer: Adam with learning rate 0.001
Loss Function: Binary Crossentropy
Evaluation Metrics: Accuracy
Validation Split: 20%
Batch Size: 64
Epochs: 4 (with EarlyStopping)

Evaluation :
After training, the model was evaluated on the test set using:
Accuracy Score
Classification Report (Precision, Recall, F1-score)
Confusion Matrix
The model achieved 99% accuracy, showing strong performance in identifying both real and fake audio.

Conclusion :

This project demonstrates an effective audio-based spam detection system using a custom deep learning architecture with LMDC and STCA.
The integration of multi-scale convolutions, attention mechanisms, and LSTM layers provides a robust approach to classifying voice messages.


Email: mohamedsaidelhosiny@gmail.com
Created in 5/17/2025

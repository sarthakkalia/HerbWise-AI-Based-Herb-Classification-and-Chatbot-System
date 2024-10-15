# HerbWise: AI-Based Herb Classification and Chatbot System

## Overview
HerbWise is a comprehensive AI-driven project that integrates a fine-tuned Convolutional Neural Network (CNN) model for herb classification and a chatbot powered by NLP and neural networks to assist users in learning about herb uses and advantages.

## Features
1. **Herb Classification**: 
   - Fine-tuned `MobileNetV2` and `VGG16` CNN models for classifying herb plants.
   - **MobileNetV2** was selected as the final model due to higher validation accuracy.
   
2. **Chatbot for Herb Information**:
   - Built using NLP techniques and neural networks.
   - Provides information on the uses and benefits of herbs based on user queries.

## Herb Classification Model

### Models Evaluated
- **VGG16**
  - Accuracy: 60.00%
  - Loss: 0.9815
  
- **MobileNetV2** *(Final Model)*
  - Accuracy: 70.00%
  - Loss: 0.7458

MobileNetV2 was chosen due to its higher performance and faster training time.

## Chatbot System

### Machine Learning Models
Various models were trained and tuned using `TfidfVectorizer` and the following algorithms:

- **Logistic Regression**: Accuracy = 0.7870 (Best parameters: `C=10.0`, `max_iter=100`, `penalty='l2'`, `solver='liblinear'`)
- **Multinomial Naive Bayes**: Accuracy = 0.7130 (Best parameters: `alpha=0.5`)
- **Linear SVC**: Accuracy = 0.7685 (Best parameters: `C=1`, `loss='squared_hinge'`, `max_iter=100`, `penalty='l2'`)
- **Decision Tree**: Accuracy = 0.8056 (Best parameters: `criterion='entropy'`, `max_depth=None`, `min_samples_leaf=2`, `min_samples_split=5`)
- **Random Forest**: Accuracy = 0.7778 (Best parameters: `max_depth=20`, `min_samples_leaf=1`, `min_samples_split=5`, `n_estimators=100`)

**Best Model**: Decision Tree with 80.56% accuracy.

### Neural Network Chatbot
A custom neural network was implemented to provide herb information:

- **Testing Accuracy**: 97.70%
- **Testing Loss**: 0.0105

#### Neural Network Architecture:
```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        output = self.softmax(x)
        return output
```

### Performance Evaluation
The chatbot was tested using a custom dataset, with the following results:
- **Average Accuracy**: 97.70%
- **Average Loss**: 0.0105

## Technologies Used
- **Languages**: Python
- **Libraries/Frameworks**: 
  - CNN Models: TensorFlow, Keras
  - NLP: TfidfVectorizer, Scikit-learn, SpaCy, NLTK
  - Neural Networks: PyTorch
- **Other Tools**: Docker, Git, Jupyter Notebooks, VS Code

## Future Enhancements
- Increase the dataset for more diverse herb classification.
- Integrate the chatbot with real-time herb data for more comprehensive information.

## How to Use
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the herb classification model:
   ```bash
   python herb_classification.py
   ```
4. Start the chatbot system:
   ```bash
   python chatbot.py
   ```

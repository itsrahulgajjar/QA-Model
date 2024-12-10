## README for Question Answering System

### Project Overview
This project implements a Question Answering (QA) system using a fine-tuned DistilBERT model. It consists of scripts to preprocess data, fine-tune the model, and provide an API endpoint to dynamically generate answers to user questions.

---

### Files and Descriptions

#### 1. **load_preprocess.py**
   - **Purpose**: Loads and preprocesses a dataset for fine-tuning a QA model.
   - **Key Features**:
     - Reads a JSON dataset.
     - Extracts key fields (e.g., `id`, `question`, `context`, `answer_text`).
     - Tokenizes data using the Hugging Face `AutoTokenizer`.
     - Encodes data into a format suitable for training.

#### 2. **train_model.py**
   - **Purpose**: Fine-tunes a pre-trained DistilBERT model for QA.
   - **Key Features**:
     - Splits the dataset into training and validation sets.
     - Fine-tunes the model using the Hugging Face `Trainer` API.
     - Saves the fine-tuned model and tokenizer for later use.

#### 3. **api.py**
   - **Purpose**: Provides a REST API for answering user questions.
   - **Key Features**:
     - Accepts a user question as input.
     - Finds the top 3 most relevant questions from the dataset.
     - Uses the fine-tuned model to generate an answer based on the closest match's context.

---

### Model Selection and Fine-Tuning Process

#### **Model Selection**
The `distilbert-base-uncased` model was chosen for its efficiency and balance between speed and performance. It is a lightweight version of BERT, designed for resource-constrained environments while maintaining high accuracy.

#### **Fine-Tuning**
1. **Dataset**: A JSON dataset with fields for questions, answers, and context.
2. **Preprocessing**:
   - Tokenized the dataset with context and questions as input.
   - Encoded data for training with answer start and end positions.
3. **Training Configuration**:
   - Optimized for CPU with reduced batch sizes and epochs.
   - Evaluated at the end of each epoch to ensure model convergence.
4. **Output**:
   - The fine-tuned model and tokenizer are saved locally for inference.

---

### API Details

#### **Endpoint**
- **URL**: `/answer`
- **Method**: `POST`
- **Input Format**: JSON
  - `question` (string): The question to be answered.

#### **Sample Input**
```json
{
  "question": "What is the purpose of the project?"
}
```

#### **Sample Response**
```json
{
  "id": "12345",
  "closest_question": "What is the purpose of this QA system?",
  "generated_response": "This project implements a Question Answering system using fine-tuned DistilBERT."
}
```

---

### Setup and Execution Instructions

#### **1. Create a Virtual Environment**
Create a virtual environment to manage dependencies:
```bash
python -m venv .venv
```

Activate the virtual environment:
- **Windows**: 
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux**: 
  ```bash
  source .venv/bin/activate
  ```

#### **2. Install Requirements**
Ensure Python is installed. Install the required packages:
```bash
pip install torch transformers flask pandas scikit-learn
```

#### **3. Preprocess the Dataset**
Ensure your dataset file `qa_model_dataset.json` is in the project directory. Run:
```bash
python load_preprocess.py
```

#### **4. Train the Model**
Fine-tune the model on your dataset:
```bash
python train_model.py
```

#### **5. Start the API**
Run the API server:
```bash
python api.py
```
The server will start at `http://127.0.0.1:5000`.

#### **6. Test the API**
Use tools like `Postman` or `curl` to send a POST request to the `/answer` endpoint:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the purpose of the project?"}' http://127.0.0.1:5000/answer
```

---

### Notes
- **Dataset Requirements**: The dataset must be a JSON file with `id`, `question`, `context`, and `answers` fields.
- **Model Storage**: The fine-tuned model is stored in the `trained_model` directory.
- **Context Combination**: Combines top 3 matched contexts to enhance response accuracy.
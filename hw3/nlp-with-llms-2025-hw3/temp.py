import os
import dspy
import json
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from dspy.evaluate import SemanticF1

# Load environment and configure LLM for SemanticF1 evaluation
load_dotenv("grok_key.ini")
lm = dspy.LM('xai/grok-3-mini', api_key=os.environ['XAI_API_KEY'], max_tokens=6000, temperature=0.1, top_p=0.9)
dspy.configure(lm=lm)

# Set up the retriever from rag.ipynb
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
embedder = dspy.Embedder(model.encode)

# Load the pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Traverse a directory and read html files - extract text from the html files
def read_html_files(dir_name, directory="../PragmatiCQA-sources"):
    texts = []
    full_path = os.path.join(directory, dir_name)
    if not os.path.exists(full_path):
        return []
    
    for filename in os.listdir(full_path):
        if filename.endswith(".html"):
            with open(os.path.join(full_path, filename), 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                texts.append(soup.get_text())
    return texts

# Create retriever for a specific topic
def make_search(topic):
    corpus = read_html_files(topic)
    if not corpus:
        return None
    max_characters = 10000 
    topk_docs_to_retrieve = 5
    return dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve, brute_force_threshold=max_characters)

# Load PragmatiCQA dataset
def read_data(filename, dataset_dir="../PragmatiCQA/data"):
    corpus = []
    with open(os.path.join(dataset_dir, filename), 'r') as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus

# Traditional QA class using DistilBERT
class TraditionalQA:
    def __init__(self):
        self.qa_pipeline = qa_pipeline
    
    def answer_from_context(self, question, context):
        """Generate answer using DistilBERT QA model"""
        if not context or not context.strip():
            return "No answer found"
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            return result['answer']
        except:
            return "No answer found"
    
    def answer_from_retrieval(self, question, search):
        """Generate answer using retrieved context"""
        if search is None:
            return "No answer found"
        
        try:
            passages = search(question).passages
            context = " ".join(passages)
            return self.answer_from_context(question, context)
        except:
            return "No answer found"

# Load datasets
val_data = read_data("val.jsonl")
test_data = read_data("test.jsonl")

# Initialize traditional QA model
traditional_qa = TraditionalQA()

# Function to evaluate a single configuration
def evaluate_configuration(dataset, config_type="retrieved"):
    """
    Evaluate model on dataset for a specific configuration
    config_type: "literal", "pragmatic", or "retrieved"
    """
    examples = []
    predictions = []
    
    for conversation in dataset:
        topic = conversation['topic']
        first_qa = conversation['qas'][0]  # Only first question in each conversation
        
        question = first_qa['q']
        gold_answer = first_qa['a']
        
        # Generate prediction based on configuration type
        if config_type == "literal":
            lit_spans = [l['text'] for l in first_qa['a_meta']['literal_obj']]
            context = ' '.join(lit_spans)
            pred_answer = traditional_qa.answer_from_context(question, context)
        elif config_type == "pragmatic":
            prag_spans = [l['text'] for l in first_qa['a_meta']['pragmatic_obj']]
            context = ' '.join(prag_spans)
            pred_answer = traditional_qa.answer_from_context(question, context)
        else:  # retrieved
            search = make_search(topic)
            pred_answer = traditional_qa.answer_from_retrieval(question, search)
        
        # Create examples for evaluation
        example = dspy.Example(question=question, response=gold_answer)
        pred = dspy.Example(question=question, response=pred_answer)
        
        examples.append(example)
        predictions.append(pred)
    
    return examples, predictions


# First: Run the model on the PRAGMATICQA test set and evaluate with SemanticF1
print("=== Running Traditional QA Model on Test Set ===")
print(f"Total conversations in test set: {len(test_data)}")

# Run predictions for all three configurations on test set
test_predictions = {}
for config in ["Literal", "Pragmatic", "Retrieved"]:
    print(f"\nRunning {config} configuration on test set...")
    examples, predictions = evaluate_configuration(test_data, config)
    test_predictions[config] = {
        'examples': examples,
        'predictions': predictions
    }
    print(f"Generated {len(predictions)} predictions for {config} configuration")

# Evaluate test set predictions using SemanticF1 (only F1 score)
print("\n=== Evaluating Test Set with SemanticF1 ===")
metric = SemanticF1(decompositional=True)
test_results = {}

for config in ["Literal", "Pragmatic", "Retrieved"]:
    print(f"\nEvaluating {config} configuration on test set...")
    examples = test_predictions[config]['examples']
    predictions = test_predictions[config]['predictions']
    
    # Evaluate with SemanticF1 - get only F1 score for test set
    f1_scores = []
    
    for example, pred in zip(examples, predictions):
        try:
            score = metric(example, pred)
            # For test set, we only need the F1 score
            f1_scores.append(float(score))
        except Exception as e:
            print(f"Error evaluating example: {e}")
            f1_scores.append(0.0)
    
    # Calculate average F1
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    test_results[config] = {
        'f1': avg_f1,
        'count': len(examples)
    }
    
    print(f"{config} Test F1: {avg_f1:.3f} ({len(examples)} examples)")

# Display test results table
print("\n=== Test Set Results Summary ===")
print("Configuration | F1 Score | Count")
print("-" * 35)
for config, results in test_results.items():
    print(f"{config:<12} | {results['f1']:.3f}   | {results['count']}")

# Second: Evaluate on validation dataset with precision, recall, and F1
print("\n=== Evaluating on Validation Dataset ===")
print(f"Total conversations in validation set: {len(val_data)}")

# For validation set, we need precision, recall, and F1
# We'll use decompositional=True to get individual components
metric_decomp = SemanticF1(decompositional=True)
val_results = {}

for config in ["Literal", "Pragmatic", "Retrieved"]:
    print(f"\nEvaluating {config} configuration on validation set...")
    examples, predictions = evaluate_configuration(val_data, config)
    
    # For validation set, we need precision, recall, and F1
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for example, pred in zip(examples, predictions):
        try:
            # Create proper dspy.Example objects for SemanticF1
            gold_example = dspy.Example(question=example.question, response=example.response)
            pred_example = dspy.Example(question=pred.question, response=pred.response)
            
            score = metric_decomp(gold_example, pred_example)
            
            # Extract precision, recall, F1 if available in decomposed format
            if hasattr(score, 'precision') and hasattr(score, 'recall') and hasattr(score, 'f1'):
                precision_scores.append(float(score.precision))
                recall_scores.append(float(score.recall))
                f1_scores.append(float(score.f1))
            else:
                # If not decomposed, use the score as F1 and approximate precision/recall
                f1_score = float(score)
                f1_scores.append(f1_score)
                precision_scores.append(f1_score)  # Approximation
                recall_scores.append(f1_score)     # Approximation
                
        except Exception as e:
            print(f"Error evaluating example: {e}")
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
    
    # Calculate averages
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    val_results[config] = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'count': len(examples)
    }
    
    print(f"{config} Validation Results:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall: {avg_recall:.3f}")
    print(f"  F1: {avg_f1:.3f}")
    print(f"  Examples: {len(examples)}")

# Display validation results table
print("\n=== Validation Results Summary ===")
print("Configuration | Precision | Recall | F1     | Count")
print("-" * 50)
for config, results in val_results.items():
    print(f"{config:<12} | {results['precision']:.3f}    | {results['recall']:.3f} | {results['f1']:.3f} | {results['count']}")

# Show some sample predictions
print("\n=== Sample Test Set Predictions ===")
for i in range(min(3, len(test_data))):
    conversation = test_data[i]
    topic = conversation['topic']
    first_qa = conversation['qas'][0]
    question = first_qa['q']
    gold_answer = first_qa['a']
    
    print(f"\nExample {i+1} - Topic: {topic}")
    print(f"Question: {question}")
    print(f"Gold Answer: {gold_answer}")
    
    for config in ["Literal", "Pragmatic", "Retrieved"]:
        pred_answer = test_predictions[config]['predictions'][i].response
        print(f"{config} Prediction: {pred_answer}")

# Analysis continues...
print("\n=== Analysis ===")

# Compare configurations
best_test_config = max(test_results.items(), key=lambda x: x[1]['f1'])
best_val_config = max(val_results.items(), key=lambda x: x[1]['f1'])

print(f"Best Test Configuration: {best_test_config[0]} (F1: {best_test_config[1]['f1']:.3f})")
print(f"Best Validation Configuration: {best_val_config[0]} (F1: {best_val_config[1]['f1']:.3f})")

# Analyze if model tends to give literal answers when pragmatic ones are needed
print("\nAnalyzing literal vs pragmatic tendencies...")
# ... rest of analysis code ...

# Cost tracking
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
print(f"\nTotal evaluation cost: ${cost:.4f}")
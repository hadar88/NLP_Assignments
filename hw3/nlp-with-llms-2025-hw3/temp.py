# DSPy Module for Pragmatic QA
class PragmaticQAModule(dspy.Module):
    def __init__(self, retriever=None):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought("context, conversation_history, question -> answer")
        
    def forward(self, question, conversation_history="", context=""):
        # If no context provided and retriever available, retrieve context
        if not context and self.retriever:
            retrieved = self.retriever(question)
            context = " ".join(retrieved.passages)
        
        # Generate pragmatic answer considering conversation history
        result = self.generate_answer(
            context=context,
            conversation_history=conversation_history,
            question=question
        )
        
        return dspy.Prediction(answer=result.answer)

# 4.4.1 - Evaluate LLM on first questions (same as Part 1 scope)
def evaluate_llm_first_questions(dataset):
    examples = []
    predictions = []
    
    cache_file = "llm_first_q_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    for conversation in dataset:
        topic = conversation['topic']
        if topic not in folders:
            continue
            
        first_qa = conversation['qas'][0]
        question = first_qa['q']
        gold_answer = first_qa['a']
        
        cache_key = f"{topic}|{question}"
        
        if cache_key not in cache:
            # Create retriever for this topic
            search = make_search(topic)
            
            # Initialize the pragmatic QA module
            pragmatic_qa = PragmaticQAModule(retriever=search)
            
            # Generate prediction (no conversation history for first question)
            pred = pragmatic_qa(question=question, conversation_history="")
            pred_answer = pred.answer
            
            cache[cache_key] = pred_answer
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        else:
            pred_answer = cache[cache_key]
        
        example = dspy.Example(question=question, response=gold_answer)
        prediction = dspy.Example(question=question, response=pred_answer)
        
        examples.append(example)
        predictions.append(prediction)
    
    return examples, predictions

# Evaluate LLM on validation set first questions
print("4.4.1 - Evaluating LLM on first questions (same scope as Part 1)...")
val_examples_llm_first, val_predictions_llm_first = evaluate_llm_first_questions(val_data)

# Evaluate LLM scores
llm_first_scores = {"precision": [], "recall": [], "f1": []}
for example, prediction in zip(val_examples_llm_first, val_predictions_llm_first):
    result = metric(example, prediction)
    llm_first_scores["precision"].append(result["precision"])
    llm_first_scores["recall"].append(result["recall"])
    llm_first_scores["f1"].append(result["f1"])

llm_first_results = {
    'precision': sum(llm_first_scores['precision']) / len(llm_first_scores['precision']),
    'recall': sum(llm_first_scores['recall']) / len(llm_first_scores['recall']),
    'f1': sum(llm_first_scores['f1']) / len(llm_first_scores['f1']),
    'count': len(llm_first_scores['f1'])
}

# Compare with Part 1 results
print("\n4.4.1 Comparison - First Questions Only:")
print("Configuration      | Precision | Recall | F1 Score | Count")
print("-" * 65)

# Print Part 1 results
for config, results in val_results.items():
    print(f"{config:<18} | {results['precision']:.4f}   | {results['recall']:.4f}  | {results['f1']:.4f}   | {results['count']}")

# Print LLM results
print(f"{'LLM (DSPy)':<18} | {llm_first_results['precision']:.4f}   | {llm_first_results['recall']:.4f}  | {llm_first_results['f1']:.4f}   | {llm_first_results['count']}")
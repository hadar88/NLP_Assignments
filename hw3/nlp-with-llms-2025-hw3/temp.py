# DSPy Module for Pragmatic QA
class PragmaticQAModule(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("context, conversation_history, question -> answer")
        
    def forward(self, question, conversation_history="", context=""):
        
        # Generate pragmatic answer considering conversation history
        result = self.generate_answer(
            context=context,
            conversation_history=conversation_history,
            question=question
        )
        
        return result.answer

# Function to format conversation history
def format_conversation_history(qas_so_far):
    history = []
    for qa in qas_so_far:
        history.append(f"Question: {qa['q']} \nAnswer: {qa['a']}")
    return "\n\n".join(history)

# 4.4.1 - Evaluate LLM on first questions (following your pattern)
def evaluate_llm_first_questions(dataset):
    examples = []
    predictions = []
    
    cache_file = "llm_first_q_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    # Initialize the pragmatic QA module once
    pragmatic_qa = PragmaticQAModule()
    
    for conversation in dataset:
        topic = conversation['topic']
        if topic not in folders:
            continue
            
        first_qa = conversation['qas'][0]
        question = first_qa['q']
        gold_answer = first_qa['a']
        
        cache_key = f"{topic}|{question}"
        
        if cache_key not in cache:
            # Create retriever and get context (following your pattern)
            search = make_search(topic)
            retrieved = search(question)
            context = " ".join(retrieved.passages)
            
            # Generate prediction with retrieved context
            pred = pragmatic_qa(question=question, conversation_history="", context=context)
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

# 4.4.2 - Evaluate the model on all questions with conversation history
def evaluate_llm_all_questions(dataset):
    examples = []
    predictions = []

    cache_file = "llm_all_questions_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Initialize the multistep prompting module
    multistep_qa = MultistepQA()

    for conversation in dataset:
        topic = conversation['topic']
        if topic not in folders:
            continue

        search = make_search(topic)
        conversation_so_far = []

        for qa_index, qa in enumerate(conversation['qas']):
            question = qa['q']
            gold_answer = qa['a']

            conversation_history = format_conversation_history(conversation_so_far)

            # Use topic and question index instead of full history
            cache_key = f"{topic}|{qa_index}|{question}"

            if cache_key not in cache:
                passages = search(question).passages
                context = " ".join(passages)

                pred_answer = multistep_qa(question=question, context=context, conversation_history=conversation_history)
                cache[cache_key] = pred_answer

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            else:
                pred_answer = cache[cache_key]

            example = dspy.Example(question=question, response=gold_answer)
            prediction = dspy.Example(question=question, response=pred_answer)

            examples.append(example)
            predictions.append(prediction)

            # Add this QA to conversation history for next questions
            conversation_so_far.append({'q': question, 'a': gold_answer})

    return examples, predictions










# 4.4.1 Evaluation
print("4.4.1 - Evaluating LLM on first questions (same scope as Part 1)...")
val_examples_llm_first, val_predictions_llm_first = evaluate_llm_first_questions(val_data)

# Cache for LLM scores
llm_score_cache_file = "llm_val_score_cache.json"
if os.path.exists(llm_score_cache_file):
    with open(llm_score_cache_file, "r", encoding="utf-8") as f:
        llm_score_cache = json.load(f)
else:
    llm_score_cache = {}

# Evaluate LLM first questions scores
llm_first_scores = {"precision": [], "recall": [], "f1": []}
for example, prediction in zip(val_examples_llm_first, val_predictions_llm_first):
    cache_key = f"LLM_First|{example.question}|{prediction.response}"
    
    if cache_key not in llm_score_cache:
        result = metric(example, prediction)
        score = [result["precision"], result["recall"], result["f1"]]
        llm_score_cache[cache_key] = score
        with open(llm_score_cache_file, "w", encoding="utf-8") as f:
            json.dump(llm_score_cache, f, ensure_ascii=False, indent=2)
    else:
        score = llm_score_cache[cache_key]
    
    llm_first_scores["precision"].append(score[0])
    llm_first_scores["recall"].append(score[1])
    llm_first_scores["f1"].append(score[2])

llm_first_results = {
    'precision': sum(llm_first_scores['precision']) / len(llm_first_scores['precision']),
    'recall': sum(llm_first_scores['recall']) / len(llm_first_scores['recall']),
    'f1': sum(llm_first_scores['f1']) / len(llm_first_scores['f1']),
    'count': len(llm_first_scores['f1'])
}








# 4.4.2 Evaluation
print("\n4.4.2 - Evaluating LLM on all questions with conversational context...")
val_examples_llm_all, val_predictions_llm_all = evaluate_llm_all_questions(val_data)

# Evaluate LLM all questions scores
llm_all_scores = {"precision": [], "recall": [], "f1": []}
for example, prediction in zip(val_examples_llm_all, val_predictions_llm_all):
    cache_key = f"LLM_All|{example.question}|{prediction.response}"
    
    if cache_key not in llm_score_cache:
        result = metric(example, prediction)
        score = [result["precision"], result["recall"], result["f1"]]
        llm_score_cache[cache_key] = score
        with open(llm_score_cache_file, "w", encoding="utf-8") as f:
            json.dump(llm_score_cache, f, ensure_ascii=False, indent=2)
    else:
        score = llm_score_cache[cache_key]
    
    llm_all_scores["precision"].append(score[0])
    llm_all_scores["recall"].append(score[1])
    llm_all_scores["f1"].append(score[2])

llm_all_results = {
    'precision': sum(llm_all_scores['precision']) / len(llm_all_scores['precision']),
    'recall': sum(llm_all_scores['recall']) / len(llm_all_scores['recall']),
    'f1': sum(llm_all_scores['f1']) / len(llm_all_scores['f1']),
    'count': len(llm_all_scores['f1'])
}







# Display comprehensive results
print("\n4.4.1 Comparison - First Questions Only:")
print("Configuration      | Precision | Recall | F1 Score | Count")
print("-" * 65)

# Print Part 1 results
for config, results in val_results.items():
    print(f"{config:<18} | {results['precision']:.4f}   | {results['recall']:.4f}  | {results['f1']:.4f}   | {results['count']}")

# Print LLM results
print(f"{'LLM (DSPy)':<18} | {llm_first_results['precision']:.4f}   | {llm_first_results['recall']:.4f}  | {llm_first_results['f1']:.4f}   | {llm_first_results['count']}")

print(f"\n4.4.2 Results - All Questions with Conversational Context:")
print(f"{'LLM (All + Ctx)':<18} | {llm_all_results['precision']:.4f}   | {llm_all_results['recall']:.4f}  | {llm_all_results['f1']:.4f}   | {llm_all_results['count']}")

print(f"\nAnalysis:")
print(f"4.4.1 - LLM vs Traditional on first questions:")
best_traditional = max(val_results.values(), key=lambda x: x['f1'])
print(f"  LLM F1: {llm_first_results['f1']:.4f} vs Best Traditional: {best_traditional['f1']:.4f}")
print(f"  Difference: {llm_first_results['f1'] - best_traditional['f1']:.4f} F1 points")

print(f"4.4.2 - Impact of conversational context:")
print(f"  LLM First Questions: {llm_first_results['f1']:.4f} F1")
print(f"  LLM All + Context:   {llm_all_results['f1']:.4f} F1")
print(f"  Context Impact: {llm_all_results['f1'] - llm_first_results['f1']:.4f} F1 points")

print(f"\nMetric used for optimization: SemanticF1")
print("- Measures semantic similarity rather than exact string matching")
print("- Provides precision, recall, and F1 for comprehensive evaluation") 
print("- Designed for conversational QA with multiple valid phrasings")
print("- Captures pragmatic relevance of generated information")
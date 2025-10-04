# generation.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import json
import warnings
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

warnings.filterwarnings("ignore")

class QuestionGenerator:
    def __init__(self, model_path='models/t5-squad-final'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Loading model on device: {self.device}")
        
        # Load the fine-tuned model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_questions(self, context, num_questions=5, max_length=128):
        """Generate questions from context using different decoding strategies"""
        
        # Format input text same as training
        input_text = f"answer_question: {context}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        results = {}
        
        # 1. Greedy decoding - use beam search with num_beams=1 for multiple sequences
        with torch.no_grad():
            greedy_outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=num_questions,  # Use beam search for multiple sequences
                num_return_sequences=num_questions,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        greedy_questions = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in greedy_outputs
        ]
        results['greedy'] = greedy_questions
        
        # 2. Top-k sampling
        with torch.no_grad():
            topk_outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=num_questions,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        topk_questions = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in topk_outputs
        ]
        results['top_k'] = topk_questions
        
        # 3. Top-p sampling
        with torch.no_grad():
            topp_outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=num_questions,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        topp_questions = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in topp_outputs
        ]
        results['top_p'] = topp_questions
        
        return results

class TextEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
    
    def compute_bleu(self, reference, hypothesis):
        """Compute BLEU score"""
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        return sentence_bleu(
            [ref_tokens], 
            hyp_tokens, 
            smoothing_function=self.smoothing.method1
        )
    
    def compute_rouge(self, reference, hypothesis):
        """Compute ROUGE scores"""
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_meteor(self, reference, hypothesis):
        """Compute METEOR score"""
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        return meteor_score([ref_tokens], hyp_tokens)

def load_test_data():
    """Load test data for evaluation"""
    try:
        test_data = pd.read_csv('data/test.csv')
        print(f"Loaded {len(test_data)} test samples")
        return test_data
    except FileNotFoundError:
        print("Warning: test.csv not found. Using sample references for evaluation.")
        return None

def evaluate_generations(test_data=None, num_samples=100):
    """Evaluate generated questions against reference questions"""
    
    evaluator = TextEvaluator()
    
    # Use test data if available, otherwise use sample references
    if test_data is not None:
        # Take a subset for evaluation
        eval_subset = test_data.sample(n=min(num_samples, len(test_data)), random_state=42)
        reference_questions = eval_subset['question'].tolist()
    else:
        reference_questions = [
            "What is artificial intelligence designed to do?",
            "When did the Renaissance take place?",
            "What has been the main driver of climate change since the 1800s?",
            "What type of play is Romeo and Juliet?",
            "What does photosynthesis create from sunlight and carbon dioxide?"
        ] * 20  # Repeat to get 100 samples
        reference_questions = reference_questions[:num_samples]
    
    results = {}
    
    for method in ['greedy', 'top_k', 'top_p']:
        print(f"\nEvaluating {method.upper()} method...")
        
        # Read generated questions
        with open(f'samples/{method}_questions.txt', 'r') as f:
            generated_questions = [line.strip() for line in f if line.strip()]
        
        # Take same number as reference questions
        generated_questions = generated_questions[:len(reference_questions)]
        
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        meteor_scores = []
        
        for ref_q, gen_q in zip(reference_questions, generated_questions):
            if not gen_q.strip():  # Skip empty questions
                continue
                
            # Compute metrics
            bleu = evaluator.compute_bleu(ref_q, gen_q)
            rouge = evaluator.compute_rouge(ref_q, gen_q)
            meteor = evaluator.compute_meteor(ref_q, gen_q)
            
            bleu_scores.append(bleu)
            rouge1_scores.append(rouge['rouge1'])
            rouge2_scores.append(rouge['rouge2'])
            rougeL_scores.append(rouge['rougeL'])
            meteor_scores.append(meteor)
        
        # Calculate averages
        avg_metrics = {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'meteor': np.mean(meteor_scores)
        }
        
        results[method] = avg_metrics
        
        print(f"  BLEU: {avg_metrics['bleu']:.4f}")
        print(f"  ROUGE-1: {avg_metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {avg_metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {avg_metrics['rougeL']:.4f}")
        print(f"  METEOR: {avg_metrics['meteor']:.4f}")
    
    return results

def find_best_method(results):
    """Find best performing method based on average of all metrics"""
    
    method_averages = {}
    
    for method, metrics in results.items():
        # Calculate average of all metrics
        avg_score = np.mean(list(metrics.values()))
        method_averages[method] = avg_score
    
    # Find best method
    best_method = max(method_averages.items(), key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print("BEST PERFORMING METHOD")
    print(f"{'='*60}")
    print(f"Method: {best_method[0].upper()}")
    print(f"Average Score: {best_method[1]:.4f}")
    print()
    
    # Show detailed metrics for best method
    best_metrics = results[best_method[0]]
    print("Detailed Metrics:")
    print(f"  BLEU: {best_metrics['bleu']:.4f}")
    print(f"  ROUGE-1: {best_metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {best_metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {best_metrics['rougeL']:.4f}")
    print(f"  METEOR: {best_metrics['meteor']:.4f}")
    
    return best_method[0]

def main():
    # Create samples directory
    os.makedirs('samples', exist_ok=True)
    
    # Initialize generator
    try:
        generator = QuestionGenerator()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model is trained and saved in 'models/t5-squad-final'")
        return
    
    # Load test data
    test_data = load_test_data()
    
    # Use first 10 contexts from test data if available, otherwise use samples
    if test_data is not None:
        sample_contexts = test_data['context'].head(10).tolist()
    else:
        sample_contexts = [
            "Artificial intelligence is a branch of computer science that aims to create intelligent machines that work and react like humans. Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning and problem solving.",
            
            "The Renaissance was a period of cultural, artistic, political and economic revival following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art.",
            
            "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate changes may be natural, human activities have been the main driver of climate change since the 1800s, primarily due to the burning of fossil fuels like coal, oil and gas.",
            
            "Shakespeare's Romeo and Juliet is a tragedy written early in the career of playwright William Shakespeare about two young star-crossed lovers whose deaths ultimately reconcile their feuding families.",
            
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar. During photosynthesis, plants take in carbon dioxide and water from the air and soil."
        ]
    
    print("="*60)
    print("CONDITIONAL TEXT GENERATION - QUESTION GENERATION")
    print("="*60)
    
    # Generate questions for all contexts
    all_greedy = []
    all_topk = []
    all_topp = []
    
    for i, context in enumerate(sample_contexts):
        print(f"\nProcessing sample {i+1}/{len(sample_contexts)}...")
        
        questions = generator.generate_questions(context, num_questions=10)
        
        all_greedy.extend(questions['greedy'])
        all_topk.extend(questions['top_k'])
        all_topp.extend(questions['top_p'])
    
    print(f"\nGeneration completed!")
    
    # Save to separate txt files
    with open('samples/greedy_questions.txt', 'w') as f:
        for q in all_greedy:
            f.write(f"{q}\n")
    
    with open('samples/top_k_questions.txt', 'w') as f:
        for q in all_topk:
            f.write(f"{q}\n")
    
    with open('samples/top_p_questions.txt', 'w') as f:
        for q in all_topp:
            f.write(f"{q}\n")
    
    print(f"\nQuestions saved to:")
    print(f"  - samples/greedy_questions.txt ({len(all_greedy)} questions)")
    print(f"  - samples/top_k_questions.txt ({len(all_topk)} questions)")  
    print(f"  - samples/top_p_questions.txt ({len(all_topp)} questions)")
    
    # Evaluate all methods
    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}")
    
    results = evaluate_generations(test_data, num_samples=min(100, len(all_greedy)))
    
    # Find and display best method
    best_method = find_best_method(results)
    
    print(f"\nGeneration complete! Best method: {best_method.upper()}")

if __name__ == "__main__":
    main()

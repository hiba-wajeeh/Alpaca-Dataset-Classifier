import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import random

# Define your subject categories
CATEGORIES = [
    "history",
    "science", 
    "mathematics",
    "literature",
    "technology",
    "business",
    "health_medicine",
    "geography",
    "arts",
    "sports",
    "law",
    "psychology",
    "philosophy",
    "education",
    "cooking_food",
    "travel",
    "entertainment",
    "language",
    "general_knowledge",
    "creative_writing",
    "other"
]

def classify_instruction(instruction, input_text="", model="llama3.2:3b"):
    """Classify instruction into a subject category using Ollama"""
    
    categories_str = ", ".join(CATEGORIES)
    
    prompt = f"""Classify the following instruction into exactly ONE subject category from this list:
{categories_str}

Instruction: {instruction}
{f"Input: {input_text}" if input_text else ""}

Respond with ONLY the category name, nothing else.
Category:"""
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10
                }
            },
            timeout=30
        )
        
        category = response.json()['response'].strip().lower()
        category = category.replace("category:", "").strip()
        category = category.split()[0] if category.split() else "other"
        
        if category not in CATEGORIES:
            return "other"
        
        return category
        
    except Exception as e:
        print(f"Error: {e}")
        return "other"

def classify_alpaca_dataset():
    """Main function to classify the entire Alpaca dataset"""
    
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")
    
    results = []
    category_counts = Counter()
    
    print(f"Classifying {len(dataset['train'])} instructions...")
    
    failed_count = 0
    
    for i, example in enumerate(tqdm(dataset['train'])):
        # Retry logic with increasing timeout
        max_retries = 3
        category = None
        
        for attempt in range(max_retries):
            try:
                timeout_val = 60 * (attempt + 1)  # 60s, 120s, 180s
                
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        "model": "qwen2.5:3b",  # ← CHANGED THIS
                        "prompt": f"""Classify into ONE category: {', '.join(CATEGORIES)}

Instruction: {example['instruction']}
{f"Input: {example['input']}" if example['input'] else ""}

Category:""",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 10
                        }
                    },
                    timeout=timeout_val
                )
                
                category = response.json()['response'].strip().lower()
                category = category.replace("category:", "").strip()
                category = category.split()[0] if category.split() else "other"
                
                if category not in CATEGORIES:
                    category = "other"
                
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nFailed after {max_retries} attempts: {e}")
                    category = "other"
                    failed_count += 1
                else:
                    print(f"\nRetrying ({attempt + 1}/{max_retries})...")
        
        results.append({
            'instruction': example['instruction'],
            'input': example['input'],
            'output': example['output'],
            'category': category
        })
        
        category_counts[category] += 1
        
        # Save progress every 500 examples
        if (i + 1) % 500 == 0:
            with open(f'alpaca_categorized_checkpoint_{i+1}.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nCheckpoint saved at {i+1} examples")
            print(f"Failed so far: {failed_count}")
    
    # Save final results
    with open('alpaca_categorized_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal failed classifications: {failed_count}")
    print("\n=== Final Category Distribution ===")
    for category, count in category_counts.most_common():
        print(f"{category}: {count} ({count/len(results)*100:.1f}%)")
    
    # Create separate files for each category
    categorized_data = {cat: [] for cat in CATEGORIES}
    for item in results:
        categorized_data[item['category']].append(item)
    
    for category, items in categorized_data.items():
        if items:
            with open(f'alpaca_{category}.json', 'w') as f:
                json.dump(items, f, indent=2)
    
    print(f"\n✓ Classification complete!")
    return results

def verify_categorization(results_file='alpaca_categorized_final.json'):
    """Check random samples from each category"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    by_category = {}
    for item in data:
        cat = item['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)
    
    print("\n=== RANDOM SAMPLES FROM EACH CATEGORY ===\n")
    
    for category in sorted(by_category.keys()):
        items = by_category[category]
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category.upper()} ({len(items)} items)")
        print('='*60)
        
        samples = random.sample(items, min(3, len(items)))
        for i, sample in enumerate(samples, 1):
            print(f"\n[Example {i}]")
            print(f"Instruction: {sample['instruction']}")
            if sample['input']:
                print(f"Input: {sample['input'][:100]}...")
            print()

def check_quality(results_file='alpaca_categorized_final.json'):
    """Check for potential classification errors"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n=== QUALITY CHECKS ===\n")
    
    # Check 1: Too many in "other" category
    other_count = sum(1 for item in data if item['category'] == 'other')
    other_pct = (other_count / len(data)) * 100
    print(f"1. Items in 'other' category: {other_count} ({other_pct:.1f}%)")
    if other_pct > 15:
        print("   ⚠️  WARNING: High percentage in 'other'")
    else:
        print("   ✓ Good distribution")
    
    # Check 2: Empty categories
    by_category = {}
    for item in data:
        cat = item['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)
    
    empty_cats = [cat for cat in CATEGORIES if cat not in by_category or len(by_category[cat]) == 0]
    print(f"\n2. Empty categories: {len(empty_cats)}")
    if empty_cats:
        print(f"   Categories: {', '.join(empty_cats)}")

def create_html_report(results_file='alpaca_categorized_final.json'):
    """Create an HTML report for easy browsing"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    html = f"""
    <html>
    <head>
        <title>Alpaca Classification Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
            h1 {{ color: #333; }}
            .category {{ margin: 20px 0; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .instruction {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-left: 4px solid #007bff; }}
            .count {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Alpaca Dataset Classification Report</h1>
        <p><strong>Total instructions:</strong> {len(data)}</p>
    """
    
    # Group by category
    by_category = {}
    for item in data:
        cat = item['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)
    
    # Add each category
    for category in sorted(by_category.keys()):
        items = by_category[category]
        html += f"""
        <div class="category">
            <h2>{category.replace('_', ' ').title()} <span class="count">({len(items)} instructions)</span></h2>
        """
        
        for item in items[:5]:
            html += f"""
            <div class="instruction">
                <strong>Instruction:</strong> {item['instruction']}<br>
                {f"<strong>Input:</strong> {item['input'][:200]}...<br>" if item['input'] else ""}
            </div>
            """
        
        html += "</div>"
    
    html += "</body></html>"
    
    with open('classification_report.html', 'w') as f:
        f.write(html)
    
    print("\n✓ Created classification_report.html")

if __name__ == "__main__":
    # Check Ollama connection
    print("Checking Ollama connection...")
    try:
        response = requests.get('http://localhost:11434/api/tags')
        print("✓ Ollama is running\n")
    except:
        print("✗ Ollama is not running!")
        print("  Start it with: ollama serve")
        print("  Then pull a model: ollama pull llama3.2:3b")
        exit(1)
    
    # Run classification
    results = classify_alpaca_dataset()
    
    # Run verification
    print("\n" + "="*60)
    print("Running verification checks...")
    print("="*60)
    
    verify_categorization()
    check_quality()
    create_html_report()
    
    print("\n✅ ALL DONE!")
    print("📊 Open 'classification_report.html' in your browser to see results")
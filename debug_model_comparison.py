"""
Debug script to compare model outputs and diagnose grading discrepancies.

Run this to see what different models are actually outputting.
"""
import json
import re
from src.llm import OllamaClient
from src.prompts.ksb_templates import KSBPromptTemplates, extract_grade_from_evaluation

def test_model_output(model_name: str, evidence_sample: str, ksb_code: str = "K1"):
    """Test how a specific model responds to the evaluation prompt."""

    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print(f"{'='*80}\n")

    # Create LLM client
    llm = OllamaClient(model=model_name, timeout=180)

    # Use the actual prompt template
    prompt = KSBPromptTemplates.format_ksb_evaluation(
        ksb_code=ksb_code,
        ksb_title="Test Knowledge Area",
        pass_criteria="Demonstrates understanding of core concepts with clear explanation",
        merit_criteria="Demonstrates deep understanding with critical analysis and application",
        referral_criteria="Does not demonstrate sufficient understanding",
        evidence_text=evidence_sample,
        brief_context="Task 1: Explain the concept and provide examples"
    )

    system_prompt = KSBPromptTemplates.get_system_prompt()

    try:
        # Generate response
        print("Generating response...\n")
        response = llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1500
        )

        # Show response
        print("RAW RESPONSE:")
        print("-" * 80)
        print(response[:1000])  # First 1000 chars
        if len(response) > 1000:
            print(f"\n... (truncated, total length: {len(response)} chars)")
        print("-" * 80)

        # Try to extract grade
        extracted = extract_grade_from_evaluation(response)

        print("\nEXTRACTED GRADE DATA:")
        print("-" * 80)
        print(f"Grade: {extracted['grade']}")
        print(f"Confidence: {extracted['confidence']}")
        print(f"Extraction Method: {extracted['method']}")
        print(f"Possible Hallucination: {extracted['possible_hallucination']}")

        if extracted.get('raw_json'):
            print(f"\nParsed JSON:")
            print(json.dumps(extracted['raw_json'], indent=2))

        print("-" * 80)

        # Check for warning signs
        warnings = []
        if extracted['method'] == 'heuristic':
            warnings.append("⚠️ Fell back to heuristic extraction (JSON parsing failed)")
        if 'NOT FOUND' in response.upper():
            warnings.append("⚠️ Model claims 'NOT FOUND' for evidence")
        if response.upper().count('❌') > response.upper().count('✅'):
            warnings.append("⚠️ More ❌ than ✅ in response")
        if extracted['possible_hallucination']:
            warnings.append("⚠️ Hallucination indicators detected")

        if warnings:
            print("\nWARNING SIGNS:")
            for w in warnings:
                print(f"  {w}")

        return {
            'model': model_name,
            'grade': extracted['grade'],
            'confidence': extracted['confidence'],
            'method': extracted['method'],
            'response_length': len(response),
            'warnings': warnings
        }

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {'model': model_name, 'error': str(e)}


def compare_models(models: list, evidence_sample: str):
    """Compare multiple models side-by-side."""

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")

    results = []
    for model in models:
        try:
            result = test_model_output(model, evidence_sample)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to test {model}: {e}\n")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Grade':<10} {'Confidence':<12} {'Method':<15} {'Warnings'}")
    print("-" * 80)

    for r in results:
        if 'error' in r:
            print(f"{r['model']:<20} ERROR: {r['error']}")
        else:
            warnings_str = f"{len(r.get('warnings', []))} warnings" if r.get('warnings') else "OK"
            print(f"{r['model']:<20} {r['grade']:<10} {r['confidence']:<12} {r['method']:<15} {warnings_str}")

    print("="*80 + "\n")


if __name__ == "__main__":
    # Sample evidence (replace with actual student work)
    sample_evidence = """
    **Section 2: Data Analysis**

    In this section, I explored the dataset using Python and pandas. I loaded the CSV file
    and performed initial exploratory data analysis (EDA). The dataset contains 1,000 rows
    and 15 columns including customer demographics and purchase history.

    I calculated summary statistics using df.describe() which showed that the average
    customer age is 42 years with a standard deviation of 12. I also created visualizations
    using matplotlib to show the distribution of ages and purchase amounts.

    **Evidence E1:** "The mean customer age is 42 years (sd=12)" (Section 2)
    **Evidence E2:** "Created bar charts and histograms to visualize the data distribution" (Section 2)
    """

    # Your actual models from ollama list
    models_to_test = [
        "mistral:7b",        # ✅ Working baseline (4.4 GB, 7B params)
        "gpt-oss:20b",       # ⚠️ Too strict despite being 20B (13 GB)
        "qwen3-vl:4b",       # ⚠️ Vision model but only 4B (3.3 GB)
        "gemma3:4b",         # ❌ Not recommended - below minimum (3.3 GB)
    ]

    print("KSB MARKER - MODEL DIAGNOSTIC TOOL")
    print("This will test each model with the same evidence and show differences.\n")

    # Run comparison
    compare_models(models_to_test, sample_evidence)

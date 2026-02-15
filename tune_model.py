"""
Interactive model tuning tool.

This helps you find the right temperature and settings for a specific model.
"""
import sys
from src.llm import OllamaClient
from src.prompts.ksb_templates import KSBPromptTemplates, extract_grade_from_evaluation

# Sample evidence (realistic student work)
SAMPLE_EVIDENCE = """
**Section 2: Data Preprocessing**

I cleaned the dataset by handling missing values. For numeric columns like 'age' and 'income',
I used the median imputation strategy because it's robust to outliers. For categorical columns
like 'gender' and 'region', I used mode imputation.

I also identified outliers using the IQR method. Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
were flagged. After analysis, I removed 47 outliers (4.7% of data) which were data entry errors.

**Section 3: Feature Engineering**

I created new features from existing ones. For example, I calculated 'age_group' by binning
ages into categories: 18-25, 26-35, 36-50, 51+. I also created 'purchase_frequency' by
counting purchases per customer in the last 6 months.

**Evidence:**
- [E1] "Used median imputation for numeric columns" (Section 2)
- [E2] "Applied IQR method for outlier detection" (Section 2)
- [E3] "Created age_group and purchase_frequency features" (Section 3)
"""

def test_temperature(model_name: str, temperature: float, max_tokens: int = 1500):
    """Test a specific temperature setting."""

    print(f"\nTesting {model_name} with temperature={temperature}, max_tokens={max_tokens}")
    print("-" * 80)

    llm = OllamaClient(model=model_name, timeout=180)

    # Use realistic KSB criteria
    prompt = KSBPromptTemplates.format_ksb_evaluation(
        ksb_code="K2",
        ksb_title="Data preprocessing and cleaning techniques",
        pass_criteria="Demonstrates appropriate data cleaning techniques with clear explanation of methods used",
        merit_criteria="Demonstrates sophisticated data cleaning with critical evaluation of technique choices and impact on data quality",
        referral_criteria="Does not demonstrate adequate understanding of data cleaning techniques",
        evidence_text=SAMPLE_EVIDENCE,
        brief_context="Task 1: Clean the provided dataset and document your preprocessing steps"
    )

    system_prompt = KSBPromptTemplates.get_system_prompt()

    try:
        response = llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Extract grade
        extracted = extract_grade_from_evaluation(response)

        print(f"✓ Grade: {extracted['grade']}")
        print(f"  Confidence: {extracted['confidence']}")
        print(f"  Extraction Method: {extracted['method']}")

        # Show first few lines of reasoning
        lines = response.split('\n')
        print(f"\n  First few lines of response:")
        for line in lines[:5]:
            if line.strip():
                print(f"    {line[:100]}")

        # Check JSON parsing
        if extracted['method'] == 'json':
            print(f"  ✓ JSON parsed successfully")
        elif extracted['method'] == 'heuristic':
            print(f"  ⚠️ WARNING: Fell back to heuristic (JSON failed)")

        return extracted['grade'], extracted['method']

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return None, None


def tune_model_interactive(model_name: str):
    """Interactive tuning session."""

    print("\n" + "="*80)
    print(f"MODEL TUNING: {model_name}")
    print("="*80)
    print("\nThis sample should get PASS or MERIT (student did well)")
    print("If you get REFERRAL, the model is too strict.\n")

    # Test range of temperatures
    temperatures = [0.1, 0.15, 0.2, 0.25, 0.3]

    results = []
    print("\nTesting different temperatures...")
    for temp in temperatures:
        grade, method = test_temperature(model_name, temp)
        if grade:
            results.append((temp, grade, method))

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Temperature':<12} {'Grade':<10} {'Method':<15} {'Recommendation'}")
    print("-" * 80)

    for temp, grade, method in results:
        recommendation = ""
        if grade == "REFERRAL":
            recommendation = "❌ Too strict - try higher temp"
        elif grade in ["PASS", "MERIT"] and method == "json":
            recommendation = "✅ Good!"
        elif grade in ["PASS", "MERIT"] and method != "json":
            recommendation = "⚠️ Grade OK but JSON failed"

        print(f"{temp:<12.2f} {grade:<10} {method:<15} {recommendation}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find best temperature (PASS/MERIT with JSON parsing)
    best = None
    for temp, grade, method in results:
        if grade in ["PASS", "MERIT"] and method == "json":
            if best is None:
                best = temp
            # Prefer lower temperature if both work
            elif temp < best:
                best = temp

    if best:
        print(f"\n✅ Recommended temperature: {best}")
        print(f"\nAdd this to config.py MODEL_PROFILES:")
        print(f"""
    "{model_name}": {{
        "temperature": {best},
        "max_tokens": 1500,
        "strictness_adjustment": "lenient",  # Adjust if needed
        "notes": "Tuned with tune_model.py"
    }},
""")
    else:
        print("\n⚠️ No configuration worked well.")
        print("This model may not be suitable for KSB assessment.")
        print("Consider using mistral:7b or a larger model (8B+ parameters).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python tune_model.py <model_name>")
        print("\nExample:")
        print("  python tune_model.py gpt-oss")
        print("  python tune_model.py llama3:8b")
        print("\nThis will test different temperatures and recommend settings.\n")
        sys.exit(1)

    model_name = sys.argv[1]
    tune_model_interactive(model_name)

"""
Test CSV export functionality.

Quick test to verify the CSV export generates correct format.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.ksb_app import export_results_to_csv

# Sample results (mimicking actual pipeline output)
sample_results = {
    "scoring_results": [
        {
            "ksb_code": "K1",
            "ksb_title": "Data analysis principles and techniques",
            "grade": "MERIT",
            "confidence": "HIGH",
            "weighted_score": 0.85,
            "pass_met": True,
            "merit_met": True,
            "rationale": "Pass met. Merit met. Evidence: strong.",
            "gaps_identified": ["Could include more statistical tests"]
        },
        {
            "ksb_code": "S15",
            "ksb_title": "Apply data visualization techniques",
            "grade": "PASS",
            "confidence": "MEDIUM",
            "weighted_score": 0.65,
            "pass_met": True,
            "merit_met": False,
            "rationale": "Pass met. Merit not met. Evidence: adequate.",
            "gaps_identified": ["Missing interactive visualizations", "Limited color theory application"]
        },
        {
            "ksb_code": "B5",
            "ksb_title": "Professional and ethical conduct",
            "grade": "REFERRAL",
            "confidence": "LOW",
            "weighted_score": 0.3,
            "pass_met": False,
            "merit_met": False,
            "rationale": "Pass NOT met. Merit not met. Evidence: weak.",
            "gaps_identified": ["No discussion of ethical considerations", "Missing GDPR compliance"]
        }
    ],
    "feedback_results": [
        {
            "ksb_code": "K1",
            "strengths": [
                "Clear explanation of statistical concepts",
                "Good use of appropriate analysis methods",
                "Strong interpretation of results"
            ],
            "improvements": [
                {"suggestion": "Include more advanced statistical tests like ANOVA"},
                {"suggestion": "Discuss assumptions behind chosen tests"}
            ]
        },
        {
            "ksb_code": "S15",
            "strengths": [
                "Effective use of bar charts and line graphs"
            ],
            "improvements": [
                {"suggestion": "Add interactive visualizations using libraries like Plotly"},
                {"suggestion": "Apply color theory principles for better accessibility"}
            ]
        },
        {
            "ksb_code": "B5",
            "strengths": [],
            "improvements": [
                {"suggestion": "Discuss ethical considerations when handling personal data"},
                {"suggestion": "Reference GDPR compliance requirements"},
                {"suggestion": "Include a section on professional conduct in data science"}
            ]
        }
    ],
    "overall_summary": {
        "total_ksbs": 3,
        "merit_count": 1,
        "pass_count": 1,
        "referral_count": 1,
        "overall_recommendation": "REFERRAL",
        "confidence": "MEDIUM",
        "key_strengths": [
            "Strong statistical analysis skills (K1)",
            "Clear data visualization fundamentals (S15)",
            "Good technical implementation overall"
        ],
        "priority_improvements": [
            "Address ethical considerations and GDPR compliance (B5)",
            "Enhance visualizations with interactivity (S15)",
            "Include more advanced statistical methods (K1)"
        ]
    }
}


def test_csv_export():
    """Test the CSV export function."""

    print("Testing CSV Export Functionality")
    print("=" * 80)

    # Generate CSV
    csv_output = export_results_to_csv(sample_results, module_code="MLCC")

    # Show first 50 lines
    lines = csv_output.split('\n')
    print(f"\nGenerated CSV ({len(lines)} lines total):\n")
    print("First 50 lines:")
    print("-" * 80)

    for i, line in enumerate(lines[:50], 1):
        print(f"{i:3d}: {line}")

    if len(lines) > 50:
        print(f"\n... ({len(lines) - 50} more lines)")

    print("\n" + "=" * 80)
    print("CSV Export Test Summary:")
    print(f"  ✓ Total lines: {len(lines)}")
    print(f"  ✓ Header row: {lines[0][:80]}...")
    print(f"  ✓ Data rows: {len([l for l in lines if l.strip() and not l.startswith('OVERALL') and not l.startswith('KEY') and not l.startswith('PRIORITY') and l != lines[0]])} KSBs")
    print(f"  ✓ Summary included: {'OVERALL SUMMARY' in csv_output}")
    print(f"  ✓ Strengths included: {'KEY STRENGTHS' in csv_output}")
    print(f"  ✓ Improvements included: {'PRIORITY IMPROVEMENTS' in csv_output}")

    # Save to file for manual inspection
    output_file = Path(__file__).parent / "test_export.csv"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(csv_output)

    print(f"\n✓ Sample CSV saved to: {output_file}")
    print(f"  Open in Excel/LibreOffice to verify formatting")
    print("=" * 80)


if __name__ == "__main__":
    test_csv_export()

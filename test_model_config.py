"""
Quick test to verify model configurations are working.
"""
from config import ModelConfig

def test_model_configs():
    """Test all your models are configured correctly."""

    print("\n" + "="*80)
    print("MODEL CONFIGURATION TEST")
    print("="*80 + "\n")

    your_models = [
        "mistral:7b",
        "gpt-oss:20b",
        "qwen3-vl:4b",
        "gemma3:4b"
    ]

    print(f"{'Model':<20} {'Temp':<8} {'Max Tokens':<12} {'Adjustment':<12} {'Notes'}")
    print("-" * 80)

    for model in your_models:
        config = ModelConfig.get_model_config(model)

        temp = config.get('temperature', '???')
        max_tok = config.get('max_tokens', '???')
        adj = config.get('strictness_adjustment', '???')
        notes = config.get('notes', 'No notes')

        # Truncate notes if too long
        if len(notes) > 30:
            notes = notes[:27] + "..."

        print(f"{model:<20} {temp:<8} {max_tok:<12} {adj:<12} {notes}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Check mistral:7b (should be balanced)
    mistral_config = ModelConfig.get_model_config("mistral:7b")
    if mistral_config.get('strictness_adjustment') == 'balanced':
        print("✅ mistral:7b configured as 'balanced' (your working baseline)")
    else:
        print("⚠️ mistral:7b should be 'balanced'")

    # Check gpt-oss:20b (should be lenient)
    gpt_config = ModelConfig.get_model_config("gpt-oss:20b")
    if gpt_config.get('strictness_adjustment') == 'lenient':
        print("✅ gpt-oss:20b configured as 'lenient' (to fix strictness)")
    else:
        print("⚠️ gpt-oss:20b should be 'lenient'")

    # Check temperature differences
    mistral_temp = mistral_config.get('temperature')
    gpt_temp = gpt_config.get('temperature')

    if gpt_temp > mistral_temp:
        print(f"✅ gpt-oss:20b has higher temperature ({gpt_temp} vs {mistral_temp}) to reduce conservatism")
    else:
        print(f"⚠️ gpt-oss:20b should have higher temperature than mistral:7b")

    # Check 4B models
    qwen_config = ModelConfig.get_model_config("qwen3-vl:4b")
    gemma_config = ModelConfig.get_model_config("gemma3:4b")

    if qwen_config.get('temperature') >= 0.3 and gemma_config.get('temperature') >= 0.3:
        print("✅ 4B models have high temperature (0.3+) to compensate for small size")
    else:
        print("⚠️ 4B models should have temperature >= 0.3")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
For best results:
1. Use mistral:7b (your proven baseline)
2. Try gpt-oss:20b now with lenient config (should be less strict)
3. Avoid qwen3-vl:4b and gemma3:4b (below 7B minimum, unreliable)

If gpt-oss:20b still gives too many REFERRALs after this config:
- Run: python tune_model.py gpt-oss:20b
- This will find the optimal temperature for that specific model
""")

if __name__ == "__main__":
    test_model_configs()

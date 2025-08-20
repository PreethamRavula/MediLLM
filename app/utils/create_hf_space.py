from huggingface_hub import HfApi
HfApi().create_repo(
    repo_id="Preetham22/medi-llm",
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)

print("✅ Space ready: Preetham22/medi-llm")

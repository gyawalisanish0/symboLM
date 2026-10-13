import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernel = sys.argv[1] if len(sys.argv) > 1 else "sanishgyawali/symbolm-stage-2-grpo-training-dual-tesla-t4"
owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel)

with api.build_kaggle_client() as kaggle:
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    request = ApiListKernelSessionOutputRequest()
    request.user_name = owner_slug
    request.kernel_slug = kernel_slug
    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

log_text = response.log or ""
out_file = Path("c:/Users/user/Dev/symboLM/research/kaggle_grpo_tpu_execution.log")
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(log_text, encoding="utf-8")

print(f"Log length: {len(log_text)} characters")
print(f"Saved cleanly to: {out_file}")
print("\n=== LOG LAST 5000 CHARACTERS ===")
print(log_text[-5000:].encode("ascii", errors="replace").decode("ascii"))

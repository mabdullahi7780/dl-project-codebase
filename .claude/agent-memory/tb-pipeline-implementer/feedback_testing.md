---
name: feedback_testing
description: Testing patterns, constraints, and common pitfalls in this repository
type: feedback
---

Always use --basetemp="C:/tmp/pytest_tb_pipeline" when running pytest on this machine.

**Why:** The default pytest temp directory at C:\Users\mabdu_8h1ndel\AppData\Local\Temp\pytest-of-mabdu_8h1ndel has Windows permission errors (WinError 5). A manual basetemp sidesteps this.

**How to apply:** Add --basetemp="C:/tmp/pytest_tb_pipeline" to all pytest invocations. Ensure C:/tmp/ exists (create with mkdir -p if needed).

---

When monkey-patching transformers classes, first check if transformers is installed.

**Why:** transformers is not in requirements.txt for this project. Attempting to patch "transformers.BioGptForCausalLM.from_pretrained" raises ModuleNotFoundError if transformers is absent.

**How to apply:** Use importlib.util.find_spec("transformers") to gate the patch, or structure the test so the no-transformers path is still exercised.

---

Never use `Component10BioGPTReport.__new__()` to bypass `__init__` for test stubs.

**Why:** Component10BioGPTReport inherits from nn.Module. Assigning .model before `Module.__init__()` is called raises AttributeError from PyTorch's __setattr__ guard.

**How to apply:** Construct via Component10BioGPTReport(BioGPTConfig(use_mock=True)) to get a fully initialised instance, then use object.__setattr__() to overwrite the model field with the test stub.

from snakepp.evaluation.metrics import mrr, pass_at_1, recall_at_k
from snakepp.generation.citations import extract_citations, extract_code
from snakepp.schemas import EvalResult


def test_extract_code_and_citations():
    out = "Here:\n```python\nimport pandas as pd\n```\nSources: [2], [1], [2]"
    assert extract_code(out) == "import pandas as pd"
    assert extract_citations(out) == [2, 1]
    assert extract_citations("no sources") == []


def test_metrics():
    rs = [EvalResult(task_id=str(i), benchmark="x", config="c", passed=i < 3) for i in range(4)]
    assert pass_at_1(rs) == 0.75
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert mrr(["a", "b"], {"b"}) == 0.5

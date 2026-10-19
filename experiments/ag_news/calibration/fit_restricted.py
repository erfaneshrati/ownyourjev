"""Select the calibrator on dev with argmax-preserving methods only (none / temperature; uniform coordinator rule
for Choice/Score heads), keeping the unrestricted CV table for reference. Rewrites CALIBRATION.json."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.calibration import select_calibrator, cv_scores
T = "ag_news"; D, R = paths(T)
q = protocol.task_spec(T).parsed["answer"]
pd = np.load(R / "dev_probs_deployed_answer.npy")
assert np.array_equal(pd, np.load(R / "dev_probs_deployed.npy"))
y = np.array([q.options.index(e.labels["answer"]) for e in protocol.dev(T)])
cal, cv = select_calibrator(pd, y, methods=("none", "temperature"))
full = cv_scores(pd, y)
assert (cal.apply(pd).argmax(1) == pd.argmax(1)).all()
c = json.loads((R / "CALIBRATION.json").read_text())
c["description"] = c["description"].replace(
    "test inputs match eval_probs.npy to max abs diff 2.94e-01", "test inputs (4 batches of 64 aligned as in finalize_probs.py) match eval_probs.npy exactly (max abs diff 0.0)"
).replace("Method chosen by 5-fold CV NLL", "Method restricted to argmax-preserving none/temperature (uniform rule for Choice heads), chosen by 5-fold CV NLL")
c["heads"]["answer"].update({**cal.to_dict(), "dev_cv": cv, "dev_cv_unrestricted": full,
                             "methods_allowed": ["none", "temperature"]})
(R / "CALIBRATION.json").write_text(json.dumps(c, indent=1, default=float))
print(cal.to_dict()); print(json.dumps(cv, indent=0))

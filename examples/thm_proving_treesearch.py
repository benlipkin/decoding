try:
    from nltk.inference import TableauProver  # type: ignore[reportMissingTypeStubs]
    from nltk.sem import Expression  # type: ignore[reportMissingTypeStubs]
    from nltk.sem.logic import (  # type: ignore[reportMissingTypeStubs]
        LogicalExpressionException,
        LogicParser,
    )
except ImportError as e:
    msg = "Please install nltk with 'pip install nltk' for this example."
    raise ImportError(msg) from e

from decoding.estimators import SelfConsistency
from decoding.generators import TreeSearch
from decoding.models import LanguageModel
from decoding.pmf import CategoricalLogPMF, ScoredItem
from decoding.scorers import Scorer

llm = LanguageModel.from_id(
    "microsoft/Phi-3-mini-4k-instruct",
    gpu_memory_utilization=0.4,
)
expr = Expression.fromstring
parser = LogicParser()
prover = TableauProver()


def step_score_fn(s: str) -> ScoredItem[str]:
    if stop_pass(s):
        return ScoredItem(item=s, score=float("inf"))
    lines = s.strip().split("\n")
    last_line = lines[-1]
    if last_line.startswith(("P:", "C:")):
        stmt = last_line[2:]
        try:
            parser.parse(stmt)
            return ScoredItem(item=s, score=len(lines))
        except LogicalExpressionException:
            pass
    backtrack = "\n".join(lines[:-1]) + "\n"
    return ScoredItem(item=backtrack, score=len(lines) - 1)


def final_score_fn(d: CategoricalLogPMF[str]) -> list[ScoredItem[str]]:
    def postproc(gen: str) -> str:
        try:
            new = gen[len(prompt) - 2 :]
            stmts = new.split("\n")
            premises = [expr(s[2:].strip()) for s in stmts if s.startswith("P:")]
            conclusions = [expr(s[2:].strip()) for s in stmts if s.startswith("C:")]
            if len(premises) == 0 or len(conclusions) != 1:
                return "Error"
            if prover.prove(conclusions[0], premises):
                return "True"
            if prover.prove(conclusions[0].negate(), premises):
                return "False"
            return "Unknown"
        except Exception:  # noqa: BLE001
            return "Error"

    def filt(s: str) -> bool:
        return s != "Error"

    return SelfConsistency(d, postproc=postproc, filt=filt, parallelize=True)


def stop_pass(s: str) -> bool:
    return s.endswith("\n\n")


step_scorer = Scorer.from_f_str_to_item(step_score_fn, parallelize=True)
final_scorer = Scorer.from_f_logpmf_to_batch_item(final_score_fn)


def run(prompt: str) -> str:
    return TreeSearch(
        prompt=prompt,
        llm=llm,
        step_scorer=step_scorer,
        final_scorer=final_scorer,
        stop_cond_pass=stop_pass,
        n=10,
        beam_width=25,
        beam_factor=5,
        sync_str="\n",
        seed=42,
    )[0].item


prompt = """Formalize the following sentences into first-order logic:

English:
P: Socrates is a man.
P: All men are mortal.
C: Is Socrates mortal?

FOL:
P: man(socrates)
P: all x.(man(x) -> mortal(x))
C: mortal(socrates)

English:
P: All rectangles have four sides.
P: All four-sided things are quadrilaterals.
C: Is a rectangle a quadrilateral?

FOL:
P:"""

out = run(prompt)
assert out == "True"

print("PASSED")

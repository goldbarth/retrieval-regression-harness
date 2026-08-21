"""The answer contract the model has to fill for a RAG run.

Citations are a schema field, not a prose convention. The smoke run on
2026-08-19 produced three different citation formats across four answers, two
of them without a separator, so nothing downstream could parse them. A field
moves that from "the model usually formats it this way" to "the provider
rejects anything else".
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from harness.core.gold import ExpectedSource

RagAnswerStatus = Literal["answered", "no_relevant_sections"]
"""Whether the answer is grounded at all, as a closed set.

Two states, not one. "no_relevant_sections" says the model deliberately answered
without a source because retrieval returned nothing usable. That is not a
measurement of the ranking, so phase 3 has to report it separately instead of
counting it as a miss, the same reasoning as JudgeVerdict.unclear.
"""


class RagAnswer(BaseModel):
    """What the model returns for one question. Not the HTTP response.

    Kept apart from schemas.responses.RagResponse because this model goes to
    the provider under strict mode, where the rules of
    scripts/schema_constraint_probe.py apply: no set, no tuple, no open dict,
    and no pydantic default, since strict makes every field required and a
    default therefore never fires. RagResponse is free of those rules and adds
    what the endpoint knows on its own, like num_chars and stop_reason.
    """

    model_config = ConfigDict(extra="forbid")

    answer_status: RagAnswerStatus
    answer: str

    citations: list[ExpectedSource]
    """Same shape as GoldQuestion.expected_sources, deliberately. A citation and
    an expected source have to be comparable in the phase 3 runner without a
    conversion step, because that step is where a mismatch would hide.

    A list, never a set: a set renders as `uniqueItems`, which strict mode
    rejects outright. Duplicates are therefore possible here and are not
    validated away, unlike in GoldQuestion. The gold file is authored once and
    a duplicate there is a typo; a repeated citation is the model's output and
    a fact about the run, so the runner has to see it rather than have it
    silently removed on the way in.
    """

    @model_validator(mode="after")
    def _citations_match_the_status(self) -> Self:
        """Enforce the rule that spans two fields, which the schema cannot.

        A Field(min_length=1) on citations would be the wrong tool twice over.
        It is the wrong shape, because the condition is not a property of the
        list but a relation between the list and answer_status. And it would be
        actively harmful: the prompt tells the model to say so when nothing
        relevant comes back, and a hard minimum of one leaves it exactly one way
        to comply, which is to invent a citation. A fabricated citation is
        indistinguishable from a real one further downstream and would corrupt
        recall@k silently, which is worse than any error this validator raises.
        """
        if self.answer_status == "answered" and not self.citations:
            raise ValueError("An answered response must cite at least one section")
        if self.answer_status == "no_relevant_sections" and self.citations:
            raise ValueError(
                "A no_relevant_sections response must not cite any section"
            )
        return self

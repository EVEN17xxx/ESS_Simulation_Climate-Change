"""ClimateAgent: an ESS-sampled LLM citizen (or stubborn FCA/denial actor)."""
from mesa import Agent

class ClimateAgent(Agent):
    # LLM-powered citizen sampled from the ESS Round 10 respondent pool, or a
    # stubborn actor (FCA / denial). Minimal state: single LLM update per step,
    # a K-item raw sliding memory window (no summarization).
    #   camp    : "high_concern"/"low_concern"/"mid" (citizen) or "fca"/"denial" (stubborn); frozen at t=0
    #   concern : climate concern (1-5, ESS10 wrclmch: 1=not at all worried, 5=extremely worried)

    def __init__(
        self, model, unique_id: int, name: str, camp: str,
        initial_concern: float, topic: str,
        gpt_model: str, temp: float,
        initial_opinion="", initial_rationale="",
        system_prompt="", memory_window: int = 3,
        ess_row: dict = None, is_stubborn: bool = False,
    ):
        import mesa as _mesa
        if int(_mesa.__version__.split(".")[0]) >= 3:
            super().__init__(model)
            self.unique_id = unique_id
        else:
            super().__init__(unique_id, model)

        self.name          = name
        self.camp          = camp
        self.ess_row       = ess_row or {}
        self.is_stubborn   = is_stubborn   # stubborn: FCA / denial (Mobilia 2003; never updates)
        self.system_prompt = system_prompt or ""
        self.topic         = topic
        self.gpt_model     = gpt_model
        self.temp          = temp
        self.memory_window = memory_window

        self.initial_opinion   = initial_opinion or ""
        self.opinion           = self.initial_opinion
        self.opinions          = [self.initial_opinion]
        self.initial_concern   = initial_concern
        self.concern           = initial_concern
        self.concerns          = [self.concern]
        self.initial_rationale = initial_rationale or ""
        self.rationale         = self.initial_rationale
        self.rationales        = [self.initial_rationale]

        self.memory            = []   # last K raw contact opinions (deterministic sliding window)
        self.agent_interaction = []
        self.contact_ids       = []
        self.contact_camps     = []

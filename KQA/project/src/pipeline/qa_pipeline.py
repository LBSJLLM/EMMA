from __future__ import annotations

from typing import Dict

from src.pipeline.controller import QAController
from src.pipeline.single_agent_controller import SingleAgentQAController


class QAPipeline:
    def __init__(self, cfg: Dict):
        mode = str(cfg.get("qa_mode") or "").strip().lower()
        if mode == "single_agent_multiround":
            self.controller = SingleAgentQAController(cfg)
        else:
            self.controller = QAController(cfg)

    def run_one(self, question_obj: Dict):
        return self.controller.run(question_obj)

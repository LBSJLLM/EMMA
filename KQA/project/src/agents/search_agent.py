from __future__ import annotations

from typing import Dict, List

from src.memory.interfaces import Retriever


class SearchAgent:
    def __init__(self, retriever: Retriever, top_k_per_query: int, top_k_per_level: int):
        self.retriever = retriever
        self.top_k_per_query = int(top_k_per_query)
        self.top_k_per_level = int(top_k_per_level)

    def run(self, retrieval_plan: Dict) -> List[Dict]:
        out: List[Dict] = []
        queries = retrieval_plan.get("queries", [])
        if not isinstance(queries, list) or not queries:
            return out

        # Enforce the latest contract: exactly one query and one memory level per search step.
        query = queries[0] if isinstance(queries[0], dict) else {}
        query_id = query.get("query_id", "")
        query_text = query.get("query_text", "")
        levels = query.get("target_levels", []) if isinstance(query.get("target_levels", []), list) else []
        time_filter = query.get("time_filter")
        if not levels:
            return out
        level = str(levels[0]).upper()

        # RAW_VIDEO is handled by the dedicated path in controller.
        if level == "RAW_VIDEO":
            return out

        k = min(self.top_k_per_query, self.top_k_per_level)
        items = self.retriever.retrieve(query_text=query_text, target_level=level, top_k=k, time_filter=time_filter)
        for item in items:
            item_dict = item.__dict__.copy()
            item_dict["source_query_id"] = query_id
            out.append(item_dict)
        return out

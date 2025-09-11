from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from core.database import db
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


# ──────────────────────────────────────────────────────────────
# helpers – keep them private to this file
# ──────────────────────────────────────────────────────────────
_RANGE_CACHE: Dict[str, float] = {}


def _range_midpoint(label: str) -> float:
    """
    "10-14" → 12.0;  caches result for speed.
    Assumes well-formed 'low-high' strings (inclusive integers).
    """
    if label not in _RANGE_CACHE:
        lo, hi = map(float, label.split("-"))
        _RANGE_CACHE[label] = (lo + hi) / 2.0
    return _RANGE_CACHE[label]


# ──────────────────────────────────────────────────────────────
# main input model
# ──────────────────────────────────────────────────────────────
class RecurrenceCreate(BaseModel):
    # raw categorical / ordinal inputs
    age: int = Field(..., example=50)  # raw integer age
    menopause: Literal["lt40", "ge40", "premeno"]
    tumor_size: Literal[
        "0-4", "5-9", "10-14", "15-19", "20-24",
        "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59"
    ]
    inv_nodes: Literal[
        "0-2", "3-5", "6-8", "9-11", "12-14",
        "15-17", "24-26", "30-32", "36-39"
    ]
    node_caps: Literal["yes", "no"] = Field(..., alias="node-caps")
    breast: Literal["left", "right"]
    breast_quad: Literal[
        "left_up", "left_low", "right_up", "right_low", "central"
    ] = Field(..., alias="breast-quad")
    irradiat: Literal["yes", "no"]

    # not used by the final model, but still accepted/validated
    deg_malig: int = Field(..., ge=1, le=3, alias="deg-malig")

    # ────────────────────────────────
    # derived helpers
    # ────────────────────────────────
    @staticmethod
    def _bin_age(age: int) -> str:
        """Convert raw age to the dataset’s categorical age-range label."""
        bins = list(range(10, 100, 10))  # 10,20,…,90
        for low in bins:
            if age < low + 10:
                return f"{low}-{low+9}"
        return "90-100"

    # ensure aliases work both directions
    class Config:
        populate_by_name = True

    # ────────────────────────────────
    # Pydantic v2: post-init validator
    # ────────────────────────────────
    @model_validator(mode="after")
    def validate_ranges(self):
        # quick sanity: tumour/inv-node strings must contain “-”
        if "-" not in self.tumor_size or "-" not in self.inv_nodes:
            raise ValueError("Malformed range string in tumor_size or inv_nodes")
        return self

    # ────────────────────────────────
    # Public utility: convert → DataFrame
    # ────────────────────────────────
    def to_dataframe(self) -> pd.DataFrame:
        """
        Build the EXACT 10-column dataframe the trained pipeline expects:
            ['menopause', 'node_caps', 'irradiat',
             'breast', 'breast_quad',
             'age_mid', 'tumor_size_mid', 'inv_nodes_mid',
             'age_x_tumor', 'tumor_per_node']
        """
        age_mid = _range_midpoint(self._bin_age(self.age))
        tumor_size_mid = _range_midpoint(self.tumor_size)
        inv_nodes_mid = _range_midpoint(self.inv_nodes)

        df = pd.DataFrame(
            [{
                # nominal   (as in build_model nominals)
                "menopause": self.menopause,
                "node-caps": self.node_caps,
                "irradiat":  self.irradiat,

                # freq-encode categoricals (original strings)
                "breast": self.breast,
                "breast-quad": self.breast_quad,

                # numeric engineered
                "age_mid": age_mid,
                "tumor_size_mid": tumor_size_mid,
                "inv_nodes_mid": inv_nodes_mid,
                "age_x_tumor": age_mid * tumor_size_mid,
                "tumor_per_node": tumor_size_mid / (inv_nodes_mid + 1.0),
            }]
        )

        return df


# ──────────────────────────────────────────────────────────────
# Mongo persistence model (unchanged)
# ──────────────────────────────────────────────────────────────
class RecurrenceRecord(BaseModel):
    id: Optional[str]
    username: str
    input_data: Dict[str, Any]
    prediction: str
    probability: float
    timestamp: datetime
    company_id: Optional[str] = None

    @staticmethod
    async def create_recurrence(
        recurrence: "RecurrenceCreate",
        username: str,
        prediction: str,
        probability: float,
        company_id: Optional[str] = None,
    ) -> str:
        rec_doc: Dict[str, Any] = {
            "username": username,
            "input_data": recurrence.model_dump(by_alias=True),
            "prediction": prediction,
            "probability": probability,
            "timestamp": datetime.now(timezone.utc),
            "model_version": "v1.0",
        }
        if company_id:
            rec_doc["company_id"] = company_id

        result = await db["recurrence_predictions"].insert_one(rec_doc)
        return str(result.inserted_id)

    @staticmethod
    async def get_recurrence(
        username: str,
        company_id: Optional[str] = None,
        limit: int = 100,
    ) -> List["RecurrenceRecord"]:
        q = {"username": username}
        if company_id:
            q["company_id"] = company_id

        cur = (
            db["recurrence_predictions"]
            .find(q).sort("timestamp", -1).limit(limit)
        )

        out: List["RecurrenceRecord"] = []
        async for doc in cur:
            out.append(
                RecurrenceRecord(
                    id=str(doc["_id"]),
                    username=doc["username"],
                    input_data=doc["input_data"],
                    prediction=doc["prediction"],
                    probability=doc["probability"],
                    timestamp=doc["timestamp"],
                    company_id=doc.get("company_id"),
                )
            )
        return out

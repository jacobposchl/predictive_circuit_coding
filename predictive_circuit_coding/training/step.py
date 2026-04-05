from __future__ import annotations

from predictive_circuit_coding.objectives import CombinedObjective
from predictive_circuit_coding.training.contracts import PopulationWindowBatch, TrainingStepOutput


def run_training_step(model, objective: CombinedObjective, batch: PopulationWindowBatch) -> TrainingStepOutput:
    forward_output = model(batch)
    objective_output = objective(batch, forward_output)
    return TrainingStepOutput(
        loss=objective_output.loss,
        losses=objective_output.losses,
        metrics=objective_output.metrics,
        batch_size=batch.batch_size,
        token_count=batch.token_count,
    )

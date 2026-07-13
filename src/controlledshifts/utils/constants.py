from enum import Enum, StrEnum


MILLION = 1e6
EPSILON = 1e-10
LARGE_FLOAT = 1e10
POSITION_DIMS = [2, 3]
MIN_VALID_POINTS = 2
INVALID_AGENT_ID = -1

DEFAULT_COLLISION_THRESHOLDS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

# NOTE: borrowed from UniTraj: https://arxiv.org/pdf/2403.15098. They don't explain why these ranges were selected.
KALMAN_DIFFICULTY = {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}


class ModelStatus(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class VizType(Enum):
    """Scenario visualization types, each loading/computing only what it needs.

    REGULAR visualizes scenarios as-is. SCORED additionally computes scenario features and scores. TRAJPRED transforms
    the scenario into agent-centric format and overlays model trajectory predictions. MODEL_OUTPUT visualizes other
    cached model outputs (e.g. causal predictions). CAUSAL_GT renders the ground-truth causal agents loaded from the
    causal-label JSON files, requiring no cached model outputs.
    """

    REGULAR = "regular"
    SCORED = "scored"
    TRAJPRED = "trajpred"
    MODEL_OUTPUT = "model_output"
    CAUSAL_GT = "causal_gt"


class DataSplits(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class TrajectoryType(Enum):
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7


class AgentBehaviorType(Enum):
    NON_CAUSAL = 0
    CAUSAL = 1


class CausalOutputType(Enum):
    GROUND_TRUTH = 0
    PREDICTION = 1


class SupportedPanes(Enum):
    ALL_AGENTS = "all_agents"
    HIGHLIGHT_RELEVANT = "highlight_relevant"
    CAUSAL_AGENTS_GT = "causal_agents_gt"
    CAUSAL_AGENTS_PRED = "causal_agents_pred"
    TRAJECTORY_PREDICTION = "trajectory_prediction"

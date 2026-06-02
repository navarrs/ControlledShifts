from typing import Annotated

import numpy as np
from characterization.utils.common import (
    BooleanNDArray2D,
    Float32NDArray1D,
    Float32NDArray2D,
    Float32NDArray3D,
    validate_array,
)
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator


BooleanNDArray1D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 1))]
Int64NDArray3D = Annotated[NDArray[np.int64], BeforeValidator(validate_array(np.int64, 3))]


class AgentCentricScenario(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents an agent-centric scenario containing information about agents and the environment.

    This scenario representation is utilized by the model and for trajectory prediction visualization. This scenario
    representation is derived from UniTraj and minimally modified. For details on how the scenario is composed, refer to
    `controlledshifts/dataset/base_dataset.py`.

    Attributes:
        scenario_id (str): Unique identifier for the scenario.
        dataset_name (str): Name of the dataset from which the scenario is derived.
        obj_ids (Int64NDArray3D): IDs of the objects (agents) in the scene.
        obj_trajs (Int32NDArray3D): Trajectory histories of the objects in the scene.
        obj_trajs_mask (BooleanNDArray2D): Trajectory validity masks of the objects in the scene.
        obj_trajs_pos (Float32NDArray3D): Trajectory position information of objects in the scene.
        obj_trajs_future_state (Float32NDArray3D): Trajectory future state information.
        obj_trajs_future_mask (BooleanNDArray2D): Trajectory future validity mask information.
        track_index_to_predict (np.int64): Index of the object track to predict.
        center_gt_trajs (Float32NDArray3D): Ground truth trajectory center information.
        center_gt_trajs_mask (BooleanNDArray1D): Ground truth trajectory center mask information.
        center_gt_final_valid_idx (np.float32): Last valid ID for each object in the scene.
        center_objects_world (Float32NDArray1D): Center object world transformation information.
        center_objects_id (np.int64): ID of the center object.
        center_objects_id (np.int64): Agent type of the center object.
        center_gt_trajs_src (Float32NDArray2D): Ground Truth trajectory information of center object.
        pad (np.float32): Amount of padded elements in the scene. For scenes with num_agents < num_max_agents.
        map_polylines (Float32NDArray3D): Map polyline information.
        map_polylines_mask (Float32NDArray2D): Map polyline validity mask information.
        map_polylines_center (Float32NDArray2D): Map polyline center information.
    """

    # Metadata
    scenario_id: str
    dataset_name: str

    # History attributes of objects in the scene
    obj_ids: Int64NDArray3D
    obj_trajs: Float32NDArray3D
    obj_trajs_mask: BooleanNDArray2D
    obj_trajs_pos: Float32NDArray3D
    obj_trajs_last_pos: Float32NDArray2D
    obj_trajs_future_state: Float32NDArray3D
    obj_trajs_future_mask: Float32NDArray2D

    # Full centered trajectory information of object in the scene
    track_index_to_predict: np.int64
    center_gt_trajs_src: Float32NDArray2D
    center_gt_trajs: Float32NDArray2D
    center_gt_trajs_mask: Float32NDArray1D
    center_gt_final_valid_idx: np.float32
    center_objects_world: Float32NDArray1D
    center_objects_id: np.int64
    center_objects_type: np.int64
    pad: np.float32

    # Map attributes
    map_polylines: Float32NDArray3D
    map_polylines_mask: BooleanNDArray2D
    map_polylines_center: Float32NDArray2D

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

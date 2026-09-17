"""Keep cuboid-local axis signs continuous for a fixed real-robot plan."""
import copy
import math
from xarm_executor import UnsafePlan


def align_execution_yaws(objects, previous_objects):
    """Choose the nearest equivalent yaw (modulo pi) to each previous yaw.

    Rectangle detection determines an unoriented long axis. Mapping it to
    [0, pi) independently each frame is unsuitable for signed local actions.
    Preserve measured small rotations while removing arbitrary half-turn flips.
    Raw transforms and centres remain exactly as detected.
    """
    references={obj['name']:obj for obj in previous_objects}
    aligned=copy.deepcopy(objects)
    for obj in aligned:
        previous=references.get(obj['name'])
        if previous is None:continue
        raw=float(obj.get('detected_yaw_rad',obj['yaw_rad']))
        reference=float(previous['yaw_rad'])
        if not math.isfinite(raw) or not math.isfinite(reference):
            raise UnsafePlan(f'{obj["name"]}: non-finite yaw')
        half_turns=math.floor((reference-raw)/math.pi+.5)
        obj['detected_yaw_rad']=raw
        obj['yaw_rad']=raw+half_turns*math.pi
        obj['yaw_continuity']=dict(policy='nearest_previous_modulo_pi',
            reference_yaw_rad=reference,half_turns_applied=half_turns,
            execution_yaw_rad=obj['yaw_rad'])
    return aligned

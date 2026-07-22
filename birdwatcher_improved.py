"""Compatibility launcher for the modular Bird Watcher runtime.

New installations should run ``python main.py``. This file remains so older
service definitions that invoked ``birdwatcher_improved.py`` keep working.
"""
from birdwatcher.app import main, process_new_event, run
from birdwatcher.classification import (
    EncodedCrop,
    candidate_species_names,
    encode_accepted_crops,
    encode_bioclip_image,
    hybrid_predictions,
)
from birdwatcher.tracking import (
    ActiveEventTracker,
    TrackedDetection,
    box_iou,
    center_distance_ratio,
    choose_initial_detection,
    collect_tracked_crops,
    detect_birds,
    match_tracked_detection,
    predict_box,
)


if __name__ == "__main__":
    main()

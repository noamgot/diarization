import itertools
import json
import os
from itertools import groupby
from typing import Iterable

from pyannote.core import Annotation, Segment, Timeline
from pyannote.core.notebook import Notebook


class MyNotebook(Notebook):
    """This subclass alows to plot annotations seperated by labels (unlike the default Notebook class).

    Args:
        Notebook (_type_): _description_
    """
    def plot_annotation(self, annotation: Annotation, ax=None, time=True, legend=True, separate_by="optimal"):
        if not self.crop:
            self.crop = annotation.get_timeline(copy=False).extent()

        cropped = annotation.crop(self.crop, mode="intersection")
        labels = cropped.labels()
        labels_dict = {label: i for i, label in enumerate(labels)}
        segments = [s for s, _ in cropped.itertracks()]

        ax = self.setup(ax=ax, time=time)

        for (segment, track, label), y in zip(cropped.itertracks(yield_label=True), self.get_y(segments)):
            if separate_by == "labels":
                y = 1.0 - 1.0 / (len(labels) + 1) * (1 + labels_dict.get(label))
            self.draw_segment(ax, segment, y, label=label)

        if legend:
            H, L = ax.get_legend_handles_labels()

            # corner case when no segment is visible
            if not H:
                return

            # this gets exactly one legend handle and one legend label per label
            # (avoids repeated legends for repeated tracks with same label)
            HL = groupby(sorted(zip(H, L), key=lambda h_l: h_l[1]), key=lambda h_l: h_l[1])
            H, L = zip(*list((next(h_l)[0], l) for l, h_l in HL))
            ax.legend(
                H,
                L,
                bbox_to_anchor=(0, 1),
                loc=3,
                ncol=5,
                borderaxespad=0.0,
                frameon=False,
            )


def json_to_segments(json_data: str | os.PathLike | dict) -> list[Segment]:
    """
    Convert JSON data to a list of segments.

    Args:
        json_data (str | os.PathLike | dict): JSON data as a string, file path, or dictionary.

    Returns:
        list[Segment]: List of segments created from the JSON data.
    """
    if isinstance(json_data, (str, os.PathLike)):
        with open(json_data, "r") as file:
            json_data = json.load(file)

    segments = [Segment(item["start"], item["end"]) for item in json_data["segments"]]

    return segments


def json_to_timeline(json_data: str | os.PathLike | dict, uri: str | None = None) -> Timeline:
    """
    Convert JSON data to a Timeline object.

    Args:
        json_data (str | os.PathLike | dict): JSON data as a string, file path, or dictionary.

    Returns:
        Timeline: Timeline object created from the JSON data.
    """
    segments = json_to_segments(json_data)
    timeline = Timeline(segments=segments, uri=uri)
    return timeline


def add_segments_to_annotation(
    segments: Iterable[Segment], annotation: Annotation, label: str, track: str | int = "_"
) -> Annotation:
    """
    Add segments to an annotation with a specified label.

    Args:
        segments (iterable[Segment]): Iterable of segments to add.
        annotation (Annotation): The annotation to which segments will be added.
        label (str): The label for the segments.

    Returns:
        Annotation: Updated annotation with the added segments.
    """
    for segment in segments:
        annotation[segment, track] = label
    return annotation


def json_to_annotation(
    json_data: str | os.PathLike | dict, label: str = "default", track: str | int = "_", uri: str | None = None
) -> Annotation:
    """
    Convert JSON data to an Annotation object.

    Args:
        json_data (str | os.PathLike | dict): JSON data as a string, file path, or dictionary.
        uri (str | None): Optional URI for the annotation.

    Returns:
        Annotation: Annotation object created from the JSON data.
    """
    segments = json_to_segments(json_data)
    annotation = Annotation(uri=uri)
    add_segments_to_annotation(segments, annotation, label, track)
    return annotation


def timeline_to_single_label_annotation(timeline: Timeline, label: str | int) -> Annotation:
    annotation = timeline.to_annotation(generator=itertools.repeat(label))
    return annotation

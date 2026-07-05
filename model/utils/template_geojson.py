# -*- coding: utf-8 -*-
# Copyright © 2024 Fabian Hörst, Jens Kleesiek (original CellViT code).
# Copyright © 2025 Olena Kosharova, FIIT STU (modifications).
#
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.
#
# NOTICE: This file has been modified from the original CellViT source
# (https://github.com/TIO-IKIM/CellViT) as part of the
# "Tissue-Context CellViT Extension" bachelor's thesis work.

# GeoJson templates
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


def get_template_point() -> dict:
    """Return a template for a Point geojson object

    Returns:
        dict: Template
    """
    template_point = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_point


def get_template_segmentation() -> dict:
    """Return a template for a MultiPolygon geojson object

    Returns:
        dict: Template
    """
    template_multipolygon = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_multipolygon

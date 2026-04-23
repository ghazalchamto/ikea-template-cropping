# streamlit_drawable_canvas 0.9.3 uses streamlit.elements.image.image_to_url, which
# Streamlit removed in favor of image_utils.image_to_url (LayoutConfig as 2nd arg).
# Import this module before st_canvas. See andfanilo/streamlit-drawable-canvas#156.
# See: https://github.com/andfanilo/streamlit-drawable-canvas/issues/156

from __future__ import annotations

import streamlit.elements.image as st_image

if not hasattr(st_image, "image_to_url"):

    def _image_to_url(
        image,
        width,
        clamp: bool,
        channels,
        output_format,
        image_id: str,
    ) -> str:
        from streamlit.elements.lib import image_utils
        from streamlit.elements.lib.layout_utils import LayoutConfig

        return image_utils.image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = _image_to_url  # type: ignore[attr-defined]

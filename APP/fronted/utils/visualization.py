import io
import plotly.graph_objects as go
from PIL import Image

__all__=["make_plotly_figure"]

LANDMARK_IDX = {
    0: "right_zygomatic_width",
    1: "right_gonial_width",
    2: "mid_symphysis",
    3: "left_gonial_width",
    4: "left_zygomatic_width",
    5: "soft_tissue_nasion",
    6: "mid_dorsum",
    7: "nasal_tip",
    8: "subnasale",
    9: "right_outer_canthus",
    10: "right_inner_canthus",
    11: "left_inner_canthus",
    12: "left_outer_canthus",
    13: "right_outer_commissure",
    14: "philtral_tubercle",
    15: "left_outer_commissure",
    16: "anterior_lower_lip",
    17: "lib_junction",
    18: "glabella",
    19: "subsnale",
    20: "trichion",
    21: "right_alar_base",
    22: "left_alar_base",
    23: "labiomental_fold",
    24: "chin_point",
    25: "inferior_chin",
    26: "reflex_point",
}

POINTS_OPTIONS = ["모두 표시", "보이는 부분만 표시"]

def make_plotly_figure(
    image,
    landmarks,
    landmarks_color,
    landmark_size,
    landmark_option=None,
    landmark_with_measurements=False,
    fig_size=512
):
    image = Image.open(io.BytesIO(image.getvalue()))
    width, height = image.size[0], image.size[1]
    
    fig = go.Figure()
    fig.update_xaxes(visible=False, range=[0,width])
    fig.update_yaxes(visible=False, range=[0,height], scaleanchor="x")
    fig.add_layout_image(
        dict(
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            xref="x",
            yref="y",
            opacity=1.0,
            sizing="stretch",
            source=image,
            layer="below"
        )
    )
    
    if isinstance(points_option, list):
        points_option = [LANDMARK_IDX[p] for p in points_option]
        
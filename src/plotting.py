import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.transforms as mtransforms
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from .constants import ARROW_LENGTH, ARRAY_INDICATOR_RADIUS, ARRAY_INDICATOR_CONE_WIDTH_DEG, DEFAULT_SUB_WIDTH, DEFAULT_SUB_DEPTH

class MatplotlibCanvas(QWidget):
    def __init__(self, parent=None, width=8, height=7, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

def draw_subwoofers_and_elements(ax, sorgenti, current_sub_idx):
    for i, sub_ in enumerate(sorgenti):
        color = "green" if i == current_sub_idx else "black"
        if sub_.get("group_id") is not None:
            if sub_["is_group_master"]:
                color = "purple"
            elif i == current_sub_idx:
                color = "darkorange"
            else:
                color = "blue"

        center_x, center_y = sub_["x"], sub_["y"]
        angle_rad = sub_["angle"]

        display_angle_deg = 90 - np.degrees(angle_rad)

        width = sub_.get("width", DEFAULT_SUB_WIDTH)
        depth = sub_.get("depth", DEFAULT_SUB_DEPTH)

        rect = patches.Rectangle(
            (-depth / 2, -width / 2),
            depth,
            width,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.6,
            gid=f"rect_sub_{sub_['id']}",
            picker=True,
            zorder=2.5,
        )
        transform = (
            mtransforms.Affine2D().rotate_deg(display_angle_deg)
            + mtransforms.Affine2D().translate(center_x, center_y)
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)
        sub_["rect_artist"] = rect

        arrow_end_x = center_x + ARROW_LENGTH * np.sin(angle_rad)
        arrow_end_y = center_y + ARROW_LENGTH * np.cos(angle_rad)
        arrow = patches.FancyArrowPatch(
            (center_x, center_y),
            (arrow_end_x, arrow_end_y),
            mutation_scale=20,
            arrowstyle="->",
            color="dimgray",
            linewidth=1.5,
            zorder=2.6,
            gid=f"arrow_sub_{sub_['id']}",
            picker=True,
        )
        ax.add_patch(arrow)
        sub_["arrow_artist"] = arrow

        pol_char = "+" if sub_["polarity"] > 0 else "-"
        group_info = ""
        if sub_.get("group_id") is not None:
            group_info = f"G{sub_['group_id']}"
            if sub_["is_group_master"]:
                group_info += " (M)"
            group_info += ", "
        sub_text_info = (
            f"S{sub_.get('id', i + 1)}: {sub_.get('spl_rms', 85.0):.0f}dB\n"
            f"{group_info}{sub_.get('gain_db', 0.0):.1f}dB, {sub_['delay_ms']:.1f}ms, Pol {pol_char}"
        )

        text_offset = max(width, depth) / 2 + 0.15
        ax.text(
            center_x,
            center_y + text_offset,
            sub_text_info,
            color=color,
            fontsize=6,
            ha="center",
            va="bottom",
            zorder=6,
        )

def draw_array_direction_indicators(ax, lista_gruppi_array, sorgenti):
    for group_id, group_info in lista_gruppi_array.items():
        members = [s for s in sorgenti if s.get("group_id") == group_id]
        if not members:
            continue

        center_x = np.mean([s["x"] for s in members])
        center_y = np.mean([s["y"] for s in members])

        steering_deg_nord = group_info.get("steering_deg", 0)
        steering_deg_est = 90 - steering_deg_nord

        wedge = patches.Wedge(
            center=(center_x, center_y),
            r=ARRAY_INDICATOR_RADIUS,
            theta1=steering_deg_est - ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
            theta2=steering_deg_est + ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
            facecolor="cyan",
            edgecolor="blue",
            alpha=0.25,
            zorder=1.5,
        )
        ax.add_patch(wedge)

def draw_room_and_vertices(ax, punti_stanza):
    if punti_stanza and len(punti_stanza) >= 3:
        points_pos = [p["pos"] for p in punti_stanza]
        patch = patches.Polygon(
            points_pos,
            closed=True,
            fill=None,
            edgecolor="blue",
            linewidth=2,
            zorder=1,
        )
        ax.add_patch(patch)
        for i, vtx_data in enumerate(punti_stanza):
            x, y = vtx_data["pos"]
            vtx_data["plot"] = ax.plot(
                x,
                y,
                marker="o",
                ms=7,
                color="red",
                picker=7,
                gid=f"stanza_vtx_{i}",
                zorder=5.2,
            )[0]

def draw_target_and_avoidance_areas(ax, lista_target_areas, lista_avoidance_areas, lista_sub_placement_areas, current_target_area_idx, current_avoidance_area_idx, current_sub_placement_area_idx):
    area_lists_and_prefixes = [
        (lista_target_areas, "target"),
        (lista_avoidance_areas, "avoid"),
        (lista_sub_placement_areas, "placement"),
    ]
    
    for area_list, type_prefix in area_lists_and_prefixes:
        for area_data in area_list:
            area_data["plots"] = []
            if (
                area_data.get("active", False)
                and len(area_data.get("punti", [])) >= 3
            ):
                is_selected = False
                if type_prefix == "target":
                    is_selected = (
                        current_target_area_idx != -1
                        and area_data["id"] == lista_target_areas[current_target_area_idx]["id"]
                    )
                    color, m_color, ls = ("green", "lime", "--")
                elif type_prefix == "avoid":
                    is_selected = (
                        current_avoidance_area_idx != -1
                        and area_data["id"] == lista_avoidance_areas[current_avoidance_area_idx]["id"]
                    )
                    color, m_color, ls = ("red", "orangered", ":")
                else:  # placement
                    is_selected = (
                        current_sub_placement_area_idx != -1
                        and area_data["id"] == lista_sub_placement_areas[current_sub_placement_area_idx]["id"]
                    )
                    color, m_color, ls = ("purple", "magenta", "-.")

                patch = patches.Polygon(
                    area_data["punti"],
                    closed=True,
                    fill=True,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3 if is_selected else 0.15,
                    ls=ls,
                    zorder=0.5,
                )
                ax.add_patch(patch)
                
                marker = "P" if type_prefix == "target" else ("X" if type_prefix == "avoid" else "s")
                for v_idx, v in enumerate(area_data["punti"]):
                    plot_artist = ax.plot(
                        v[0],
                        v[1],
                        marker=marker,
                        ms=9,
                        color=m_color,
                        picker=7,
                        zorder=5.1,
                    )[0]
                    area_data["plots"].append(plot_artist)

def draw_grid(ax, grid_show_enabled, grid_snap_spacing):
    if grid_show_enabled and grid_snap_spacing > 0:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_ticks = np.arange(
            round(xlim[0] / grid_snap_spacing) * grid_snap_spacing,
            xlim[1],
            grid_snap_spacing,
        )
        y_ticks = np.arange(
            round(ylim[0] / grid_snap_spacing) * grid_snap_spacing,
            ylim[1],
            grid_snap_spacing,
        )
        for x in x_ticks:
            ax.axvline(
                x, color="gray", linestyle=":", linewidth=0.5, alpha=0.6, zorder=-1
            )
        for y in y_ticks:
            ax.axhline(
                y, color="gray", linestyle=":", linewidth=0.5, alpha=0.6, zorder=-1
            )

def draw_background_image(ax, bg_image_props, is_in_calibration_mode):
    if bg_image_props.get("artist") is not None:
        try:
            if bg_image_props["artist"] in ax.images:
                bg_image_props["artist"].remove()
        except (AttributeError, ValueError, NotImplementedError):
            ax.images.clear() 

        bg_image_props["artist"] = None

    if (
        bg_image_props.get("data") is not None
        and not is_in_calibration_mode
    ):
        image_data = bg_image_props.get("cached_transformed") or bg_image_props["data"]
        
        bg_image_props["artist"] = ax.imshow(
            image_data,
            alpha=bg_image_props.get("alpha", 0.5),
            zorder=-10,
            origin="upper",
        )

        h, w = bg_image_props["data"].shape[:2]
        scale = bg_image_props.get("scale", 1.0)
        rotation = bg_image_props.get("rotation_deg", 0.0)
        center_x = bg_image_props.get("center_x", 0.0)
        center_y = bg_image_props.get("center_y", 0.0)

        transform = None

        if (
            "anchor_pixel" in bg_image_props
            and bg_image_props["anchor_pixel"] is not None
        ):
            anchor_px = bg_image_props["anchor_pixel"]

            transform = (
                mtransforms.Affine2D()
                .translate(-anchor_px[0], -anchor_px[1])
                .rotate_deg(rotation)
                .scale(scale, -scale) 
                .translate(center_x, center_y)
            )

        else:
            transform = (
                mtransforms.Affine2D()
                .translate(-w / 2, -h / 2)
                .scale(scale, -scale) 
                .rotate_deg(rotation)
                .translate(center_x, center_y)
            )

        if transform:
            bg_image_props["artist"].set_transform(
                transform + ax.transData
            )

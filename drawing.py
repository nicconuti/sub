import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms

from constants import (
    DEFAULT_SUB_WIDTH,
    DEFAULT_SUB_DEPTH,
    ARROW_LENGTH,
    ARRAY_INDICATOR_RADIUS,
    ARRAY_INDICATOR_CONE_WIDTH_DEG,
)


class DrawingMixin:
    """Mixin with drawing routines for SubwooferSimApp."""

    def disegna_subwoofer_e_elementi(self):
        for i, sub_ in enumerate(self.sorgenti):
            color = "green" if i == self.current_sub_idx else "black"
            if sub_.get('group_id') is not None:
                if sub_['is_group_master']:
                    color = "purple"
                elif i == self.current_sub_idx:
                    color = "darkorange"
                else:
                    color = "blue"

            center_x, center_y = sub_['x'], sub_['y']
            angle_rad = sub_['angle']
            display_angle_deg = 90 - np.degrees(angle_rad)
            width = sub_.get('width', DEFAULT_SUB_WIDTH)
            depth = sub_.get('depth', DEFAULT_SUB_DEPTH)

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
                + self.ax.transData
            )
            rect.set_transform(transform)
            self.ax.add_patch(rect)
            sub_['rect_artist'] = rect

            arrow_end_x = center_x + ARROW_LENGTH * np.sin(angle_rad)
            arrow_end_y = center_y + ARROW_LENGTH * np.cos(angle_rad)
            arrow = patches.FancyArrowPatch(
                (center_x, center_y),
                (arrow_end_x, arrow_end_y),
                mutation_scale=20,
                arrowstyle='->',
                color='dimgray',
                linewidth=1.5,
                zorder=2.6,
                gid=f"arrow_sub_{sub_['id']}",
                picker=True,
            )
            self.ax.add_patch(arrow)
            sub_['arrow_artist'] = arrow

            pol_char = '+' if sub_['polarity'] > 0 else '-'
            group_info = ""
            if sub_.get('group_id') is not None:
                group_info = f"G{sub_['group_id']}"
                if sub_['is_group_master']:
                    group_info += " (M)"
                group_info += ", "
            sub_text_info = (
                f"S{sub_.get('id', i + 1)}: {sub_.get('spl_rms', DEFAULT_SUB_SPL_RMS):.0f}dB\n"
                f"{group_info}{sub_.get('gain_db', 0.0):.1f}dB, {sub_['delay_ms']:.1f}ms, Pol {pol_char}"
            )
            text_offset = max(width, depth) / 2 + 0.15
            self.ax.text(
                center_x,
                center_y + text_offset,
                sub_text_info,
                color=color,
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=6,
            )

    def disegna_array_direction_indicators(self):
        for group_id, group_info in self.lista_gruppi_array.items():
            members = [s for s in self.sorgenti if s.get('group_id') == group_id]
            if not members:
                continue
            center_x = np.mean([s['x'] for s in members])
            center_y = np.mean([s['y'] for s in members])
            steering_deg_nord = group_info.get('steering_deg', 0)
            steering_deg_est = 90 - steering_deg_nord
            wedge = patches.Wedge(
                center=(center_x, center_y),
                r=ARRAY_INDICATOR_RADIUS,
                theta1=steering_deg_est - ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
                theta2=steering_deg_est + ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
                facecolor='cyan',
                edgecolor='blue',
                alpha=0.25,
                zorder=1.5,
            )
            self.ax.add_patch(wedge)

    def disegna_stanza_e_vertici(self):
        if self.punti_stanza and len(self.punti_stanza) >= 3:
            points_pos = [p['pos'] for p in self.punti_stanza]
            patch = patches.Polygon(
                points_pos,
                closed=True,
                fill=None,
                edgecolor="blue",
                linewidth=2,
                zorder=1,
            )
            self.ax.add_patch(patch)
            for i, vtx_data in enumerate(self.punti_stanza):
                x, y = vtx_data['pos']
                vtx_data['plot'] = self.ax.plot(
                    x,
                    y,
                    marker="o",
                    ms=7,
                    color="red",
                    picker=7,
                    gid=f'stanza_vtx_{i}',
                    zorder=5.2,
                )[0]

    def disegna_aree_target_e_avoidance(self):
        for area_list, type_prefix in [
            (self.lista_target_areas, 'target'),
            (self.lista_avoidance_areas, 'avoid'),
        ]:
            for area_data in area_list:
                area_data['plots'] = []
                if area_data.get('active', False) and len(area_data.get('punti', [])) >= 3:
                    is_selected_target = (
                        type_prefix == 'target'
                        and self.current_target_area_idx != -1
                        and area_data['id'] == self.lista_target_areas[self.current_target_area_idx]['id']
                    )
                    is_selected_avoid = (
                        type_prefix == 'avoid'
                        and self.current_avoidance_area_idx != -1
                        and area_data['id'] == self.lista_avoidance_areas[self.current_avoidance_area_idx]['id']
                    )
                    is_selected = is_selected_target or is_selected_avoid
                    color, m_color, ls = (
                        ('green', 'lime', '--') if type_prefix == 'target' else ('red', 'orangered', ':')
                    )
                    patch = patches.Polygon(
                        area_data['punti'],
                        closed=True,
                        fill=True,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3 if is_selected else 0.15,
                        ls=ls,
                        zorder=0.5,
                    )
                    self.ax.add_patch(patch)
                    for v_idx, v in enumerate(area_data['punti']):
                        plot_artist = self.ax.plot(
                            v[0],
                            v[1],
                            marker='P' if type_prefix == 'target' else 'X',
                            ms=9,
                            color=m_color,
                            picker=7,
                            zorder=5.1,
                        )[0]
                        area_data['plots'].append(plot_artist)

    def disegna_griglia(self):
        if self.grid_show_enabled and self.grid_snap_spacing > 0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_ticks = np.arange(
                round(xlim[0] / self.grid_snap_spacing) * self.grid_snap_spacing,
                xlim[1],
                self.grid_snap_spacing,
            )
            y_ticks = np.arange(
                round(ylim[0] / self.grid_snap_spacing) * self.grid_snap_spacing,
                ylim[1],
                self.grid_snap_spacing,
            )
            for x in x_ticks:
                self.ax.axvline(x, color='gray', linestyle=':', linewidth=0.5, alpha=0.6, zorder=-1)
            for y in y_ticks:
                self.ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.6, zorder=-1)

    def disegna_elementi_statici_senza_spl(self):
        self.disegna_stanza_e_vertici()
        self.disegna_aree_target_e_avoidance()
        self.disegna_subwoofer_e_elementi()
        self.disegna_array_direction_indicators()
        self.disegna_griglia()


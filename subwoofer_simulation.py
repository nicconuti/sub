# --- START DIAGNOSTIC CODE ---
import sys
import os
try:
    import matplotlib
    import PyQt6
    print("--- DIAGNOSTICS ---")
    print("Python Executable:", sys.executable)
    print("Matplotlib Version:", matplotlib.__version__)
    print("Matplotlib Path:", os.path.dirname(matplotlib.__file__))
    print("PyQt6 Path:", os.path.dirname(PyQt6.__file__))
    print("--- END DIAGNOSTICS ---\n\n")
except Exception as e:
    print(f"DIAGNOSTIC ERROR: {e}")
# --- END DIAGNOSTIC CODE ---

import sys
import numpy as np
from matplotlib.path import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QSlider, QCheckBox, QRadioButton,
                             QGroupBox, QFormLayout, QGridLayout, QSizePolicy, QStatusBar,
                             QMessageBox, QButtonGroup, QScrollArea, QComboBox, QInputDialog,
                             QListWidget, QListWidgetItem, QSplitter, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

import warnings
from constants import *
from optimization_worker import OptimizationWorker
from spl_calculations import calculate_spl_vectorized
from ui_setup import UISetupMixin
from drawing import DrawingMixin
from array_setup import ArraySetupMixin
from project_io import ProjectIOMixin



class SubwooferSimApp(QMainWindow, UISetupMixin, DrawingMixin, ArraySetupMixin, ProjectIOMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulatore Posizionamento Subwoofer Avanzato")
        self.setGeometry(50, 50, 1600, 1000)

        # Stato dell'applicazione
        self.punti_stanza = []
        for p in DEFAULT_ROOM_VERTICES:
            self.punti_stanza.append({'pos': p, 'plot': None})

        self.sorgenti = []
        self.next_sub_id = 1
        self.next_group_id = 1
        self.max_group_id = 0
        self.current_sub_idx = -1
        self.lista_gruppi_array = {} 

        self.lista_target_areas = []
        self.current_target_area_idx = -1
        self.next_target_area_id = 1
        
        self.lista_avoidance_areas = []
        self.current_avoidance_area_idx = -1
        self.next_avoidance_area_id = 1
        
        self.selected_stanza_vtx_idx = -1
        self.drag_object = None
        self.original_mouse_pos = None
        self.original_object_pos = None
        self.original_object_angle = None
        self.original_group_states = []
        
        self.current_spl_map = None
        self._cax_for_colorbar_spl = None
                                
        self.optimization_thread = None
        self.optimization_worker = None
        
        self.global_sub_width = DEFAULT_SUB_WIDTH
        self.global_sub_depth = DEFAULT_SUB_DEPTH
        self.global_sub_spl_rms = DEFAULT_SUB_SPL_RMS
        self.use_global_for_new_manual_subs = False
        
        self.grid_snap_spacing = 0.25
        self.grid_snap_enabled = False
        self.grid_show_enabled = False
        
        self.max_spl_avoidance_ui_val = DEFAULT_MAX_SPL_AVOIDANCE
        self.target_min_spl_desired_ui_val = DEFAULT_TARGET_MIN_SPL_DESIRED
        self.balance_slider_ui_val = DEFAULT_BALANCE_SLIDER_VALUE

        # --- INIZIO NUOVO CODICE ---
        # Memorizza gli ultimi parametri usati per l'ottimizzazione
        self.last_optim_criterion = None
        self.last_optim_freq_s = None
        self.last_optim_freq_min = None
        self.last_optim_freq_max = None
        # --- FINE NUOVO CODICE ---


        self._setup_ui()
        
        if self.sorgenti: self.current_sub_idx = 0
        
        self.auto_fit_view_to_room()
        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()
        self.update_optim_freq_fields_visibility()
        # _update_array_ui() is now called within _setup_group_array_ui


    def auto_fit_view_to_room(self):
        if not hasattr(self, 'ax'): return
        if not self.punti_stanza:
            self.ax.set_xlim(-7, 7)
            self.ax.set_ylim(-5, 5)
        else:
            all_x = [p['pos'][0] for p in self.punti_stanza]
            all_y = [p['pos'][1] for p in self.punti_stanza]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            px = (max_x - min_x) * 0.15 if (max_x - min_x) > 0 else 1.5
            py = (max_y - min_y) * 0.15 if (max_y - min_y) > 0 else 1.5
            self.ax.set_xlim(min_x - px, max_x + px)
            self.ax.set_ylim(min_y - py, max_y + py)
        self.ax.set_aspect('equal', adjustable='box')
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.canvas.draw_idle()

    def get_active_areas_points(self, area_list):
        return [list(area['punti']) for area in area_list if area.get('active', False) and len(area.get('punti', [])) >= 3]

    def snap_to_grid(self, value):
        return round(value / self.grid_snap_spacing) * self.grid_snap_spacing if self.grid_snap_enabled and self.grid_snap_spacing > 0 else value
        
    def add_stanza_vtx(self, event=None):
        xlims, ylims = self.ax.get_xlim(), self.ax.get_ylim()
        new_vtx_coord = [np.mean(xlims), np.mean(ylims)]
        if self.punti_stanza and len(self.punti_stanza) > 1: p_last = self.punti_stanza[-1]['pos']; new_vtx_coord = [p_last[0] + 1, p_last[1]]
        snapped_pos = [self.snap_to_grid(new_vtx_coord[0]), self.snap_to_grid(new_vtx_coord[1])]
        self.punti_stanza.append({'pos': snapped_pos, 'plot': None})
        self.full_redraw(preserve_view=False)

    def remove_stanza_vtx(self, event=None):
        if len(self.punti_stanza) > 0:
            self.punti_stanza.pop()
            self.full_redraw(preserve_view=False)

    def update_stanza_vtx_editor(self):
        is_valid_idx = 0 <= self.selected_stanza_vtx_idx < len(self.punti_stanza)
        for w in [self.tb_stanza_vtx_x, self.tb_stanza_vtx_y, self.btn_update_stanza_vtx]: w.setEnabled(is_valid_idx)
        if is_valid_idx:
            vtx = self.punti_stanza[self.selected_stanza_vtx_idx]['pos']
            self.selected_vtx_label.setText(f"Vertice Selezionato {self.selected_stanza_vtx_idx + 1}:")
            self.tb_stanza_vtx_x.setText(f"{vtx[0]:.2f}"); self.tb_stanza_vtx_y.setText(f"{vtx[1]:.2f}")
        else: self.selected_vtx_label.setText("Nessun Vertice Selezionato"); self.tb_stanza_vtx_x.setText(""); self.tb_stanza_vtx_y.setText("")
        self.error_text_stanza_vtx_edit.setText("")

    def on_update_selected_stanza_vertex(self):
        if not (0 <= self.selected_stanza_vtx_idx < len(self.punti_stanza)): return
        try:
            x, y = float(self.tb_stanza_vtx_x.text()), float(self.tb_stanza_vtx_y.text())
            self.punti_stanza[self.selected_stanza_vtx_idx]['pos'] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.full_redraw(preserve_view=False)
            self.update_stanza_vtx_editor()
        except ValueError: self.error_text_stanza_vtx_edit.setText("Coordinate non valide.")

    def on_toggle_use_global_for_new(self, checked): self.use_global_for_new_manual_subs = checked
    
    def apply_global_settings_to_all_subs(self):
        try:
            new_spl = float(self.tb_global_sub_spl.text())
            if not (PARAM_RANGES['spl_rms'][0] <= new_spl <= PARAM_RANGES['spl_rms'][1]): raise ValueError(f"SPL fuori range")
            self.global_sub_width = float(self.tb_global_sub_width.text()); self.global_sub_depth = float(self.tb_global_sub_depth.text()); self.global_sub_spl_rms = new_spl
            for sub in self.sorgenti: sub.update({'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms, 'pressure_val_at_1m_relative_to_pref': 10**(new_spl/20.0)})
            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()
        except ValueError as e: self.error_text_global_settings.setText(f"Errore: {e}")

    def select_next_sub(self, event=None):
        if self.sorgenti: 
            self.current_sub_idx = (self.current_sub_idx + 1) % len(self.sorgenti)
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)
    def select_prev_sub(self, event=None):
        if self.sorgenti: 
            self.current_sub_idx = (self.current_sub_idx - 1 + len(self.sorgenti)) % len(self.sorgenti)
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)

    def add_subwoofer(self, event=None, specific_config=None, redraw=True):
        new_sub_data = specific_config or {}
        if not specific_config:
            new_sub_data = {'x': 0.0, 'y': 0.0}
            if self.use_global_for_new_manual_subs: new_sub_data.update({'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms})
        
        defaults = {'id': self.next_sub_id, 'angle': 0.0, 'delay_ms': 0.0, 'polarity': 1, 'gain_db': 0.0, 
                    'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms, 
                    'param_locks': {'angle': False, 'delay': False, 'gain': False, 'polarity': False}, 
                    'group_id': None, 'is_group_master': False}
        for k, v in defaults.items(): new_sub_data.setdefault(k, v)
        
        new_sub_data['gain_lin'] = 10**(new_sub_data['gain_db']/20.0)
        new_sub_data['pressure_val_at_1m_relative_to_pref'] = 10**(new_sub_data['spl_rms']/20.0)
        
        self.sorgenti.append(new_sub_data)
        self.next_sub_id += 1
        self.current_sub_idx = len(self.sorgenti) - 1
        
        if redraw:
            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()

    def remove_subwoofer(self, event=None):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub_to_remove = self.sorgenti.pop(self.current_sub_idx)
        gid_to_check = sub_to_remove.get('group_id')
        if gid_to_check is not None:
            remaining_in_group = [s for s in self.sorgenti if s.get('group_id') == gid_to_check]
            if not remaining_in_group and gid_to_check in self.lista_gruppi_array:
                del self.lista_gruppi_array[gid_to_check]
            elif sub_to_remove.get('is_group_master') and remaining_in_group:
                remaining_in_group[0]['is_group_master'] = True
        
        if not self.sorgenti:
            self.current_sub_idx = -1
        else:
            self.current_sub_idx = min(self.current_sub_idx, len(self.sorgenti) - 1)
            
        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()

    def aggiorna_ui_sub_fields(self):
        enable = bool(self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti))
        for w in [self.tb_sub_x, self.tb_sub_y, self.tb_sub_angle, self.check_sub_angle_lock, self.tb_sub_delay, self.check_sub_delay_lock, self.tb_sub_gain_db, self.check_sub_gain_lock, self.tb_sub_polarity, self.check_sub_polarity_lock, self.tb_sub_width, self.tb_sub_depth, self.tb_sub_spl_rms, self.btn_submit_sub_params, self.btn_rem_sub, self.apply_array_config_button]: w.setEnabled(enable)
        
        # Resetta le etichette al loro stato di default
        self.sub_gain_label.setText("Trim Gain (dB):")
        self.sub_delay_label.setText("Delay (ms):")
        self.sub_polarity_label.setText("Polarità (+1/-1):")
        self.tb_sub_gain_db.setPlaceholderText("")
        self.tb_sub_delay.setPlaceholderText("")
        self.tb_sub_polarity.setPlaceholderText("")

        if enable:
            sub = self.sorgenti[self.current_sub_idx]
            self.sub_selector_text_widget.setText(f"Sub ID:{sub.get('id', '')} ({self.current_sub_idx + 1}/{len(self.sorgenti)})")
            
            if sub.get('group_id') is not None:
                centroid = self._get_group_centroid(sub['group_id'])
                if centroid:
                    self.tb_sub_x.setText(f"{centroid[0]:.2f}")
                    self.tb_sub_y.setText(f"{centroid[1]:.2f}")
                self.sub_pos_label.setText("X/Y Gruppo (m):")
                # NUOVA FUNZIONALITÀ: Modifica UI per controllo di gruppo
                self.sub_gain_label.setText("Gain Relativo (+/- dB):")
                self.sub_delay_label.setText("Delay Relativo (+/- ms):")
                self.sub_polarity_label.setText("Polarità Assoluta:")
                self.tb_sub_gain_db.setText("")
                self.tb_sub_delay.setText("")
                self.tb_sub_polarity.setText("")
                self.tb_sub_gain_db.setPlaceholderText("es. 1.5 o -2")
                self.tb_sub_delay.setPlaceholderText("es. 5 o -10")
                self.tb_sub_polarity.setPlaceholderText("1 o -1")

            else:
                self.tb_sub_x.setText(f"{sub['x']:.2f}")
                self.tb_sub_y.setText(f"{sub['y']:.2f}")
                self.sub_pos_label.setText("X/Y Sub (m):")
                self.tb_sub_gain_db.setText(f"{sub['gain_db']:.1f}")
                self.tb_sub_delay.setText(f"{sub['delay_ms']:.2f}")
                self.tb_sub_polarity.setText(str(int(sub['polarity'])))

            self.tb_sub_angle.setText(f"{np.degrees(sub['angle']):.1f}")
            self.tb_sub_width.setText(f"{sub['width']:.2f}"); self.tb_sub_depth.setText(f"{sub['depth']:.2f}"); self.tb_sub_spl_rms.setText(f"{sub['spl_rms']:.1f}")
            for p, cb in [('angle', self.check_sub_angle_lock), ('delay', self.check_sub_delay_lock), ('gain', self.check_sub_gain_lock), ('polarity', self.check_sub_polarity_lock)]:
                try: cb.toggled.disconnect()
                except TypeError: pass
                cb.setChecked(sub['param_locks'].get(p, False)); cb.toggled.connect(self.on_toggle_param_lock)
        else:
            self.sub_selector_text_widget.setText("Nessun Sub")
            for w in [self.tb_sub_x, self.tb_sub_y, self.tb_sub_angle, self.tb_sub_delay, self.tb_sub_gain_db, self.tb_sub_polarity, self.tb_sub_width, self.tb_sub_depth, self.tb_sub_spl_rms]:
                w.clear()
            self.sub_pos_label.setText("X/Y Sub (m):")

        self.error_text_sub.setText(""); self._update_group_ui_status()
        if enable and sub.get('group_id') is not None:
           self.group_details_label.setVisible(True)
           self.group_members_list.setVisible(True)
           self.group_members_list.clear()

           group_id = sub.get('group_id')
           members = [s for s in self.sorgenti if s.get('group_id') == group_id]
           
           for member in members:
               pol_char = '+' if member['polarity'] > 0 else '-'
               master_char = " (M)" if member.get('is_group_master') else ""
               info_string = (f"S{member['id']}{master_char}: "
                              f"{member['gain_db']:.1f}dB, "
                              f"{member['delay_ms']:.2f}ms, "
                              f"Pol {pol_char}")
               self.group_members_list.addItem(QListWidgetItem(info_string))
        else:
           # Se il sub non è in un gruppo, nascondi i dettagli
           self.group_details_label.setVisible(False)
           self.group_members_list.setVisible(False)
           self.group_members_list.clear()
        
    def _get_group_centroid(self, group_id):
        if group_id is None: return None
        members = [s for s in self.sorgenti if s.get('group_id') == group_id]
        if not members: return None
        center_x = np.mean([s['x'] for s in members])
        center_y = np.mean([s['y'] for s in members])
        return (center_x, center_y)

    def on_toggle_param_lock(self, checked):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]; sender = self.sender(); param_name = sender.objectName()
        if sub.get('group_id') is not None and sub['param_locks'].get(param_name) and not checked: QMessageBox.warning(self, "Attenzione", f"Il parametro è controllato dall'array. Sciogli il gruppo per sbloccarlo."); sender.setChecked(True); return
        sub['param_locks'][param_name] = checked

    def on_submit_sub_param_qt(self):
        if not self.sorgenti or self.current_sub_idx < 0: return
        try:
            sub = self.sorgenti[self.current_sub_idx]
            
            if sub.get('group_id') is not None:
                group_id = sub['group_id']
                current_centroid = self._get_group_centroid(group_id)
                new_centroid_x = float(self.tb_sub_x.text())
                new_centroid_y = float(self.tb_sub_y.text())
                if current_centroid:
                    dx = new_centroid_x - current_centroid[0]
                    dy = new_centroid_y - current_centroid[1]
                    for member in self.sorgenti:
                        if member.get('group_id') == group_id:
                            member['x'] += dx
                            member['y'] += dy
                
                if self.tb_sub_gain_db.text().strip():
                    gain_delta = float(self.tb_sub_gain_db.text())
                    for member in self.sorgenti:
                        if member.get('group_id') == group_id: member['gain_db'] += gain_delta
                
                if self.tb_sub_delay.text().strip():
                    delay_delta = float(self.tb_sub_delay.text())
                    for member in self.sorgenti:
                        if member.get('group_id') == group_id: member['delay_ms'] += delay_delta

                if self.tb_sub_polarity.text().strip():
                    new_pol = int(self.tb_sub_polarity.text())
                    if new_pol in [1, -1]:
                        for member in self.sorgenti:
                            if member.get('group_id') == group_id: member['polarity'] = new_pol

            else:
                sub['x']=float(self.tb_sub_x.text())
                sub['y']=float(self.tb_sub_y.text())
                if not sub['param_locks'].get('gain', False): sub['gain_db']=float(self.tb_sub_gain_db.text())
                if not sub['param_locks'].get('delay', False): sub['delay_ms']=float(self.tb_sub_delay.text())
                if not sub['param_locks'].get('polarity', False): sub['polarity']=int(self.tb_sub_polarity.text())

            if not sub['param_locks'].get('angle', False): sub['angle']=np.radians(float(self.tb_sub_angle.text()))
            sub.update({'width':float(self.tb_sub_width.text()), 'depth':float(self.tb_sub_depth.text()), 'spl_rms':float(self.tb_sub_spl_rms.text())})
            
            for s_ in self.sorgenti:
                s_['gain_lin'] = 10**(s_['gain_db']/20.0)
                s_['pressure_val_at_1m_relative_to_pref'] = 10**(s_['spl_rms'] / 20.0)

            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()
        except ValueError: self.error_text_sub.setText("Errore: Dati non validi.")
        except Exception as e: self.error_text_sub.setText(f"Errore: {e}")

    def _update_max_group_id(self):
        self.max_group_id = max([s.get('group_id', 0) for s in self.sorgenti if s.get('group_id') is not None] or [0])
        self.next_group_id = self.max_group_id + 1
        
    def _update_group_ui_status(self):
        if not hasattr(self, 'btn_add_to_group') or not self.sorgenti:
            if hasattr(self, 'group_status_label'): self.group_status_label.setText("Nessun Sub")
            for w in [self.btn_create_new_group, self.btn_add_to_group, self.btn_remove_from_group, self.btn_ungroup_all]: w.setEnabled(False)
            return
            
        enable = 0 <= self.current_sub_idx < len(self.sorgenti)
        if not enable:
            self.group_status_label.setText("Nessun Sub Selezionato")
            for w in [self.btn_create_new_group, self.btn_add_to_group, self.btn_remove_from_group, self.btn_ungroup_all]: w.setEnabled(False)
            return

        sub = self.sorgenti[self.current_sub_idx]
        g_id = sub.get('group_id')
        is_master = sub.get('is_group_master')
        self.btn_create_new_group.setEnabled(g_id is None)
        self.btn_add_to_group.setEnabled(g_id is None)
        self.btn_remove_from_group.setEnabled(g_id is not None)
        self.btn_ungroup_all.setEnabled(g_id is not None)
        
        if g_id:
            self.group_status_label.setText(f"Sub selezionato: Gruppo ID {g_id}{' (M)' if is_master else ''}")
        else:
            self.group_status_label.setText("Sub selezionato: Nessun Gruppo")

    def create_new_group(self):
        if not self.sorgenti or self.current_sub_idx == -1: return
        sub = self.sorgenti[self.current_sub_idx]
        if sub.get('group_id'): self.error_text_grouping.setText("Sub già in un gruppo."); return
        self._update_max_group_id(); sub['group_id'] = self.next_group_id; sub['is_group_master'] = True
        self.status_bar.showMessage(f"Creato Gruppo ID {self.next_group_id}.", 3000)
        self.aggiorna_ui_sub_fields()
        
    def add_sub_to_existing_group(self):
        if not self.sorgenti or self.current_sub_idx == -1: return
        sub = self.sorgenti[self.current_sub_idx]
        if sub.get('group_id'): self.error_text_grouping.setText("Sub già in un gruppo."); return
        
        existing_groups = sorted(list(set(s['group_id'] for s in self.sorgenti if s['group_id'] is not None)))
        if not existing_groups:
            self.error_text_grouping.setText("Nessun gruppo esistente."); return
            
        group_id_str, ok = QInputDialog.getItem(self, "Aggiungi a Gruppo", "Seleziona ID del gruppo:", [str(g) for g in existing_groups], 0, False)
        if not ok or not group_id_str: return
        
        target_id = int(group_id_str)
        sub['group_id'] = target_id
        sub['is_group_master'] = False
        self.aggiorna_ui_sub_fields()
        
    def remove_sub_from_group(self):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]
        if not sub.get('group_id'): return
        
        gid = sub.get('group_id')
        is_master = sub.get('is_group_master')
        
        sub['group_id'] = None
        sub['is_group_master'] = False
        for p in sub['param_locks']: sub['param_locks'][p] = False

        if is_master:
            remaining_in_group = [s for s in self.sorgenti if s.get('group_id') == gid]
            if remaining_in_group:
                remaining_in_group[0]['is_group_master'] = True
            elif gid in self.lista_gruppi_array:
                del self.lista_gruppi_array[gid]

        self.aggiorna_ui_sub_fields()
        self.full_redraw(preserve_view=True)
        
    def ungroup_selected_sub_group(self):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]
        g_id = sub.get('group_id')
        if g_id is None: return

        if g_id in self.lista_gruppi_array:
            del self.lista_gruppi_array[g_id]

        for s in self.sorgenti:
            if s.get('group_id') == g_id:
                s['group_id'] = None; s['is_group_master'] = False
                for p in s['param_locks']: s['param_locks'][p] = False
        self.aggiorna_ui_sub_fields()
        self.full_redraw(preserve_view=True)
        

    def _update_array_ui(self):
        array_type = self.array_type_combo.currentText()
        is_none = array_type == "Nessuno"
        is_card_endfire = array_type in ["Coppia Cardioide (2 sub)", "Array End-Fire"]
        is_line_array = array_type == "Array Lineare (Steering Elettrico)"
        is_vortex = array_type == "Array Vortex"
        is_auto_spacing = self.array_auto_spacing_check.isChecked()

        self.array_freq_label.setVisible(not is_none)
        self.array_freq_input.setVisible(not is_none)
        self.array_auto_spacing_check.setVisible(is_card_endfire or is_line_array or is_vortex)
        
        show_frac_lambda = is_auto_spacing and (is_card_endfire or is_line_array or is_vortex)
        self.array_wavelength_fraction_label.setVisible(show_frac_lambda)
        self.array_wavelength_fraction_combo.setVisible(show_frac_lambda)
        
        self.array_spacing_label.setVisible(not is_none)
        self.array_spacing_input.setVisible(not is_none)
        
        if is_vortex:
            self.array_spacing_label.setText("Raggio Array (m):")
            self.array_spacing_input.setReadOnly(is_auto_spacing)
        elif is_card_endfire or is_line_array:
            self.array_spacing_label.setText("Spaziatura Fisica (m):")
            self.array_spacing_input.setReadOnly(is_auto_spacing)
        else:
            self.array_spacing_label.setText("Spaziatura/Raggio (m):")
            self.array_spacing_input.setReadOnly(False)
            
        self.array_elements_label.setVisible(not is_none)
        self.array_elements_input.setVisible(not is_none)
        self.array_elements_input.setEnabled(array_type not in ["Nessuno", "Coppia Cardioide (2 sub)"])
        if array_type == "Coppia Cardioide (2 sub)": self.array_elements_input.setText("2")
        
        should_show_start_angle = is_line_array or is_vortex or is_card_endfire
        self.array_start_angle_label.setVisible(should_show_start_angle)
        self.array_start_angle_input.setVisible(should_show_start_angle)
        
        if is_line_array or is_card_endfire:
            self.array_start_angle_label.setText("Orientamento (°):")
        elif is_vortex:
            self.array_start_angle_label.setText("Angolo Iniziale (°):")

        for w in [self.array_line_steering_angle_label, self.array_line_steering_angle_input, self.array_line_coverage_angle_label, self.array_line_coverage_angle_input]: w.setVisible(is_line_array)
        
        self.array_vortex_mode_label.setVisible(is_vortex)
        self.array_vortex_mode_input.setVisible(is_vortex)
        self.array_vortex_steering_angle_label.setVisible(is_vortex)
        self.array_vortex_steering_angle_input.setVisible(is_vortex)

    def on_array_type_change(self):
        self._update_array_ui()
        array_type = self.array_type_combo.currentText()
        info_text = "Il sub selezionato (se esiste) verrà SOSTITUITO dal nuovo array."
        if array_type == "Nessuno":
            self.array_info_label.setText("Seleziona un tipo di array da configurare.")
        else:
            self.array_info_label.setText(info_text)
        self.error_text_array_params.setText("")

    def on_auto_spacing_toggle(self):
        self._update_array_ui()

    def apply_array_configuration(self):
        ref_sub_idx = self.current_sub_idx if self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti) else -1
        ref_sub_model = self.sorgenti[ref_sub_idx] if ref_sub_idx != -1 else None
        
        center_x, center_y = (ref_sub_model['x'], ref_sub_model['y']) if ref_sub_model else (0,0)
        base_sub_params = {
            'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms,
            'gain_db': 0, 'polarity': 1, 'delay_ms': 0
        }
        if ref_sub_model:
            base_sub_params.update({k: v for k, v in ref_sub_model.items() if k in base_sub_params})

        self.error_text_array_params.setText("")
        array_type = self.array_type_combo.currentText()
        if array_type == "Nessuno": return
        
        try:
            c_sound = float(self.tb_velocita_suono.text())
            design_freq = float(self.array_freq_input.text())
            num_elements = int(self.array_elements_input.text())
            spacing_or_radius = float(self.array_spacing_input.text())
            start_angle_deg = float(self.array_start_angle_input.text()) if self.array_start_angle_input.isVisible() else 0.0

            if self.array_auto_spacing_check.isChecked() and array_type != "Nessuno":
                if design_freq <= 0: raise ValueError("La frequenza di design deve essere positiva.")
                wavelength = c_sound / design_freq
                spacing_or_radius = wavelength / 4.0 if self.array_wavelength_fraction_combo.currentText() == "λ/4" else wavelength / 2.0
                self.array_spacing_input.setText(f"{spacing_or_radius:.3f}")
            
            array_params = {}
            if array_type == "Coppia Cardioide (2 sub)":
                self.setup_cardioid_pair(center_x, center_y, spacing_or_radius, c_sound, start_angle_deg, base_sub_params, ref_sub_idx)
            elif array_type == "Array End-Fire":
                self.setup_end_fire_array(center_x, center_y, num_elements, spacing_or_radius, c_sound, start_angle_deg, base_sub_params, ref_sub_idx)
            elif array_type == "Array Lineare (Steering Elettrico)":
                steering_angle_deg = float(self.array_line_steering_angle_input.text())
                coverage_angle_deg = float(self.array_line_coverage_angle_input.text())
                array_params = {'steering_deg': steering_angle_deg, 'coverage_deg': coverage_angle_deg}
                self.setup_line_array_steered(center_x, center_y, num_elements, spacing_or_radius, start_angle_deg, steering_angle_deg, coverage_angle_deg, c_sound, base_sub_params, array_params, ref_sub_idx)
            elif array_type == "Array Vortex":
                vortex_mode = int(self.array_vortex_mode_input.text())
                steering_deg = float(self.array_vortex_steering_angle_input.text())
                array_params = {'steering_deg': steering_deg, 'mode': vortex_mode}
                self.setup_vortex_array(center_x, center_y, num_elements, spacing_or_radius, vortex_mode, design_freq, start_angle_deg, steering_deg, c_sound, base_sub_params, array_params, ref_sub_idx)

        except Exception as e:
            self.error_text_array_params.setText(f"Errore parametri array: {e}")
            import traceback
            traceback.print_exc()


    
    def select_prev_target_area(self):
        if not self.lista_target_areas: self.current_target_area_idx = -1
        else: self.current_target_area_idx = (self.current_target_area_idx - 1 + len(self.lista_target_areas)) % len(self.lista_target_areas)
        self.update_ui_for_selected_target_area()
    def select_next_target_area(self):
        if not self.lista_target_areas: self.current_target_area_idx = -1
        else: self.current_target_area_idx = (self.current_target_area_idx + 1) % len(self.lista_target_areas)
        self.update_ui_for_selected_target_area()

    def _add_vtx_to_current_target_area(self):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            area = self.lista_target_areas[self.current_target_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area['punti'].append([new_x, new_y])
            self.update_ui_for_selected_target_area()
        else:
            self.error_text_target_area_mgmt.setText("Selezionare prima un'area target o crearne una nuova.")

    def _add_new_area_data(self, area_list, default_vertices, base_name, next_id_attr, activate=True):
        new_id = getattr(self, next_id_attr)
        snapped_default_vertices = [[self.snap_to_grid(p[0]), self.snap_to_grid(p[1])] for p in default_vertices]
        area_data = { 'id': new_id, 'nome': f"{base_name} {new_id}", 'punti': snapped_default_vertices, 'active': activate, 'plots': [] }
        area_list.append(area_data); setattr(self, next_id_attr, new_id + 1)
        return len(area_list) - 1
    def _get_area_center_and_default_size(self):
        xlims, ylims = self.ax.get_xlim(), self.ax.get_ylim(); cx, cy = np.mean(xlims), np.mean(ylims)
        size = min(xlims[1]-xlims[0], ylims[1]-ylims[0]) / 4.0
        return cx, cy, max(1.0, size)
    def add_new_target_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size(); hs = size/2.0
        default_verts = [ [cx-hs, cy-hs], [cx+hs, cy-hs], [cx+hs, cy+hs], [cx-hs, cy+hs] ]
        self.current_target_area_idx = self._add_new_area_data(self.lista_target_areas, default_verts, "Target", 'next_target_area_id')
        self.update_ui_for_selected_target_area()
    def remove_selected_target_area_ui(self):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            self.lista_target_areas.pop(self.current_target_area_idx)
            if not self.lista_target_areas: self.current_target_area_idx = -1
            elif self.current_target_area_idx >= len(self.lista_target_areas): self.current_target_area_idx = len(self.lista_target_areas) - 1
            self.update_ui_for_selected_target_area()
    def toggle_selected_target_area_active(self, checked):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas): self.lista_target_areas[self.current_target_area_idx]['active'] = checked; self.update_ui_for_selected_target_area()
    
    def update_ui_for_selected_target_area(self):
        is_valid_idx = 0 <= self.current_target_area_idx < len(self.lista_target_areas)
        self.btn_add_target_vtx.setEnabled(is_valid_idx)

        for w in [self.btn_prev_target_area, self.btn_next_target_area, self.btn_remove_selected_target_area, self.check_activate_selected_target_area, self.target_vtx_list_widget]: w.setEnabled(is_valid_idx)
        
        self.target_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_target_areas[self.current_target_area_idx]
            self.label_current_target_area.setText(f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})")
            try: self.check_activate_selected_target_area.toggled.disconnect()
            except TypeError: pass
            self.check_activate_selected_target_area.setChecked(area['active']); self.check_activate_selected_target_area.toggled.connect(self.toggle_selected_target_area_active)
            for i, p in enumerate(area['punti']):
                self.target_vtx_list_widget.addItem(f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
        else:
            self.label_current_target_area.setText("Nessuna Area Target")
        
        self.on_target_vtx_selection_change()
        self.full_redraw(preserve_view=True)
        self.update_optim_freq_fields_visibility()
        
    def on_target_vtx_selection_change(self):
        is_valid_area = 0 <= self.current_target_area_idx < len(self.lista_target_areas)
        selected_items = self.target_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)
        
        for w in [self.tb_target_vtx_x, self.tb_target_vtx_y, self.btn_update_target_vtx]:
            w.setEnabled(can_edit)
            
        if can_edit:
            vtx_idx = self.target_vtx_list_widget.currentRow()
            vtx = self.lista_target_areas[self.current_target_area_idx]['punti'][vtx_idx]
            self.tb_target_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_target_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_target_vtx_x.clear()
            self.tb_target_vtx_y.clear()

    def on_update_selected_target_vertex(self):
        if not (0 <= self.current_target_area_idx < len(self.lista_target_areas)): return
        vtx_idx = self.target_vtx_list_widget.currentRow()
        if vtx_idx < 0: return
        
        try:
            x = float(self.tb_target_vtx_x.text())
            y = float(self.tb_target_vtx_y.text())
            self.lista_target_areas[self.current_target_area_idx]['punti'][vtx_idx] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.update_ui_for_selected_target_area()
        except ValueError:
            self.error_text_target_area_mgmt.setText("Coordinate non valide.")

    def select_prev_avoidance_area(self):
        if not self.lista_avoidance_areas: self.current_avoidance_area_idx = -1
        else: self.current_avoidance_area_idx = (self.current_avoidance_area_idx - 1 + len(self.lista_avoidance_areas)) % len(self.lista_avoidance_areas)
        self.update_ui_for_selected_avoidance_area()
    def select_next_avoidance_area(self):
        if not self.lista_avoidance_areas: self.current_avoidance_area_idx = -1
        else: self.current_avoidance_area_idx = (self.current_avoidance_area_idx + 1) % len(self.lista_avoidance_areas)
        self.update_ui_for_selected_avoidance_area()
    def add_new_avoidance_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size(); hs = size/2.0 * 0.8
        default_verts = [ [cx-hs, cy-hs], [cx+hs, cy-hs], [cx+hs, cy+hs], [cx-hs, cy+hs] ]
        self.current_avoidance_area_idx = self._add_new_area_data(self.lista_avoidance_areas, default_verts, "Evitamento", 'next_avoidance_area_id')
        self.update_ui_for_selected_avoidance_area()
    def remove_selected_avoidance_area_ui(self):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            self.lista_avoidance_areas.pop(self.current_avoidance_area_idx)
            if not self.lista_avoidance_areas: self.current_avoidance_area_idx = -1
            elif self.current_avoidance_area_idx >= len(self.lista_avoidance_areas): self.current_avoidance_area_idx = len(self.lista_avoidance_areas) - 1
            self.update_ui_for_selected_avoidance_area()
    def toggle_selected_avoidance_area_active(self, checked):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas): self.lista_avoidance_areas[self.current_avoidance_area_idx]['active'] = checked; self.update_ui_for_selected_avoidance_area()
    
    def _add_vtx_to_current_avoidance_area(self):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            area = self.lista_avoidance_areas[self.current_avoidance_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area['punti'].append([new_x, new_y])
            self.update_ui_for_selected_avoidance_area()
        else:
            self.error_text_avoid_area_mgmt.setText("Selezionare prima un'area di evitamento o crearne una nuova.")


    def update_ui_for_selected_avoidance_area(self):
        is_valid_idx = 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)
        self.btn_add_avoid_vtx.setEnabled(is_valid_idx)

        for w in [self.btn_prev_avoid_area, self.btn_next_avoid_area, self.btn_remove_selected_avoid_area, self.check_activate_selected_avoid_area, self.avoid_vtx_list_widget]: w.setEnabled(is_valid_idx)
        
        self.avoid_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_avoidance_areas[self.current_avoidance_area_idx]
            self.label_current_avoid_area.setText(f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})")
            try: self.check_activate_selected_avoid_area.toggled.disconnect()
            except TypeError: pass
            self.check_activate_selected_avoid_area.setChecked(area['active']); self.check_activate_selected_avoid_area.toggled.connect(self.toggle_selected_avoidance_area_active)
            for i, p in enumerate(area['punti']):
                self.avoid_vtx_list_widget.addItem(f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
        else:
            self.label_current_avoid_area.setText("Nessuna Area di Evitamento")
        
        self.on_avoid_vtx_selection_change()
        self.full_redraw(preserve_view=True)
        self.update_optim_freq_fields_visibility()

    def on_avoid_vtx_selection_change(self):
        is_valid_area = 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)
        selected_items = self.avoid_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)
        
        for w in [self.tb_avoid_vtx_x, self.tb_avoid_vtx_y, self.btn_update_avoid_vtx]:
            w.setEnabled(can_edit)
            
        if can_edit:
            vtx_idx = self.avoid_vtx_list_widget.currentRow()
            vtx = self.lista_avoidance_areas[self.current_avoidance_area_idx]['punti'][vtx_idx]
            self.tb_avoid_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_avoid_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_avoid_vtx_x.clear()
            self.tb_avoid_vtx_y.clear()

    def on_update_selected_avoid_vertex(self):
        if not (0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)): return
        vtx_idx = self.avoid_vtx_list_widget.currentRow()
        if vtx_idx < 0: return
        
        try:
            x = float(self.tb_avoid_vtx_x.text())
            y = float(self.tb_avoid_vtx_y.text())
            self.lista_avoidance_areas[self.current_avoidance_area_idx]['punti'][vtx_idx] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.update_ui_for_selected_avoidance_area()
        except ValueError:
            self.error_text_avoid_area_mgmt.setText("Coordinate non valide.")

    def update_grid_snap_params(self, *args):
        self.grid_snap_enabled = self.check_grid_snap_enabled.isChecked(); self.grid_show_enabled = self.check_show_grid.isChecked()
        try: self.grid_snap_spacing = float(self.tb_grid_snap_spacing.text())
        except: self.grid_snap_spacing = 0.25
        self.full_redraw(preserve_view=True)
        
    def get_slider_freq_val(self): return self.slider_freq.value()
    def on_freq_change_ui_qt(self, value): self.label_slider_freq_val.setText(f"{value} Hz"); self.trigger_spl_map_recalculation()
    def trigger_spl_map_recalculation(self, force_redraw=False, fit_view=False):
        if self.check_auto_spl_update.isChecked() or force_redraw:
            self.full_redraw(preserve_view=not fit_view)

    def full_redraw(self, preserve_view=False):
        self.visualizza_mappatura_spl(self.get_slider_freq_val(), preserve_view)

    def on_press_mpl(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        self.status_bar.clearMessage()
        
        for area_list, type_prefix in [(self.lista_target_areas, 'target'), (self.lista_avoidance_areas, 'avoid')]:
            for area_idx, area_data in enumerate(area_list):
                if not area_data.get('active', False): continue
                for vtx_idx, plot_artist in enumerate(area_data.get('plots', [])): 
                    if plot_artist and plot_artist.contains(event)[0]:
                        if type_prefix == 'target':
                            if self.current_target_area_idx != area_idx:
                                self.current_target_area_idx = area_idx
                                self.update_ui_for_selected_target_area()
                            self.target_vtx_list_widget.setCurrentRow(vtx_idx)
                        elif type_prefix == 'avoid':
                            if self.current_avoidance_area_idx != area_idx:
                                self.current_avoidance_area_idx = area_idx
                                self.update_ui_for_selected_avoidance_area()
                            self.avoid_vtx_list_widget.setCurrentRow(vtx_idx)

                        self.drag_object = (f'{type_prefix}_vtx', area_idx, vtx_idx)
                        self.original_mouse_pos = (event.xdata, event.ydata)
                        self.original_object_pos = tuple(area_data['punti'][vtx_idx]) 
                        return

        for vtx_idx, vtx_data in enumerate(self.punti_stanza):
            if vtx_data.get('plot') and vtx_data['plot'].contains(event)[0]:
                self.drag_object = ('stanza_vtx', vtx_idx)
                self.original_mouse_pos = (event.xdata, event.ydata)
                self.original_object_pos = tuple(vtx_data['pos'])
                self.selected_stanza_vtx_idx = vtx_idx
                self.update_stanza_vtx_editor()
                return

        for i in reversed(range(len(self.sorgenti))):
            sub = self.sorgenti[i]
            if sub.get('arrow_artist') and sub['arrow_artist'].contains(event)[0]:
                if sub.get('param_locks', {}).get('angle', False):
                    self.status_bar.showMessage(f"Angolo Sub ID:{sub.get('id', i+1)} bloccato.", 2000)
                    return
                self.current_sub_idx = i
                self.aggiorna_ui_sub_fields()
                self.original_mouse_pos = (event.xdata, event.ydata)
                drag_type = 'group_rotate' if sub.get('group_id') is not None else 'sub_rotate'
                self.original_object_angle = sub['angle']
                self.drag_object = (drag_type, i)
                if 'group' in drag_type: self.original_group_states = self._get_group_states(sub.get('group_id'))
                return
            
            if sub.get('rect_artist') and sub['rect_artist'].contains(event)[0]:
                self.current_sub_idx = i
                self.aggiorna_ui_sub_fields()
                self.original_mouse_pos = (event.xdata, event.ydata)
                drag_type = 'group_pos' if sub.get('group_id') is not None else 'sub_pos'
                self.original_object_pos = (sub['x'], sub['y'])
                self.drag_object = (drag_type, i)
                if 'group' in drag_type: self.original_group_states = self._get_group_states(sub.get('group_id'))
                return
                                
        self.drag_object = None

    def on_motion_mpl(self, event):
        # Always update SPL display if mouse is over axes, regardless of drag state
        # self.on_mouse_move_for_spl_display(event) 

        if self.drag_object is None or event.inaxes != self.ax or event.xdata is None: 
            return

        dx = event.xdata - self.original_mouse_pos[0]; dy = event.ydata - self.original_mouse_pos[1]
        obj_type = self.drag_object[0]
        
        redraw_needed = True

        if obj_type == 'sub_pos':
            main_idx = self.drag_object[1]
            self.sorgenti[main_idx]['x'] = self.snap_to_grid(self.original_object_pos[0] + dx)
            self.sorgenti[main_idx]['y'] = self.snap_to_grid(self.original_object_pos[1] + dy)
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'group_pos':
            for s_state in self.original_group_states:
                orig_x, orig_y = s_state['original_pos']
                self.sorgenti[s_state['sub_idx']]['x'] = self.snap_to_grid(orig_x + dx)
                self.sorgenti[s_state['sub_idx']]['y'] = self.snap_to_grid(orig_y + dy)
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'sub_rotate':
            main_idx = self.drag_object[1]
            sub = self.sorgenti[main_idx]
            self.sorgenti[main_idx]['angle'] = np.arctan2(event.xdata - sub['x'], event.ydata - sub['y'])
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'group_rotate':
            group_center = self.original_group_states[0]['group_center']
            initial_mouse_angle = np.arctan2(self.original_mouse_pos[1] - group_center[1], self.original_mouse_pos[0] - group_center[0])
            current_mouse_angle = np.arctan2(event.ydata - group_center[1], event.xdata - group_center[0])
            angle_delta = current_mouse_angle - initial_mouse_angle
            for s_state in self.original_group_states:
                sub_idx, orig_rel_pos, orig_angle = s_state['sub_idx'], s_state['rel_pos'], s_state['original_angle']
                new_rel_x = orig_rel_pos[0] * np.cos(angle_delta) - orig_rel_pos[1] * np.sin(angle_delta)
                new_rel_y = orig_rel_pos[0] * np.sin(angle_delta) + orig_rel_pos[1] * np.cos(angle_delta)
                self.sorgenti[sub_idx]['x'] = self.snap_to_grid(group_center[0] + new_rel_x)
                self.sorgenti[sub_idx]['y'] = self.snap_to_grid(group_center[1] + new_rel_y)
                self.sorgenti[sub_idx]['angle'] = (orig_angle + angle_delta) % (2 * np.pi)
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'stanza_vtx':
            main_idx = self.drag_object[1]
            self.punti_stanza[main_idx]['pos'][0] = self.snap_to_grid(self.original_object_pos[0] + dx)
            self.punti_stanza[main_idx]['pos'][1] = self.snap_to_grid(self.original_object_pos[1] + dy)
            self.update_stanza_vtx_editor()
        elif obj_type in ['target_vtx', 'avoid_vtx']:
            area_type_prefix, area_idx, vtx_idx = self.drag_object[0].split('_')[0], self.drag_object[1], self.drag_object[2]
            area_list = self.lista_target_areas if 'target' in area_type_prefix else self.lista_avoidance_areas
            area_list[area_idx]['punti'][vtx_idx] = [self.snap_to_grid(self.original_object_pos[0] + dx), self.snap_to_grid(self.original_object_pos[1] + dy)]
            if area_type_prefix == 'target':
                self.update_ui_for_selected_target_area()
            else: 
                self.update_ui_for_selected_avoidance_area()
        else:
            redraw_needed = False

        if redraw_needed:
            self.full_redraw(preserve_view=True)

    def on_release_mpl(self, event):
        if self.drag_object:
            self.status_bar.showMessage(f"Rilasciato oggetto.", 2000)
            is_room_drag = self.drag_object and self.drag_object[0] == 'stanza_vtx'
            self.drag_object = None
            self.original_group_states = []
            self.trigger_spl_map_recalculation(force_redraw=True, fit_view=is_room_drag)

    def _get_group_states(self, group_id):
        states = [];
        if group_id is None: return states
        members = [s for s in self.sorgenti if s.get('group_id') == group_id]
        if not members: return states
        center_x = np.mean([s['x'] for s in members]); center_y = np.mean([s['y'] for s in members])
        for s in members:
            states.append({'sub_idx': self.sorgenti.index(s), 'original_pos': (s['x'], s['y']), 'original_angle': s['angle'], 'rel_pos': (s['x'] - center_x, s['y'] - center_y), 'group_center': (center_x, center_y)})
        return states

    def visualizza_mappatura_spl(self, frequenza, preserve_view=False):
        current_xlim, current_ylim = None, None
        if preserve_view and self.ax.lines:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
        
        self.ax.cla()

        # --- START: FIX FOR DARK THEME ---
        # Set a visible background color for the figure and the axes area
        self.plot_canvas.figure.set_facecolor("#323232")  # Dark gray for the outer area
        self.ax.set_facecolor("#404040")                 # Lighter gray for the plot area

        # Make the axes lines (spines) and labels white so they are visible
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        # --- END: FIX FOR DARK THEME ---
        
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.set_xlabel("X (m)", loc='right')
        self.ax.set_ylabel("Y (m)", loc='top')
        
        if hasattr(self, '_cax_for_colorbar_spl') and self._cax_for_colorbar_spl:
            try: self.plot_canvas.figure.delaxes(self._cax_for_colorbar_spl)
            except (KeyError, AttributeError): pass
        self._cax_for_colorbar_spl = None

        self.plot_canvas.figure.subplots_adjust(right=0.92)
        
        room_points = [p['pos'] for p in self.punti_stanza]
        if room_points and len(room_points) >= 3 and self.sorgenti:
            try:
                min_spl=float(self.tb_spl_min.text()); max_spl=float(self.tb_spl_max.text())
                c_val=float(self.tb_velocita_suono.text()); grid_res=float(self.tb_grid_res_spl.text())
                if not(c_val > 0 and grid_res > 0 and min_spl < max_spl): raise ValueError()
                
                min_x_room=min(p[0] for p in room_points); max_x_room=max(p[0] for p in room_points)
                min_y_room=min(p[1] for p in room_points); max_y_room=max(p[1] for p in room_points)
                x=np.arange(min_x_room, max_x_room + grid_res, grid_res); y=np.arange(min_y_room, max_y_room + grid_res, grid_res)
                
                if len(x)>=2 and len(y)>=2:
                    X, Y = np.meshgrid(x,y)
                    SPL_map_plot = np.full(X.shape, np.nan)
                    room_mask = Path(room_points).contains_points(np.vstack((X.ravel(),Y.ravel())).T).reshape(X.shape)
                    
                    if np.any(room_mask):
                        points_to_calc_x = X[room_mask]
                        points_to_calc_y = Y[room_mask]
                        spl_values = calculate_spl_vectorized(points_to_calc_x, points_to_calc_y, frequenza, c_val, self.sorgenti)
                        self.current_spl_map = np.copy(SPL_map_plot) # Store the current SPL map for mouse hovering
                        self.current_spl_map[room_mask] = spl_values

                        masked_spl_map = np.ma.masked_where(~room_mask, self.current_spl_map)
                        if masked_spl_map.count() > 0:
                            contour = self.ax.contourf(X, Y, masked_spl_map, levels=100, cmap='jet', alpha=0.75, extend='both', vmin=min_spl, vmax=max_spl, zorder=0.3)
                            self.plot_canvas.figure.subplots_adjust(right=0.85)
                            self._cax_for_colorbar_spl = self.plot_canvas.figure.add_axes([0.87, 0.15, 0.03, 0.7])
                            self.plot_canvas.figure.colorbar(contour, cax=self._cax_for_colorbar_spl, label="SPL (dB)")
            except Exception as e:
                print(f"Errore calcolo/disegno mappa SPL: {e}")
                self.current_spl_map = None # Reset map on error

        self.disegna_elementi_statici_senza_spl()
        
        self.ax.set_aspect('equal', adjustable='box')
        if preserve_view and current_xlim is not None:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        else:
            self.auto_fit_view_to_room()
            
        self.plot_canvas.canvas.draw_idle()
    

    def update_optim_freq_fields_visibility(self, *args):
        is_copertura = self.radio_copertura.isChecked()
        for w in [self.label_opt_freq_single_widget, self.tb_opt_freq_single]: w.setVisible(is_copertura)
        for w in [self.label_opt_freq_min_widget, self.tb_opt_freq_min, self.label_opt_freq_max_widget, self.tb_opt_freq_max, self.label_opt_n_freq_widget, self.tb_opt_n_freq]: w.setVisible(not is_copertura)
        
        has_active_target_areas = len(self.get_active_areas_points(self.lista_target_areas)) > 0
        has_active_avoidance_areas = len(self.get_active_areas_points(self.lista_avoidance_areas)) > 0
        
        show_balance_slider = has_active_target_areas and has_active_avoidance_areas

        self.label_balance_slider.setVisible(show_balance_slider)
        self.slider_balance.setVisible(show_balance_slider)
        self.label_balance_value.setVisible(show_balance_slider)

    def on_mouse_move_for_spl_display(self, event):
        if event.inaxes != self.ax or self.current_spl_map is None or event.xdata is None or event.ydata is None:
            self.status_bar.showMessage("Muovi il mouse sul grafico per visualizzare l'SPL.", 0)
            return

        try:
            x_coord = event.xdata
            y_coord = event.ydata
            
            if not self.punti_stanza:
                self.status_bar.showMessage("Definisci la stanza per visualizzare l'SPL al mouse.", 0)
                return

            min_x_room=min(p['pos'][0] for p in self.punti_stanza); max_x_room=max(p['pos'][0] for p in self.punti_stanza)
            min_y_room=min(p['pos'][1] for p in self.punti_stanza); max_y_room=max(p['pos'][1] for p in self.punti_stanza)
            
            grid_res_text = self.tb_grid_res_spl.text()
            if not grid_res_text:
                self.status_bar.showMessage("Risoluzione griglia SPL non definita. Impossibile calcolare SPL al mouse.", 0)
                return
            grid_res = float(grid_res_text)

            if grid_res <= 0:
                self.status_bar.showMessage("Risoluzione griglia SPL non valida. Impossibile calcolare SPL al mouse.", 0)
                return
            
            col_idx = int(np.floor((x_coord - min_x_room) / grid_res))
            row_idx = int(np.floor((y_coord - min_y_room) / grid_res))
            
            if 0 <= row_idx < self.current_spl_map.shape[0] and 0 <= col_idx < self.current_spl_map.shape[1]:
                spl_val = self.current_spl_map[row_idx, col_idx]
                if not np.isnan(spl_val):
                    self.status_bar.showMessage(f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): {spl_val:.1f} dB", 0)
                else:
                    self.status_bar.showMessage(f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): Fuori Area Stanza", 0)
            else:
                self.status_bar.showMessage(f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): Fuori Limiti di Plot", 0)

        except ValueError:
            self.status_bar.showMessage("Dati non numerici per calcolo SPL al mouse.", 0)
        except Exception as e:
            self.status_bar.showMessage(f"Errore: {e}", 0)


    def unlock_dsp_for_optimization(self):
        if not self.sorgenti: self.status_bar.showMessage("Nessun subwoofer da sbloccare.", 3000); return
        unlocked_count = 0
        for sub in self.sorgenti:
            if sub.get('group_id') is None:
                for param in ['delay', 'gain', 'polarity', 'angle']:
                    if sub['param_locks'].get(param, False):
                        sub['param_locks'][param] = False
                        unlocked_count += 1
        self.aggiorna_ui_sub_fields(); self.status_bar.showMessage(f"Parametri DSP sbloccati.", 3000)

    def avvia_ottimizzazione_ui_qt(self):
        if self.optimization_thread is not None: return
        if not self.sorgenti: self.status_text_optim.setText("Aggiungere almeno un subwoofer."); return
        
        room_points_list = [p['pos'] for p in self.punti_stanza]
        if len(room_points_list) < 3: self.status_text_optim.setText("Definire una stanza valida."); return

        active_targets = self.get_active_areas_points(self.lista_target_areas)
        active_avoidances = self.get_active_areas_points(self.lista_avoidance_areas)
        
        if not active_targets and not active_avoidances:
            self.status_text_optim.setText("Attivare almeno un'area target o di evitamento per l'ottimizzazione."); 
            return

        try:
            criterion = self.radio_btn_group_crit.checkedButton().text(); pop_s=int(self.tb_opt_pop_size.text()); gens=int(self.tb_opt_generations.text()); 
            
            max_spl_avoid=float(self.tb_max_spl_avoid.text())
            target_min_spl_desired_val = float(self.tb_target_min_spl_desired.text())
            
            balance_target_avoidance_val = self.slider_balance.value()
            
            c_val=float(self.tb_velocita_suono.text()); grid_res=float(self.tb_grid_res_spl.text())
            optim_f_s, optim_f_min, optim_f_max, optim_n_f = None, None, None, None
            self.last_optim_criterion = criterion

            if criterion == 'Copertura SPL':
                optim_f_s = float(self.tb_opt_freq_single.text())
                self.last_optim_freq_s = optim_f_s
            else:
                optim_f_min=float(self.tb_opt_freq_min.text())
                optim_f_max=float(self.tb_opt_freq_max.text())
                optim_n_f=int(self.tb_opt_n_freq.text())
                self.last_optim_freq_min = optim_f_min
                self.last_optim_freq_max = optim_f_max
                if optim_n_f < 2:
                    self.status_text_optim.setText("Per 'Omogeneità', usare almeno 2 punti frequenza.")
                    return
        except (ValueError, AttributeError) as e: self.status_text_optim.setText(f"Errore parametri ottimizzazione: {e}"); return
        
        self.optimization_thread = QThread()
        self.optimization_worker = OptimizationWorker(criterion, optim_f_s, optim_f_min, optim_f_max, optim_n_f, pop_s, gens, c_val, grid_res, 
            room_points_list, active_targets, active_avoidances,
            max_spl_avoid, target_min_spl_desired_val, balance_target_avoidance_val,
            [s.copy() for s in self.sorgenti], [s['param_locks'].copy() for s in self.sorgenti])
        
        self.optimization_worker.moveToThread(self.optimization_thread)

        self.optimization_worker.status_update.connect(self.update_optim_status_text)
        self.optimization_worker.finished.connect(self.optimization_thread.quit)
        self.optimization_worker.finished.connect(self.handle_optim_finished)
        self.optimization_worker.finished.connect(self.optimization_worker.deleteLater)
        self.optimization_thread.finished.connect(self.optimization_thread.deleteLater)
        self.optimization_thread.finished.connect(self.on_optim_thread_finished)
        
        self.optimization_thread.started.connect(self.optimization_worker.run)
        self.optimization_thread.start()
        
        self.btn_optimize_widget.setEnabled(False)
        self.btn_stop_optimize_widget.setEnabled(True)

    def stop_ottimizzazione_ui_qt(self):
        if self.optimization_worker: self.optimization_worker.request_stop()
        
    def handle_optim_finished(self, best_solution):
        self.btn_optimize_widget.setEnabled(True)
        self.btn_stop_optimize_widget.setEnabled(False)

        new_freq = None
        if self.last_optim_criterion == 'Copertura SPL':
            if self.last_optim_freq_s is not None:
                new_freq = self.last_optim_freq_s
        elif self.last_optim_criterion == 'Omogeneità SPL':
            if self.last_optim_freq_min is not None and self.last_optim_freq_max is not None:
            # Usiamo la media aritmetica del range come frequenza di visualizzazione
                new_freq = (self.last_optim_freq_min + self.last_optim_freq_max) / 2.0

        if new_freq is not None:
            # Assicuriamoci che il valore sia nei limiti dello slider per evitare errori
            slider_min = self.slider_freq.minimum()
            slider_max = self.slider_freq.maximum()
            clamped_freq = max(slider_min, min(slider_max, new_freq))
        
        # Impostiamo il valore dello slider. Questo aggiornerà automaticamente
        # anche l'etichetta del valore (es. "80 Hz")
        self.slider_freq.setValue(int(clamped_freq))
        
        if best_solution:
            for i, sub_dsp in enumerate(best_solution):
                current_sub = self.sorgenti[i]
                current_sub['delay_ms'] = sub_dsp['delay_ms']
                current_sub['gain_db'] = sub_dsp['gain_db']
                current_sub['gain_lin'] = sub_dsp['gain_lin']
                current_sub['polarity'] = sub_dsp['polarity']
                if 'angle' in sub_dsp and not current_sub['param_locks'].get('angle', False):
                    current_sub['angle'] = sub_dsp['angle']

            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)

    def on_optim_thread_finished(self):
        self.optimization_thread = None
        self.optimization_worker = None
        self.status_bar.showMessage("Thread di ottimizzazione terminato.", 3000)

    def update_optim_status_text(self, message):
        if hasattr(self, 'status_text_optim'):
            self.status_text_optim.setText(message)
        QApplication.processEvents()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        main_win = SubwooferSimApp()
        main_win.show()
    except Exception as e:
        print(f"ERRORE DURANTE L'INIZIALIZZAZIONE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(app.exec())
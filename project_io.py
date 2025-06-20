import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QFileDialog, QMessageBox


class ProjectIOMixin:
    """Mixin providing load/save routines for projects."""

    def save_project_to_excel(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Salva Progetto Completo", "", "File Excel (*.xlsx)")
        if not filepath:
            return
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if self.sorgenti:
                    sub_data_to_save = []
                    for sub in self.sorgenti:
                        sub_data_to_save.append(
                            {
                                'ID': sub.get('id'),
                                'X (m)': sub.get('x'),
                                'Y (m)': sub.get('y'),
                                'Angolo (°)': np.degrees(sub.get('angle')),
                                'Gain (dB)': sub.get('gain_db'),
                                'Polarità': sub.get('polarity'),
                                'Delay (ms)': sub.get('delay_ms'),
                                'Larghezza (m)': sub.get('width'),
                                'Profondità (m)': sub.get('depth'),
                                'SPL (dB)': sub.get('spl_rms'),
                                'group_id': sub.get('group_id'),
                            }
                        )
                    df_subs = pd.DataFrame(sub_data_to_save)
                    df_subs.to_excel(writer, sheet_name='Subwoofers', index=False)
                if self.punti_stanza:
                    room_verts = [p['pos'] for p in self.punti_stanza]
                    df_room = pd.DataFrame(room_verts, columns=['X', 'Y'])
                    df_room.to_excel(writer, sheet_name='Stanza', index=False)
                if self.lista_target_areas:
                    target_areas_data = []
                    for area in self.lista_target_areas:
                        for vtx in area['punti']:
                            target_areas_data.append(
                                {
                                    'Area_ID': area['id'],
                                    'Nome': area['nome'],
                                    'Attiva': area['active'],
                                    'Vertice_X': vtx[0],
                                    'Vertice_Y': vtx[1],
                                }
                            )
                    df_target = pd.DataFrame(target_areas_data)
                    df_target.to_excel(writer, sheet_name='Aree_Target', index=False)
                if self.lista_avoidance_areas:
                    avoid_areas_data = []
                    for area in self.lista_avoidance_areas:
                        for vtx in area['punti']:
                            avoid_areas_data.append(
                                {
                                    'Area_ID': area['id'],
                                    'Nome': area['nome'],
                                    'Attiva': area['active'],
                                    'Vertice_X': vtx[0],
                                    'Vertice_Y': vtx[1],
                                }
                            )
                    df_avoid = pd.DataFrame(avoid_areas_data)
                    df_avoid.to_excel(writer, sheet_name='Aree_Evitamento', index=False)
            self.status_bar.showMessage(f"Progetto completo salvato in {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Errore di Salvataggio", f"Impossibile salvare il file di progetto:\n{e}")

    def load_project_from_excel(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Carica Progetto Completo", "", "File Excel (*.xlsx)")
        if not filepath:
            return
        try:
            self.sorgenti.clear()
            self.punti_stanza.clear()
            self.lista_target_areas.clear()
            self.lista_avoidance_areas.clear()
            self.lista_gruppi_array.clear()
            self.current_sub_idx = -1
            self.next_sub_id = 1
            self.next_group_id = 1
            self.next_target_area_id = 1
            self.next_avoidance_area_id = 1
            try:
                df_subs = pd.read_excel(filepath, sheet_name='Subwoofers')
                for _, row in df_subs.iterrows():
                    config = {
                        'id': int(row['ID']),
                        'x': row['X (m)'],
                        'y': row['Y (m)'],
                        'angle': np.radians(row['Angolo (°)']),
                        'gain_db': row['Gain (dB)'],
                        'polarity': row['Polarità'],
                        'delay_ms': row['Delay (ms)'],
                        'width': row.get('Larghezza (m)', self.global_sub_width),
                        'depth': row.get('Profondità (m)', self.global_sub_depth),
                        'spl_rms': row.get('SPL (dB)', self.global_sub_spl_rms),
                        'group_id': int(row['group_id']) if pd.notna(row['group_id']) else None,
                    }
                    self.add_subwoofer(specific_config=config, redraw=False)
                if 'ID' in df_subs.columns and pd.notna(df_subs['ID'].max()):
                    self.next_sub_id = int(df_subs['ID'].max()) + 1
            except Exception:
                print("Foglio 'Subwoofers' non trovato o errore nel caricarlo. Ignorato.")
            try:
                df_room = pd.read_excel(filepath, sheet_name='Stanza')
                for _, row in df_room.iterrows():
                    self.punti_stanza.append({'pos': [row['X'], row['Y']], 'plot': None})
            except Exception:
                print("Foglio 'Stanza' non trovato o errore nel caricarlo. Ignorato.")
            try:
                df_target = pd.read_excel(filepath, sheet_name='Aree_Target')
                if not df_target.empty:
                    for area_id, group in df_target.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id),
                            'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]),
                            'punti': punti,
                            'plots': [],
                        }
                        self.lista_target_areas.append(area_data)
                    if pd.notna(df_target['Area_ID'].max()):
                        self.next_target_area_id = int(df_target['Area_ID'].max()) + 1
            except Exception:
                print("Foglio 'Aree_Target' non trovato o errore nel caricarlo. Ignorato.")
            try:
                df_avoid = pd.read_excel(filepath, sheet_name='Aree_Evitamento')
                if not df_avoid.empty:
                    for area_id, group in df_avoid.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id),
                            'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]),
                            'punti': punti,
                            'plots': [],
                        }
                        self.lista_avoidance_areas.append(area_data)
                    if pd.notna(df_avoid['Area_ID'].max()):
                        self.next_avoidance_area_id = int(df_avoid['Area_ID'].max()) + 1
            except Exception:
                print("Foglio 'Aree_Evitamento' non trovato o errore nel caricarlo. Ignorato.")
            all_group_ids = {s['group_id'] for s in self.sorgenti if s['group_id'] is not None}
            for gid in all_group_ids:
                group_members = [s for s in self.sorgenti if s.get('group_id') == gid]
                if group_members:
                    group_members[0]['is_group_master'] = True
            if self.sorgenti:
                self.current_sub_idx = 0
            self.aggiorna_ui_sub_fields()
            self.update_ui_for_selected_target_area()
            self.update_ui_for_selected_avoidance_area()
            self.full_redraw()
            self.status_bar.showMessage(f"Progetto completo caricato da {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Errore di Caricamento", f"Impossibile caricare il file di progetto:\n{e}")
            import traceback
            traceback.print_exc()

